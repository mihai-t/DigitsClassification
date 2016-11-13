package ro.ubbcluj.cs.dl;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ro.ubbcluj.cs.io.DigitImageLoadingService;
import ro.ubbcluj.cs.io.TrainingExample;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 * @author Mihai Teletin
 */
public class Dl4jApp {

    /**
     * File names
     */
    private static final String TRAINING_LABELS = "/train/trn60k-labels-ubyte";
    private static final String TRAIN_FEATURES = "/train/trn60k-images-ubyte";
    private static final String TESTING_LABELS = "/test/tst10k-labels-ubyte";
    private static final String TESTING_FEATURES = "/test/tst10k-images-ubyte";
    /**
     * Learning parameters
     */
    private static final int ITERATIONS = 20;
    private static final int BATCH_SIZE = 80;
    private static final double ETA = 0.01;
    private static final double BIAS_ETA = 0.02;
    private static final int CHANNELS = 1;
    private static final int CLASSES = 10;
    private static final double MOMENTUM = 0.9;
    private static final double L2_COEFFICIENT = 0.0005;
    private static final double TRAINING_PERCENT = 0.85;
    private static final boolean UNSUPERVISED_PRE_TRAIN = true;
    private static Logger log = LoggerFactory.getLogger(Dl4jApp.class);

    public static void main(final String[] args) throws Exception {
        final List<TrainingExample> trainData = new DigitImageLoadingService(TRAINING_LABELS, TRAIN_FEATURES, 1).loadDigitImages();
        final List<TrainingExample> testData = new DigitImageLoadingService(TESTING_LABELS, TESTING_FEATURES, 1).loadDigitImages();

        Collections.shuffle(trainData);

        final List<DataSet> training = trainData
                .subList(0, (int) (TRAINING_PERCENT * trainData.size()))
                .stream()
                .map(TrainingExample::getDataSet)
                .collect(toList());

        final List<DataSet> validation = trainData
                .subList((int) (TRAINING_PERCENT * trainData.size()), trainData.size())
                .stream()
                .map(TrainingExample::getDataSet)
                .collect(toList());

        final List<DataSet> testing = testData
                .stream()
                .map(TrainingExample::getDataSet)
                .collect(toList());


        log.info(String.format("IMAGE DATA LOADED: %d training, %d validation, %d test", training.size(), validation.size(), testData.size()));
        log.info("Build nn....");


        final DataSetIterator trainIterator = new ListDataSetIterator(training, BATCH_SIZE);
        final DataSetIterator validationIterator = new ListDataSetIterator(validation);
        final DataSetIterator testIterator = new ListDataSetIterator(testing);

        /**
         * LeNet-5 like architecture
         */
        final MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration
                .Builder()
                .iterations(1)
                .regularization(true)
                .l2(L2_COEFFICIENT)
                .learningRate(ETA)
                .biasLearningRate(BIAS_ETA)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse)
                .lrPolicyDecayRate(0.001)
                .lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .momentum(MOMENTUM)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(20)//feature maps
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(CLASSES)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(UNSUPERVISED_PRE_TRAIN);

        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
        new ConvolutionLayerSetup(builder, 28, 28, 1);

        final MultiLayerConfiguration conf = builder.build();
        final MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setLayerWiseConfigurations(conf);

        if (UNSUPERVISED_PRE_TRAIN) {
            log.info("Pre sgd....");
            model.pretrain(trainIterator);
            conf.setPretrain(false);
        }

        log.info("Train model....");

        double best = Double.MIN_VALUE;
        for (int iteration = 1; iteration <= ITERATIONS; ++iteration) {
            model.fit(trainIterator);
            log.info("*** Completed iteration {} ***", iteration);
            log.info("Evaluate model....");
            final Evaluation eval = evaluateANN(model, validationIterator);
            if (eval.accuracy() > best) {
                best = eval.accuracy();
                log.info("New best model:");
                log.info(eval.stats());
                log.info("\n" + eval.confusionToString());
                saveANN(model);
                log.info("Model saved");
            }

        }


        log.info("****************Training finished********************");

        log.info("****************Testing best model********************");

        final MultiLayerNetwork bestModel = loadANN();

        final Evaluation eval = evaluateANN(bestModel, testIterator);

        log.info(eval.stats());
        log.info("\n" + eval.confusionToString());
    }

    private static void saveANN(final MultiLayerNetwork model) throws IOException {
        try (FileOutputStream fileOutputStream = new FileOutputStream(new File("nn.txt"))) {
            ModelSerializer.writeModel(model, fileOutputStream, true);
        }
    }

    private static MultiLayerNetwork loadANN() throws IOException {
        try (FileInputStream fis = new FileInputStream(new File("nn.txt"))) {
            return ModelSerializer.restoreMultiLayerNetwork(fis);
        }
    }

    private static Evaluation evaluateANN(MultiLayerNetwork model, DataSetIterator dataSetIterator) {
        final Evaluation eval = new Evaluation(CLASSES);
        while (dataSetIterator.hasNext()) {
            final DataSet ds = dataSetIterator.next();
            final INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        dataSetIterator.reset();
        return eval;
    }
}
