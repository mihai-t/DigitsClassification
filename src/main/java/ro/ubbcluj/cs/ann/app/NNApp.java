package ro.ubbcluj.cs.ann.app;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ro.ubbcluj.cs.ann.impl.NeuralNetwork;
import ro.ubbcluj.cs.ann.impl.NeuralNetworkBuilder;
import ro.ubbcluj.cs.ann.validation.Statistics;
import ro.ubbcluj.cs.ann.validation.Validator;
import ro.ubbcluj.cs.io.DigitImageLoadingService;
import ro.ubbcluj.cs.io.TrainingExample;

import java.util.Collections;
import java.util.List;

import static ro.ubbcluj.cs.ann.impl.NeuralNetworkBuilder.Activation;
import static ro.ubbcluj.cs.ann.impl.NeuralNetworkBuilder.Activation.SIGMOID;
import static ro.ubbcluj.cs.ann.impl.NeuralNetworkBuilder.CostFunction.MEAN_SQUARED;
import static ro.ubbcluj.cs.ann.impl.NeuralNetworkBuilder.WeightsInitialization.XAVIER;

/**
 * @author Mihai Teletin
 */
public class NNApp {

    private static Logger log = LoggerFactory.getLogger(NNApp.class);

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
    private static final int ITERATIONS = 10;
    private static final int BATCH_SIZE = 10;
    private static final double TRAINING_PERCENT = 5.0 / 6.0;
    private static final double ETA = 0.3;


    private static final Activation ACTIVATION_FUNCTION = SIGMOID;
    private static final Activation ACTIVATION_FUNCTION_OUTPUT = SIGMOID;

    public static void main(final String[] args) throws Exception {
        final List<TrainingExample> trainData = new DigitImageLoadingService(TRAINING_LABELS, TRAIN_FEATURES, ACTIVATION_FUNCTION.maxValue).loadDigitImages();
        Collections.shuffle(trainData);
        final List<TrainingExample> training = trainData.subList(0, (int) (TRAINING_PERCENT * trainData.size()));
        final List<TrainingExample> validation = trainData.subList((int) (TRAINING_PERCENT * trainData.size()), trainData.size());
        final List<TrainingExample> testData = new DigitImageLoadingService(TESTING_LABELS, TESTING_FEATURES, ACTIVATION_FUNCTION.maxValue).loadDigitImages();

        log.info(String.format("IMAGE DATA LOADED: %d training, %d validation, %d test", training.size(), validation.size(), testData.size()));

        final NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
                .havingSizes(784, 50, 10)
                .havingLearningRate(ETA)
                .withActivationFunction(ACTIVATION_FUNCTION)//todo integration needed
                .withOutputFunction(ACTIVATION_FUNCTION_OUTPUT)
                .withWeightsInitialization(XAVIER)
                .withCostFunction(MEAN_SQUARED)
                .withWeightDecayL2(0.02)//todo integration needed
                .withMomentum(0.9)//todo integration needed
                .build();

        int best = 0;
        NeuralNetwork bestNN = null;
        for (int i = 1; i <= ITERATIONS; ++i) {
            for (int j = 0; j < training.size(); j += BATCH_SIZE) {
                final List<TrainingExample> batch = training.subList(j, Math.min(training.size(), j + BATCH_SIZE));
                neuralNetwork.sgd(batch);
            }
            final Statistics statistics = Validator.getStatistics(neuralNetwork, validation);
            final int result = statistics.getCorrectAnswers();
            if (result > best) {
                bestNN = new NeuralNetwork(neuralNetwork);
                best = result;
            }
            log.info(String.format("Iteration %d (Accuracy) : %.4f ", i, statistics.getAccuracy()));
            log.info(String.format("Iteration %d (F-measure): %.4f", i, statistics.getFMeasure()));
        }

        final Statistics statisticsValidation = Validator.getStatistics(bestNN, validation);
        final int resultValidation = statisticsValidation.getCorrectAnswers();

        log.info(String.format("Best on validation: %d/%d", resultValidation, validation.size()));
        log.info(statisticsValidation.toString());

        final Statistics statisticsTest = Validator.getStatistics(bestNN, testData);
        final int result = statisticsTest.getCorrectAnswers();
        log.info(String.format("Best on test: %d/%d", result, testData.size()));


        log.info(statisticsTest.toString());
    }


}
