package ro.ubbcluj.cs.svm;

import libsvm.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ro.ubbcluj.cs.io.DigitImageLoadingService;
import ro.ubbcluj.cs.io.TrainingExample;

import java.util.List;

public class LibSVMApp {


    private static Logger log = LoggerFactory.getLogger(LibSVMApp.class);

    private static final int CLASSES = 10;
    private static final int INPUTS = 28 * 28;


    /**
     * File names
     */
    private static final String TRAINING_LABELS = "/train/trn60k-labels-ubyte";
    private static final String TRAIN_FEATURES = "/train/trn60k-images-ubyte";
    private static final String TESTING_LABELS = "/test/tst10k-labels-ubyte";
    private static final String TESTING_FEATURES = "/test/tst10k-images-ubyte";

    private static final svm_parameter svm_parameter;

    static {
        svm_parameter = new svm_parameter();

        svm_parameter.gamma = 0.03;
        svm_parameter.C = 64;

        svm_parameter.nu = 0.5;
        svm_parameter.svm_type = svm_parameter.C_SVC;
        svm_parameter.kernel_type = svm_parameter.RBF;
        svm_parameter.cache_size = 20000;
        svm_parameter.eps = 0.001;
    }


    public static void main(String[] args) throws Exception {

        final List<TrainingExample> trainData = new DigitImageLoadingService(TRAINING_LABELS, TRAIN_FEATURES, 1).loadDigitImages();
        final List<TrainingExample> testData = new DigitImageLoadingService(TESTING_LABELS, TESTING_FEATURES, 1).loadDigitImages();

        log.info(String.format("IMAGE DATA LOADED: %d training, %d test", trainData.size(), testData.size()));


        final svm_problem problem = new svm_problem();
        final int records = trainData.size();

        problem.y = new double[records];
        problem.l = records;
        problem.x = new svm_node[records][INPUTS];

        for (int i = 0; i < records; i++) {
            problem.x[i] = trainData.get(i).getSvmNodes();
            problem.y[i] = trainData.get(i).getClazz();
        }


        log.info("TRAINING SVM");
        svm_model model = svm.svm_train(problem, svm_parameter);

        log.info("TESTING SVM");
        int correct = 0;
        for (final TrainingExample test : testData) {

            final svm_node[] nodes = test.getSvmNodes();
            final int[] labels = new int[CLASSES];


            svm.svm_get_labels(model, labels);

            final double predicted = svm.svm_predict(model, nodes);

            if ((int) predicted == test.getClazz()) {
                correct++;
            }
        }

        log.info(String.format("Accuracy: %d/%d", correct, testData.size()));


    }


}