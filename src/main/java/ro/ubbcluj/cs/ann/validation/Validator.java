package ro.ubbcluj.cs.ann.validation;


import ro.ubbcluj.cs.ann.impl.NeuralNetwork;
import ro.ubbcluj.cs.io.TrainingExample;

import java.util.List;
import java.util.stream.IntStream;

public class Validator {


    /**
     * Compute the number of correctly classified instances
     * of an ANN over a set of training examples
     *
     * @param neuralNetwork given neuralNetwork
     * @param testData      given set of test data
     * @return - Computed number of correct items
     */
    public static int validateClassification(final NeuralNetwork neuralNetwork, final List<TrainingExample> testData) {
        int totalCorrect = 0;
        for (TrainingExample trainingExample : testData) {
            final double[] targetValue = trainingExample.getTargetValue();
            final int predicted = neuralNetwork.classify(trainingExample.getFeatures());
            final int actual = IntStream.range(0, targetValue.length)
                    .reduce((a, b) -> targetValue[a] < targetValue[b] ? b : a)
                    .orElse(-1);
            if (predicted == actual) {
                totalCorrect++;
            }
        }
        return totalCorrect;
    }


    public static int validateMax(final NeuralNetwork neuralNetwork, final List<TrainingExample> list) {
        int ok = 0;
        for (TrainingExample trainingExample : list) {
            double[] outputs = neuralNetwork.feedForward(trainingExample.getFeatures());
            if (outputs[0] < 0.5) {
                if (trainingExample.getTargetValue()[0] < 0.5) {
                    ok++;
                }
            } else {
                if (trainingExample.getTargetValue()[0] > 0.5) {
                    ok++;
                }
            }
        }
        return ok;
    }


}
