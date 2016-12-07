package ro.ubbcluj.cs.ann.validation;

import java.util.Arrays;

/**
 * Statistics class used to keep track of the testing results
 *
 * @author Mihai Teletin
 */
public class Statistics implements Comparable<Statistics> {

    /**
     * Total number of classes
     */
    private int lastClass;

    /**
     * The multi-class confusion matrix
     * <p>
     * confusionMatrix[i][j] is the number of j that are predicted as i
     * Notice that confusionMatrix[i][i] represents correct answers
     */
    private int confusionMatrix[][];

    public void addResult(final int predictedClass, final int actualClass) {
        final int newDimension = Math.max(predictedClass, actualClass) + 1;
        if (newDimension > lastClass) {
            lastClass = newDimension;
            confusionMatrix = reallocate(confusionMatrix, newDimension);
        }

        confusionMatrix[predictedClass][actualClass]++;
    }

    /**
     * Retrieves the generated multi class confusion matrix
     *
     * @return copy of the stored confusion matrix
     */
    public int[][] getConfusionMatrix() {
        return Arrays.stream(confusionMatrix)
                .map((int[] row) -> row.clone())
                .toArray((int length) -> new int[length][]);
    }

    /**
     * Computes number of correctly classified items
     *
     * @param clazz requested class
     * @return true positives
     */
    public int getTruePositives(final int clazz) {
        if (clazz > lastClass) {
            return 0;
        }
        return confusionMatrix[clazz][clazz];
    }

    public int getFalsePositives(int clazz) {
        int sum = 0;
        for (int j = 0; j < lastClass; ++j) {//sum on columns
            if (j != clazz) {
                sum += confusionMatrix[clazz][j];
            }
        }
        return sum;
    }

    public int getFalseNegatives(int clazz) {
        int sum = 0;
        for (int i = 0; i < lastClass; ++i) {//sum on rows
            if (i != clazz) {
                sum += confusionMatrix[i][clazz];
            }
        }
        return sum;
    }


    /**
     * Computes an averaged precision
     * precision = TP/(TP+FP)
     *
     * @return
     */
    public double getPrecision() {
        double precision = 0;

        int actualClasses = 0;

        for (int i = 0; i < lastClass; ++i) {
            final int total = getTruePositives(i) + getFalsePositives(i);
            if (total != 0) {
                actualClasses++;
                precision += (1.0 * getTruePositives(i)) / total;
            }
        }

        return precision / actualClasses;
    }


    /**
     * recall = TP/(FN+TP)
     *
     * @return
     */
    public double getRecall() {
        double recall = 0;

        int actualClasses = 0;

        for (int i = 0; i < lastClass; ++i) {
            final int total = getTruePositives(i) + getFalseNegatives(i);
            if (total != 0) {
                actualClasses++;
                recall += (1.0 * getTruePositives(i)) / total;
            }
        }

        return recall / actualClasses;
    }

    public double getAccuracy() {

        int good = 0;
        int total = 0;
        for (int i = 0; i < lastClass; ++i) {
            final int truePositives = getTruePositives(i);
            good += truePositives;
            total += getFalsePositives(i) + truePositives;
        }


        return (1.0 * good) / total;
    }

    public int getCorrectAnswers() {
        int sum = 0;

        for (int i = 0; i < lastClass; ++i) {
            sum += getTruePositives(i);
        }


        return sum;
    }

    public double getFMeasure() {
        final double recall = getRecall();
        final double precision = getPrecision();
        return 2 * (precision * recall) / (precision + recall);
    }

    synchronized static int[][] reallocate(int[][] data, int newDimension) {
        final int[][] matrix = new int[newDimension][newDimension];
        if (data == null) {
            return matrix;
        }


        for (int i = 0; i < data.length; ++i) {
            System.arraycopy(data[i], 0, matrix[i], 0, data[i].length);
        }

        return matrix;
    }

    @Override
    public String toString() {
        final StringBuilder stringBuilder = new StringBuilder("\n");
        stringBuilder.append("--------------------------------------------------");
        stringBuilder.append("\n");
        for (int i = 0; i < lastClass; ++i) {
            for (int j = 0; j < lastClass; ++j) {
                if (confusionMatrix[j][i] != 0) {
                    stringBuilder.append(String.format("Class %d classified by model as %d for %4d times\n", i, j, confusionMatrix[j][i]));
                }
            }
        }
        stringBuilder.append("--------------------------------------------------");
        stringBuilder.append("\n");
        stringBuilder.append("Accuracy:  ").append(getAccuracy()).append("\n");
        stringBuilder.append("Precision: ").append(getPrecision()).append("\n");
        stringBuilder.append("Recall:    ").append(getRecall()).append("\n");
        stringBuilder.append("F-measure: ").append(getFMeasure()).append("\n");
        stringBuilder.append("--------------------------------------------------");
        stringBuilder.append("\n");
        return stringBuilder.toString();
    }

    @Override
    public int compareTo(Statistics o) {
        return Double.compare(this.getFMeasure(), o.getFMeasure());
    }
}
