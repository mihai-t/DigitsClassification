package ro.ubbcluj.cs.ann.validation;

import java.util.Arrays;

/**
 * @author Mihai Teletin
 */
public class Statistics implements Comparable<Statistics> {
    private int classes = -1;
    private int confusionMatrix[][];

    public void addResult(final int predictedClass, final int actualClass) {
        final int newDimension = Math.max(predictedClass, actualClass) + 1;
        if (newDimension > classes) {
            classes = newDimension;
            confusionMatrix = reallocate(confusionMatrix, newDimension);
        }

        confusionMatrix[predictedClass][actualClass]++;
    }

    /**
     * Retrieves the generated multi class confussion matrix
     *
     * @return confusion matrix
     */
    public int[][] getConfusionMatrix() {
        return Arrays.stream(confusionMatrix)
                .map((int[] row) -> row.clone())
                .toArray((int length) -> new int[length][]);
    }

    public int getTruePositives(int clazz) {
        if (clazz > classes) {
            return 0;
        }
        return confusionMatrix[clazz][clazz];
    }

    public int getFalsePositives(int clazz) {
        int sum = 0;
        for (int j = 0; j < classes; ++j) {//sum on columns
            if (j != clazz) {
                sum += confusionMatrix[clazz][j];
            }
        }
        return sum;
    }

    public int getFalseNegatives(int clazz) {
        int sum = 0;
        for (int i = 0; i < classes; ++i) {//sum on rows
            if (i != clazz) {
                sum += confusionMatrix[i][clazz];
            }
        }
        return sum;
    }

    //precision = TP/(TP+FP)
    public double getPrecision() {
        double precision = 0;

        int actualClasses = 0;

        for (int i = 0; i < classes; ++i) {
            final int total = getTruePositives(i) + getFalsePositives(i);
            if (total != 0) {
                actualClasses++;
                precision += (1.0 * getTruePositives(i)) / total;
            }
        }

        return precision / actualClasses;
    }

    //recall = TP/(FN+TP)
    public double getRecall() {
        double recall = 0;

        int actualClasses = 0;

        for (int i = 0; i < classes; ++i) {
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
        for (int i = 0; i < classes; ++i) {
            final int truePositives = getTruePositives(i);
            good += truePositives;
            total += getFalsePositives(i) + truePositives;
        }


        return (1.0 * good) / total;
    }

    public int getCorrectAnswers() {
        int sum = 0;

        for (int i = 0; i < classes; ++i) {
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
        for (int i = 0; i < classes; ++i) {
            for (int j = 0; j < classes; ++j) {
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
