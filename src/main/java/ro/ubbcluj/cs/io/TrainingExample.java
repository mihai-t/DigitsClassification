package ro.ubbcluj.cs.io;

import libsvm.svm_node;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.IntStream;

public class TrainingExample {
    private double[] features, targetValue;

    public TrainingExample(final double[] features, final double[] targetValue) {
        this.features = Arrays.copyOf(features, features.length);
        this.targetValue = Arrays.copyOf(targetValue, targetValue.length);
    }

    public double[] getFeatures() {
        return features;
    }


    public double[] getTargetValue() {
        return targetValue;
    }

    /**
     * Convert training sample to DataSet
     *
     * @return sample in dl4j format
     */
    public DataSet getDataSet() {
        return new DataSet(Nd4j.create(features), Nd4j.create(targetValue));
    }

    /**
     * Convert training sample to MLDataPair
     *
     * @return sample in encog format
     */
    public MLDataPair getMlDataPair() {
        final MLDataPair pair = BasicMLDataPair.createPair(features.length, targetValue.length);
        pair.setInputArray(features);
        pair.setIdealArray(new double[]{getClazz()});
        return pair;
    }

    public svm_node[] getSvmNodes() {
        final svm_node[] nodes = new svm_node[features.length];
        for (int i = 0; i < features.length; i++) {
            final svm_node node = new svm_node();
            node.index = i;
            node.value = features[i];
            nodes[i] = node;
        }
        return nodes;
    }

    public int getClazz() {
        return IntStream.range(0, targetValue.length)
                .reduce((a, b) -> targetValue[a] < targetValue[b] ? b : a)
                .orElse(0);
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TrainingExample that = (TrainingExample) o;

        return Arrays.equals(features, that.features) && Arrays.equals(targetValue, that.targetValue);

    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(features);
        result = 31 * result + Arrays.hashCode(targetValue);
        return result;
    }

    @Override
    public String toString() {
        return "TrainingExample{" +
                "features=" + Arrays.toString(features) +
                ", targetValue=" + Arrays.toString(targetValue) +
                '}';
    }


}
