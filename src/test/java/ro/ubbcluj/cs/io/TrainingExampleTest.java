package ro.ubbcluj.cs.io;

import org.encog.ml.data.MLDataPair;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author Mihai Teletin
 */
public class TrainingExampleTest {

    private static final double DELTA = Math.pow(10, -7);

    @Test
    public void testTrainingExample() {
        final double[] features = {1.1, 2.2223, 3.112};
        final double[] targetValue = {0.123421221421, 0.922, 0.923};
        final TrainingExample trainingExample = new TrainingExample(features, targetValue);
        assertArrayEquals(trainingExample.getFeatures(), features, 0);
        assertArrayEquals(trainingExample.getTargetValue(), targetValue, 0);
        features[0]--;
        targetValue[1]++;
        assertFalse(Arrays.equals(trainingExample.getFeatures(), features));
        assertFalse(Arrays.equals(trainingExample.getTargetValue(), targetValue));
    }

    @Test
    public void testTrainingExampleModified() {
        final double[] features = {1.11, 2.22123, 3.12312, 3.312321, 3123.32132, 321312132.121};
        final double[] targetValue = {0.1234211221421, 0.922, 0.923, 0.23131322, 0.000000001, 0.223232, 0.3223};
        final TrainingExample trainingExample = new TrainingExample(features, targetValue);
        assertArrayEquals(trainingExample.getFeatures(), features, 0);
        assertArrayEquals(trainingExample.getTargetValue(), targetValue, 0);
        trainingExample.getFeatures()[0]++;
        trainingExample.getTargetValue()[1]--;
        assertFalse(Arrays.equals(trainingExample.getFeatures(), features));
        assertFalse(Arrays.equals(trainingExample.getTargetValue(), targetValue));
    }

    @Test
    public void testTrainingExampleEquals() {
        final double[] features = {1.112, 2.221123, 3.112};
        final double[] targetValue = {0.12342231221421, 0.922, 0.923};
        final TrainingExample trainingExample1 = new TrainingExample(features, targetValue);
        final TrainingExample trainingExample2 = new TrainingExample(features, targetValue);
        assertEquals(trainingExample1, trainingExample2);
        assertTrue(trainingExample1.hashCode() == trainingExample2.hashCode());
        trainingExample1.getFeatures()[0]++;
        assertNotEquals(trainingExample1, trainingExample2);
    }

    @Test
    public void testGetMLDataPair() {
        final TrainingExample trainingExample = new TrainingExample(new double[]{1.1, 2.2223, 3.112}, new double[]{0.123421221421, 0.92299998, 0.923});
        final MLDataPair mlDataPair = trainingExample.getMlDataPair();
        assertArrayEquals(mlDataPair.getInput().getData(), new double[]{1.1, 2.2223, 3.112}, 0);
        assertEquals(mlDataPair.getInput().size(), 3);
        assertEquals(mlDataPair.getIdeal().size(), 1);
        assertEquals(mlDataPair.getIdeal().getData(0), 2, 0);
    }

    @Test
    public void testGetDataSet() {
        final TrainingExample trainingExample = new TrainingExample(new double[]{1.1, 2.2223, 3.112}, new double[]{0.1123, 0.211, 0.91});
        final DataSet dataSet = trainingExample.getDataSet();
        assertEquals(dataSet.getFeatures().getDouble(0), 1.1, DELTA);
        assertEquals(dataSet.getFeatures().getDouble(1), 2.2223, DELTA);
        assertEquals(dataSet.getFeatures().getDouble(2), 3.112, DELTA);
        assertEquals(dataSet.getLabels().getDouble(0), 0.1123, DELTA);
        assertEquals(dataSet.getLabels().getDouble(1), 0.211, DELTA);
        assertEquals(dataSet.getLabels().getDouble(2), 0.91, DELTA);
    }


    @Test
    public void testGetClazz() {
        final TrainingExample trainingExample = new TrainingExample(new double[]{1.1, 2.2223, 3.112}, new double[]{0.1123, 0.211, 0.91});
        final int clazz = trainingExample.getClazz();
        assertEquals(clazz, 2, 0);
    }
}
