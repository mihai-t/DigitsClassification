package ro.ubbcluj.cs.ann.impl;

import org.junit.Test;
import ro.ubbcluj.cs.ann.validation.Validator;
import ro.ubbcluj.cs.io.TrainingExample;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author Mihai Teletin
 */
public class NeuralNetworkTest {


    @Test
    public void testBackPropagationOnIdentity() {
        final NeuralNetwork neuralNetwork = new NeuralNetwork(3, 5, 4);

        final TrainingExample t1 = new TrainingExample(new double[]{1.0, 0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 0});
        final TrainingExample t2 = new TrainingExample(new double[]{0.0, 1.0, 0.0}, new double[]{0.0, 1.0, 0.0, 0});
        final TrainingExample t3 = new TrainingExample(new double[]{0.0, 0.0, 1.0}, new double[]{0.0, 0.0, 1.0, 0});
        final List<TrainingExample> list = Arrays.asList(t1, t2, t3);

        for (int i = 0; i < 1000; ++i) {
            neuralNetwork.sgd(list);
        }
        assertEquals(Validator.validateClassification(neuralNetwork, list), list.size());

    }

    @Test
    public void testBackPropagationXor() {
        final NeuralNetwork neuralNetwork = new NeuralNetwork(2, 6, 1);

        final TrainingExample t1 = new TrainingExample(new double[]{0, 0}, new double[]{0});
        final TrainingExample t2 = new TrainingExample(new double[]{1, 0}, new double[]{1});
        final TrainingExample t3 = new TrainingExample(new double[]{0, 1}, new double[]{1});
        final TrainingExample t4 = new TrainingExample(new double[]{1, 1}, new double[]{0});

        final List<TrainingExample> list = Arrays.asList(t1, t2, t3, t4);
        for (int i = 0; i < 2000; ++i) {
            neuralNetwork.sgd(list);
        }
        assertEquals(Validator.validateMax(neuralNetwork, list), list.size());
    }


    @Test
    public void testFeedForward() throws Exception {
        final NeuralNetwork neuralNetwork = new NeuralNetwork(2, 1, 2);
        double[] testIn = new double[]{21, 1.5};
        final List<double[]> activations = new ArrayList<>();
        final List<double[]> zs = new ArrayList<>();

        Field b = neuralNetwork.getClass().getDeclaredField("biases");
        b.setAccessible(true);
        double[][] biases = (double[][]) b.get(neuralNetwork);

        Field w = neuralNetwork.getClass().getDeclaredField("weights");
        w.setAccessible(true);
        double[][][] weights = (double[][][]) w.get(neuralNetwork);

        Field f = neuralNetwork.getClass().getDeclaredField("activationFunction");
        f.setAccessible(true);
        ActivationFunction activationFunction = (ActivationFunction) f.get(neuralNetwork);

        Method feedForward = neuralNetwork.getClass().getDeclaredMethod("feedForward", double[].class, List.class, List.class);

        final double x = biases[0][0] + weights[0][0][0] * testIn[0] + weights[0][0][1] * testIn[1];
        final double rez1 = activationFunction.function(x);
        final double x1 = biases[1][0] + weights[1][0][0] * rez1;
        final double x2 = biases[1][1] + weights[1][1][0] * rez1;


        feedForward.setAccessible(true);

        final double[] result = (double[]) feedForward.invoke(neuralNetwork, testIn, activations, zs);


        assertEquals(activationFunction.function(x1), result[0], 0.001);
        assertEquals(activationFunction.function(x2), result[1], 0.001);
        assertEquals(testIn[0], activations.get(0)[0], 0.001);
        assertEquals(testIn[1], activations.get(0)[1], 0.001);
        assertEquals(rez1, activations.get(1)[0], 0.001);
        assertEquals(activationFunction.function(x1), activations.get(2)[0], 0.001);
        assertEquals(activationFunction.function(x2), activations.get(2)[1], 0.001);
        assertEquals(result[0], activations.get(2)[0], 0.001);
        assertEquals(result[1], activations.get(2)[1], 0.001);
        assertEquals(x, zs.get(0)[0], 0.001);
        assertEquals(x1, zs.get(1)[0], 0.001);
        assertEquals(x2, zs.get(1)[1], 0.001);

    }


}
