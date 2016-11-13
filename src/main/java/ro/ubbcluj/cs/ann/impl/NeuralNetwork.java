package ro.ubbcluj.cs.ann.impl;


import ro.ubbcluj.cs.io.TrainingExample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;


/**
 * @author Mihai Teletin
 */
public class NeuralNetwork implements Cloneable {

    /**
     * Activation function
     */
    private final ActivationFunction activationFunction;

    /**
     * Activation function for the last layer
     */
    private final ActivationFunction outputFunction;

    /**
     * biases of this NeuralNetwork
     * <p>
     * biases[layer][neuron_target]
     */
    private final double[][] biases;

    /**
     * weights of this NeuralNetwork
     * <p>
     * weights[layer][neuron_target][neuron_source]
     */
    private final double[][][] weights;


    /**
     * Neuron layers sizes
     */
    private final int[] sizes;

    /**
     * ANN's learning rate
     */
    private final double learningRate;

    /**
     * Momentum coefficient
     */
    private final double momentum;//todo integrate

    /**
     * Weigh decay coefficient
     */
    private final double l2;//todo integrate

    /**
     * Cost function
     */
    private final NeuralNetworkBuilder.CostFunction costFunction;//todo integrate


    /**
     * Constructs a neural network
     *
     * @param activationFunction    activation function to be used by NeuralNetwork
     * @param weightsInitialization strategy for initialization
     * @param costFunction          used cost function
     * @param learningRate          global rate
     * @param momentum              coefficient for momentum
     * @param l2                    weight decay coefficient
     * @param sizes                 array consisting of each nn layer's size
     *                              example new NeuralNetwork(new int[]{1, 2, 5, 3}) creates a NeuralNetwork consisting of:
     *                              - an input neuron
     *                              - 2 neurons on the first hidden layer
     *                              - 5 neurons on the second hidden layer
     *                              - 3 output neurons
     */
    NeuralNetwork(final ActivationFunction activationFunction,
                  final ActivationFunction outputFunction,
                  final NeuralNetworkBuilder.WeightsInitialization weightsInitialization,
                  final NeuralNetworkBuilder.CostFunction costFunction,
                  final double learningRate,
                  final double momentum,
                  final double l2,
                  final int... sizes) {
        this.validateSizes(sizes);

        this.sizes = sizes;
        this.biases = new double[sizes.length - 1][];
        this.weights = new double[sizes.length - 1][][];

        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.outputFunction = outputFunction;
        this.momentum = momentum;
        this.l2 = l2;
        this.costFunction = costFunction;

        for (int i = 1; i < sizes.length; ++i) {
            biases[i - 1] = MathUtils.randMatrix(1, sizes[i], 0, 1)[0];
        }


        for (int i = 1; i < sizes.length; ++i) {
            switch (weightsInitialization) {
                case XAVIER:
                    final double xavierStdev = 1.0 / Math.sqrt(sizes[i - 1]);
                    this.weights[i - 1] = MathUtils.randMatrix(sizes[i], sizes[i - 1], 0, xavierStdev);
                    break;
                case NORMAL:
                    this.weights[i - 1] = MathUtils.randMatrix(sizes[i], sizes[i - 1], 0, 1);
                    break;
                case UNIFORM:
                    this.weights[i - 1] = MathUtils.randUniformMatrix(sizes[i], sizes[i - 1]);
                    break;
            }

        }
    }


    /**
     * Creates a new ANN by copying the state of the given ANN
     *
     * @param neuralNetwork given ANN to copy
     */
    public NeuralNetwork(NeuralNetwork neuralNetwork) {
        this.weights = MathUtils.copyMatrix(neuralNetwork.weights);
        this.biases = MathUtils.copyMatrix(neuralNetwork.biases);
        this.activationFunction = neuralNetwork.activationFunction;
        this.outputFunction = neuralNetwork.outputFunction;
        this.sizes = neuralNetwork.sizes;
        this.momentum = neuralNetwork.momentum;
        this.l2 = neuralNetwork.l2;
        this.learningRate = neuralNetwork.learningRate;
        this.costFunction = neuralNetwork.costFunction;
    }

    /**
     * Constructs a neural network using the logistic activationFunction
     * <p>
     * - activation = SIGMOID
     * - XAVIER initialization
     * - MEAN SQUARED ERROR
     * - ETA = 0.3
     * - MOMENTUM = 0
     * - L2 = 0
     *
     * @param sizes array consisting of each nn layer's size
     *              example new NeuralNetwork(new int[]{1, 2, 5, 3}) creates a NeuralNetwork consisting of:
     *              - an input neuron
     *              - 2 neurons on the hidden layer
     *              - 5 neurons on the second hidden layer
     *              - 3 output neurons
     */
    NeuralNetwork(final int... sizes) {
        this(new ActivationFunction.Sigmoid(),
                new ActivationFunction.Sigmoid(),
                NeuralNetworkBuilder.WeightsInitialization.XAVIER,
                NeuralNetworkBuilder.CostFunction.MEAN_SQUARED,
                0.3,
                0,
                0,
                sizes);
    }


    /**
     * Computes nn's value for the given input
     *
     * @param inputs testing sample
     * @return values outputted by the ANN
     */
    public double[] feedForward(final double[] inputs) {
        validateInput(inputs);
        return feedForward(inputs, null, null);
    }

    /**
     * Given an input set returns the associated class
     *
     * @param inputs testing sample
     * @return index of highest output value c, 0 <= c < number_outputs
     */
    public int classify(final double[] inputs) {
        final double[] outputs = this.feedForward(inputs);
        return IntStream.range(0, outputs.length)
                .reduce((a, b) -> outputs[a] < outputs[b] ? b : a)
                .orElse(-1);
    }

    /**
     * Performs Stochastic Gradient Descent Algorithm
     * on the given batch of samples
     *
     * @param batch given training samples
     */
    public void sgd(final List<TrainingExample> batch) {
        final double eta = learningRate;//possible use of learning rate decay on eta?

        double[][] deviationBiases = MathUtils.makeEmptyCopy(biases);
        double[][][] deviationWeights = MathUtils.makeEmptyCopy(weights);
        for (final TrainingExample t : batch) {
            validateInput(t.getFeatures());
            final Deviations<double[][], double[][][]> deviations = backPropagation(t);
            deviationBiases = MathUtils.add(deviations.deviationBiases, deviationBiases);
            deviationWeights = MathUtils.add(deviations.deviationWeights, deviationWeights);
        }

        for (int i = 0; i < weights.length; ++i) {
            for (int j = 0; j < weights[i].length; ++j) {
                for (int k = 0; k < weights[i][j].length; ++k) {
                    weights[i][j][k] -= eta * deviationWeights[i][j][k];
                }
            }
        }

        for (int i = 0; i < biases.length; ++i) {
            for (int j = 0; j < biases[i].length; ++j) {
                biases[i][j] -= eta * biases[i][j];
            }
        }

    }


    private Deviations<double[][], double[][][]> backPropagation(final TrainingExample t) {
        final double[][] deviationBiases = MathUtils.makeEmptyCopy(biases);
        final double[][][] deviationWeights = MathUtils.makeEmptyCopy(weights);

        final List<double[]> activations = new ArrayList<>();
        final List<double[]> zs = new ArrayList<>();
        final double[] result = this.feedForward(t.getFeatures(), activations, zs);

        final double[] errors = computeCost(result, t.getTargetValue());
        final double[] lastOutDerivative = outputFunction.derivative(zs.get(zs.size() - 1));


        double[] delta = MathUtils.multiply(errors, lastOutDerivative);//deviation on output layer: derivative * (target - out)

        deviationBiases[deviationBiases.length - 1] = delta;
        final double[] lastInput = activations.get(activations.size() - 2);

        for (int i = 0; i < delta.length; ++i) {
            for (int j = 0; j < lastInput.length; ++j) {
                deviationWeights[deviationWeights.length - 1][i][j] = delta[i] * lastInput[j];
            }
        }

        final int numberOfLayers = this.sizes.length;
        for (int i = numberOfLayers - 2; i > 0; --i) {//hidden layers
            final double[] out = zs.get(i - 1);
            final double[] derivative = activationFunction.derivative(out);

            final double[][] weightsToUpdate = weights[i];

            final double[][] transposesWeights = MathUtils.transpose(weightsToUpdate);
            final double[] newDelta = new double[derivative.length];


            for (int h = 0; h < derivative.length; ++h) {//weights from h
                final double sum = MathUtils.dot(delta, transposesWeights[h]);
                final double v = sum * derivative[h];
                newDelta[h] = v;
            }

            delta = newDelta;

            deviationBiases[i - 1] = delta;


            for (int j = 0; j < deviationWeights[i - 1].length; ++j) {
                for (int q = 0; q < deviationWeights[i - 1][j].length; ++q) {
                    final double[] act = activations.get(i - 1);
                    final double v = act[q];
                    deviationWeights[i - 1][j][q] = delta[j] * v;
                }
            }


        }


        return new Deviations<>(deviationBiases, deviationWeights);

    }


    private double[] feedForward(double[] inputs, final List<double[]> activations, final List<double[]> zs) {
        if (activations != null) {
            activations.add(inputs);
        }
        final int numberOfLayers = this.sizes.length;
        for (int i = 0; i < numberOfLayers - 1; ++i) {
            final int numberOfNeurons = biases[i].length;
            double[] next = new double[numberOfNeurons];
            for (int j = 0; j < numberOfNeurons; ++j) {
                next[j] = biases[i][j] + MathUtils.dot(weights[i][j], inputs);
            }
            if (zs != null) {
                zs.add(Arrays.copyOf(next, next.length));
            }
            if (i == numberOfLayers - 2) {//last layer
                inputs = outputFunction.function(next);
            } else {
                inputs = activationFunction.function(next);
            }
            if (activations != null) {
                activations.add(Arrays.copyOf(inputs, inputs.length));
            }
        }

        return inputs;
    }

    private double[] computeCost(final double[] output, final double[] desired) {
        return MathUtils.add(output, MathUtils.minus(desired));
    }


    private double[] crossEntropy(final double[] output, final double[] desired) {//fixme
        return MathUtils.add(MathUtils.multiply(MathUtils.minus(desired), MathUtils.log(output)), MathUtils.minus(MathUtils.multiply(MathUtils.add(1, MathUtils.minus(desired)), MathUtils.log(MathUtils.add(1, MathUtils.minus(output))))));
    }

    private void validateSizes(final int[] sizes) {
        if (sizes.length < 2) {
            throw new IllegalArgumentException("Ann needs at least an input and an output layer");
        }
        for (int size : sizes) {
            if (size < 1) {
                throw new IllegalArgumentException("All layers must have at least one neural unit");
            }
        }
    }


    private void validateInput(final double[] input) {
        if (input.length != this.sizes[0]) {
            throw new IllegalArgumentException("Invalid number of inputs, expected " + this.sizes[0]);
        }
    }


    @Override
    public String toString() {
        return "NeuralNetwork{" +
                "activationFunction=" + activationFunction +
                ", biases=" + Arrays.toString(biases) +
                ", weights=" + Arrays.toString(weights) +
                ", sizes=" + Arrays.toString(sizes) +
                '}';
    }

    private class Deviations<X, Y> {
        private final X deviationBiases;
        private final Y deviationWeights;

        private Deviations(X x, Y y) {
            this.deviationBiases = x;
            this.deviationWeights = y;
        }
    }


}
