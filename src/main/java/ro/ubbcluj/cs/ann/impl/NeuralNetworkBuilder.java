package ro.ubbcluj.cs.ann.impl;

public class NeuralNetworkBuilder {

    private int[] sizes;
    private ActivationFunction activationFunction = new ActivationFunction.Sigmoid();
    private ActivationFunction outputFunction = new ActivationFunction.Sigmoid();
    private WeightsInitialization weightsInitialization = WeightsInitialization.NORMAL;
    private CostFunction costFunction = CostFunction.MEAN_SQUARED;
    private double momentum = 0;
    private double l2 = 0;
    private double eta = 0.3;


    public NeuralNetworkBuilder() {
    }

    public NeuralNetworkBuilder havingSizes(int... sizes) {
        this.sizes = sizes;
        return this;
    }

    public NeuralNetworkBuilder havingLearningRate(final double eta) {
        this.eta = eta;
        return this;
    }

    public NeuralNetworkBuilder withCostFunction(CostFunction costFunction) {
        this.costFunction = costFunction;
        return this;
    }

    public NeuralNetworkBuilder withMomentum(final double momentum) {
        this.momentum = momentum;
        return this;
    }

    public NeuralNetworkBuilder withWeightDecayL2(final double l2) {
        this.l2 = l2;
        return this;
    }

    public NeuralNetworkBuilder withActivationFunction(Activation activationFunction) {
        if (activationFunction == null) {
            throw new IllegalArgumentException("Can't set parameter to null");
        }
        this.activationFunction = activationFunction.activationFunction;
        return this;
    }

    public NeuralNetworkBuilder withOutputFunction(Activation activationFunction) {
        if (activationFunction == null) {
            throw new IllegalArgumentException("Can't set parameter to null");
        }
        this.outputFunction = activationFunction.activationFunction;
        return this;
    }

    public NeuralNetworkBuilder withWeightsInitialization(WeightsInitialization weightsInitialization) {
        if (weightsInitialization == null) {
            throw new IllegalArgumentException("Can't set parameter to null");
        }
        this.weightsInitialization = weightsInitialization;
        return this;
    }

    public NeuralNetwork build() {
        return new NeuralNetwork(activationFunction,outputFunction, weightsInitialization, costFunction, eta, momentum, l2, sizes);
    }


    public enum Activation {

        SIGMOID(new ActivationFunction.Sigmoid()),
        ReLU(new ActivationFunction.ReLU()),
        TanH(new ActivationFunction.TanH()),
        SOFT_PLUS(new ActivationFunction.SoftPlus()),
        LEAKY_ReLU(new ActivationFunction.LeakyReLU()),
        SOFT_MAX(new ActivationFunction.SoftMax());//todo implementation needed

        private ActivationFunction activationFunction;
        public double maxValue;
        public double minValue;

        Activation(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            this.maxValue = activationFunction.maxValue();
            this.minValue = activationFunction.minValue();
        }
    }

    public enum WeightsInitialization {
        NORMAL, XAVIER, UNIFORM
    }

    public enum CostFunction {
        MEAN_SQUARED, CROSS_ENTROPY, LOG_LIKELIHOOD
    }
}
