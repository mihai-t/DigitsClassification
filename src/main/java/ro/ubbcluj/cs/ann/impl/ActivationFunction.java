package ro.ubbcluj.cs.ann.impl;

/**
 * @author Mihai Teletin
 */
interface ActivationFunction {

    int INFINITY = 1000;

    double function(final double x);

    double derivative(final double x);

    double minValue();

    double maxValue();

    default double[] function(final double[] x) {
        final double[] result = new double[x.length];
        for (int i = 0; i < result.length; ++i) {
            result[i] = function(x[i]);
        }
        return result;
    }

    default double[] derivative(final double[] x) {
        final double[] result = new double[x.length];
        for (int i = 0; i < result.length; ++i) {
            result[i] = derivative(x[i]);
        }
        return result;
    }

    /**
     * F:R->(0,inf)
     */
    class ReLU implements ActivationFunction {
        @Override
        public double function(final double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(final double x) {
            return x <= 0 ? 0 : 1;
        }

        @Override
        public double minValue() {
            return 0;
        }

        @Override
        public double maxValue() {
            return INFINITY;
        }

    }

    /**
     * F:R->(0,inf)
     */
    class LeakyReLU implements ActivationFunction {
        @Override
        public double function(final double x) {
            return x < 0 ? 0.01 * x : x;
        }

        @Override
        public double derivative(final double x) {
            return x <= 0 ? 0.01 : 1;
        }

        @Override
        public double minValue() {
            return 0;
        }

        @Override
        public double maxValue() {
            return INFINITY;
        }

    }

    /**
     * F:R->(0,1)
     */
    class Sigmoid implements ActivationFunction {
        @Override
        public double function(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            final double sigmoid = function(x);
            return sigmoid * (1 - sigmoid);
        }

        @Override
        public double minValue() {
            return 0;
        }

        @Override
        public double maxValue() {
            return 1;
        }

    }

    /**
     * F:R->(0,inf)
     */
    class SoftPlus implements ActivationFunction {
        @Override
        public double function(final double x) {
            return Math.log(1 + Math.exp(x));
        }

        @Override
        public double derivative(final double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double minValue() {
            return 0;
        }

        @Override
        public double maxValue() {
            return INFINITY;
        }
    }

    /**
     * F:R->(-1,1)
     */
    class TanH implements ActivationFunction {
        @Override
        public double function(final double x) {
            return 2.0 / (1 + Math.exp(-2 * x)) + 1;
        }

        @Override
        public double derivative(final double x) {
            final double function = function(x);
            return 1.0 - function * function;
        }

        @Override
        public double minValue() {
            return -1;
        }

        @Override
        public double maxValue() {
            return 1;
        }
    }


    /**
     *
     */
    class SoftMax implements ActivationFunction {//todo implementation needed
        @Override
        public double function(double x) {
            return 0;
        }

        @Override
        public double derivative(double x) {
            return 0;
        }

        @Override
        public double minValue() {
            return 0;
        }

        @Override
        public double maxValue() {
            return 0;
        }
    }
}
