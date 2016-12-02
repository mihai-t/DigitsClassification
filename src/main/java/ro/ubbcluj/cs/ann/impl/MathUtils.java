package ro.ubbcluj.cs.ann.impl;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * @author Mihai Teletin
 */
public class MathUtils {

    private static volatile Random random = new Random();

    static double[][] randMatrix(final int n, final int m, final double mean, final double stdev) {
        final double[][] a = new double[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                //Gaussian distributions with mean @mean and standard deviation @stdev
                a[i][j] = random.nextGaussian() * stdev + mean;
            }
        }
        return a;
    }

    static double[][] randUniformMatrix(final int n, final int m) {
        final double[][] a = new double[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                a[i][j] = random.nextDouble();
            }
        }
        return a;
    }

    static double[][] copyMatrix(final double[][] c) {
        final double[][] a = new double[c.length][];
        for (int i = 0; i < c.length; ++i) {
            a[i] = new double[c[i].length];
            for (int j = 0; j < c[i].length; ++j) {
                //Gaussian distributions with mean 0 and standard deviation 1
                a[i][j] = c[i][j];
            }
        }
        return a;
    }

    static double[][][] copyMatrix(final double[][][] c) {
        final double[][][] a = new double[c.length][][];
        for (int i = 0; i < c.length; ++i) {
            a[i] = copyMatrix(c[i]);
        }
        return a;
    }

    static double[] add(double[] a, double[] b) {
        final double[] r = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            r[i] = a[i] + b[i];
        }
        return r;
    }

    static double[] add(double a, double[] b) {
        final double[] r = new double[b.length];
        for (int i = 0; i < b.length; ++i) {
            r[i] = a + b[i];
        }
        return r;
    }


    static double[] minus(double[] a) {
        final double[] r = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            r[i] = -a[i];
        }
        return r;
    }

    static double[][] add(double[][] a, double[][] b) {
        final double[][] r = new double[a.length][];
        for (int i = 0; i < a.length; ++i) {
            r[i] = add(a[i], b[i]);
        }
        return r;
    }

    static double sum(double[] a) {
        double s = 0;
        for (double aa : a) {
            s += aa;
        }
        return s;
    }

    static double[] log(double[] a) {
        final double[] rez = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            rez[i] = Math.log(a[i]);
        }
        return rez;
    }

    static double[][][] add(double[][][] a, double[][][] b) {
        final double[][][] r = new double[a.length][][];
        for (int i = 0; i < a.length; ++i) {
            r[i] = add(a[i], b[i]);
        }
        return r;
    }

    static double[][] makeEmptyCopy(double[][] array) {
        final double[][] b = new double[array.length][];
        for (int row = 0; row < array.length; ++row) {
            b[row] = array[row] == null ? null : new double[array[row].length];
        }
        return b;
    }

    static double[][][] makeEmptyCopy(double[][][] array) {
        final double[][][] b = new double[array.length][][];
        for (int row = 0; row < array.length; ++row) {
            b[row] = array[row] == null ? null : makeEmptyCopy(array[row]);
        }
        return b;
    }

    static double[] multiply(double[] a, double[] b) {
        final double[] res = Arrays.copyOf(a, a.length);
        for (int i = 0; i < res.length; ++i) {
            res[i] *= b[i];
        }
        return res;
    }


    static double[][] transpose(final double[][] original) {

        final double[][] d = new double[original[0].length][];

        if (original.length > 0) {
            for (int i = 0; i < original[0].length; i++) {
                d[i] = new double[original.length];
                for (int j = 0; j < original.length; j++) {
                    d[i][j] = original[j][i];
                }
            }
        }

        return d;
    }


    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; ++i) {
            s += a[i] * b[i];
        }
        return s;
    }


    public static int getResult(double[] targetValue){
        return IntStream.range(0, targetValue.length)
                .reduce((a, b) -> targetValue[a] < targetValue[b] ? b : a)
                .orElse(-1);
    }

}
