package ro.ubbcluj.cs.ann.impl;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static ro.ubbcluj.cs.ann.impl.MathUtils.*;

/**
 * @author Mihai Teletin
 */
public class MathTest {

    @Test
    public void testRandomMatrix() {
        final double[][] doubles = randMatrix(5, 6, 0, 1);
        assertEquals(doubles.length, 5);
        for (int i = 0; i < 5; ++i) {
            assertEquals(doubles[i].length, 6);
        }
    }

    @Test
    public void testRandomUniformMatrix() {
        final double[][] doubles = randUniformMatrix(5, 6);
        assertEquals(doubles.length, 5);
        for (int i = 0; i < 5; ++i) {
            assertEquals(doubles[i].length, 6);
        }
    }

    @Test
    public void testAdd() {
        double[] a = new double[]{1, 2, 3, 4, 1};
        double[] b = new double[]{-1, -1, 2, 3, 4};
        Assert.assertArrayEquals(add(a, b), new double[]{0, 1, 5, 7, 5}, 0);
        Assert.assertArrayEquals(add(add(a, b), b), new double[]{-1, 0, 7, 10, 9}, 0);
    }

    @Test
    public void testAdd2d() {
        double[][] a = new double[][]{{1, 2}, {3}, {4, 1}};
        double[][] b = new double[][]{{-1, -1}, {2}, {3, 4}};
        Assert.assertTrue(Arrays.deepEquals(add(a, b), new double[][]{{0, 1}, {5}, {7, 5}}));
        Assert.assertTrue(Arrays.deepEquals(add(add(a, b), b), new double[][]{{-1, 0}, {7}, {10, 9}}));
    }

    @Test
    public void testAdd3d() {
        double[][][] a = new double[][][]{{{1}, {2}}, {{3}}, {{4, 1}}};
        double[][][] b = new double[][][]{{{-1}, {-1}}, {{2}}, {{3, 4}}};
        Assert.assertTrue(Arrays.deepEquals(add(a, b), new double[][][]{{{0}, {1}}, {{5}}, {{7, 5}}}));
        Assert.assertTrue(Arrays.deepEquals(add(add(a, b), b), new double[][][]{{{-1}, {0}}, {{7}}, {{10, 9}}}));
    }

    @Test
    public void testMinus() {
        double[] a = new double[]{1, -2, 3, -4, 1};
        Assert.assertArrayEquals(minus(a), new double[]{-1, 2, -3, 4, -1}, 0);
        Assert.assertArrayEquals(minus(minus(a)), a, 0);
    }

    @Test
    public void testMakeEmptyCopy2d() {
        double[][] a = new double[][]{{1, 2}, {3}, {4, 1}};
        Assert.assertTrue(Arrays.deepEquals(makeEmptyCopy(a), new double[][]{{0, 0}, {0}, {0, 0}}));
    }

    @Test
    public void testMakeEmptyCopy3d() {
        double[][][] a = new double[][][]{{{1}, {2}}, {{3}}, {{4, 1}}};
        Assert.assertTrue(Arrays.deepEquals(makeEmptyCopy(a), new double[][][]{{{0}, {0}}, {{0}}, {{0, 0}}}));
    }

    @Test
    public void testDot() {
        double[] a = new double[]{1, 2, 3, 4, 1};
        double[] b = new double[]{-1, -1, 2, 3, 4};
        Assert.assertEquals(dot(a, b), 19, 0);
        Assert.assertEquals(dot(b, a), 19, 0);
        Assert.assertEquals(dot(b, b), 31, 0);
    }

    @Test
    public void testCopy2d() {
        double[][] a = new double[][]{{1, 2}, {3}, {4, 1}};
        double[][] b = copyMatrix(a);
        Assert.assertTrue(Arrays.deepEquals(a, b));
        a[0][0] = 22;
        Assert.assertFalse(Arrays.deepEquals(a, b));
    }

    @Test
    public void testCopy3d() {
        double[][][] a = new double[][][]{{{1}, {2}}, {{3}}, {{4, 1}}};
        double[][][] b = copyMatrix(a);
        Assert.assertTrue(Arrays.deepEquals(a, b));
        a[0][0][0] = 22;
        Assert.assertFalse(Arrays.deepEquals(a, b));
    }
}
