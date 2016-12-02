package ro.ubbcluj.cs.ann.validation;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Mihai Teletin
 */
public class StatisticsTest {

    @Test
    public void testReallocateNull() {
        int[][] a = null;
        int[][] b = Statistics.reallocate(a, 1);

        Assert.assertEquals(b.length, 1);
        Assert.assertEquals(b[0].length, 1);
        Assert.assertNotEquals(a, b);
    }

    @Test
    public void testReallocateOne() {
        int[][] a = {{1}};
        int[][] b = Statistics.reallocate(a, 3);

        Assert.assertEquals(b.length, 3);
        Assert.assertEquals(b[0].length, 3);
        Assert.assertNotEquals(a, b);
        Assert.assertArrayEquals(b, new int[][]{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    }

    @Test
    public void testReallocateTwo() {
        int[][] a = {{1, 2}, {3, 4}};
        int[][] b = Statistics.reallocate(a, 4);

        Assert.assertEquals(b.length, 4);
        Assert.assertEquals(b[0].length, 4);
        Assert.assertEquals(b[1].length, 4);
        Assert.assertEquals(b[2].length, 4);
        Assert.assertEquals(b[3].length, 4);
        Assert.assertNotEquals(a, b);
        Assert.assertArrayEquals(b, new int[][]{{1, 2, 0, 0}, {3, 4, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
    }

    @Test
    public void testConfusionMatrix() {
        final Statistics statistics = new Statistics();


        statistics.addResult(1, 1);
        Assert.assertArrayEquals(statistics.getConfusionMatrix(), new int[][]{{0, 0}, {0, 1}});
        statistics.addResult(2, 1);
        statistics.addResult(2, 2);

        Assert.assertArrayEquals(statistics.getConfusionMatrix(), new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 1, 1}});

        statistics.addResult(2, 2);

        Assert.assertArrayEquals(statistics.getConfusionMatrix(), new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 1, 2}});


    }

    @Test
    public void testFMeasure() {
        final Statistics statistics = new Statistics();

        statistics.addResult(1, 1);
        Assert.assertEquals(statistics.getFMeasure(), 1.0, 0);

        statistics.addResult(2, 2);
        Assert.assertEquals(statistics.getFMeasure(), 1.0, 0);

        statistics.addResult(3, 4);

        Assert.assertEquals(statistics.getFMeasure(), 0.6666666, 0.00001);

        statistics.addResult(3, 3);


        Assert.assertEquals(statistics.getFMeasure(), 0.789473, 0.00001);
    }

    @Test
    public void testAccuracy() {
        final Statistics statistics = new Statistics();

        statistics.addResult(1, 1);
        Assert.assertEquals(statistics.getAccuracy(), 1.0, 0);

        statistics.addResult(2, 2);

        Assert.assertEquals(statistics.getAccuracy(), 1.0, 0);

        statistics.addResult(3, 4);

        Assert.assertEquals(statistics.getAccuracy(), 2.0 / 3, 0);

        statistics.addResult(3, 3);


        Assert.assertEquals(statistics.getAccuracy(), 3.0 / 4, 0);

        statistics.addResult(5, 6);


        Assert.assertEquals(statistics.getAccuracy(), 3.0 / 5, 0);


        statistics.addResult(6, 6);
        Assert.assertEquals(statistics.getAccuracy(), 4.0 / 6, 0);

        statistics.addResult(7, 6);
        statistics.addResult(7, 6);
        statistics.addResult(8, 6);

        Assert.assertEquals(statistics.getAccuracy(), 4.0 / 9, 0);

        statistics.addResult(6, 6);

        Assert.assertEquals(statistics.getAccuracy(), 5.0 / 10, 0);
    }
}
