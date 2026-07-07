package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class MatrixAggregationTests {

    double[][] aEntries;
    Matrix A;
    Double expAg;
    int[] expIndices;

    @Test
    void sumTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 1.334+-2.3112+334.3+4.13+-35.33+6;
        Assertions.assertEquals(expAg, A.sum());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.sum());
    }


    @Test
    void minTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = -35.33;
        Assertions.assertEquals(expAg, A.min());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.min());
    }


    @Test
    void maxTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 334.3;
        Assertions.assertEquals(expAg, A.max());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.max());
    }


    @Test
    void maxAbsTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 334.3;
        Assertions.assertEquals(expAg, A.maxAbs());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.maxAbs());
    }


    @Test
    void minAbsTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 1.334;
        Assertions.assertEquals(expAg, A.minAbs());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.minAbs());
    }


    @Test
    void argminTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expIndices = new int[]{1, 1};
        Assertions.assertArrayEquals(expIndices, A.argmin());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expIndices = new int[]{};
        Assertions.assertArrayEquals(expIndices, A.argmin());
    }


    @Test
    void argmaxTestCase() {
        // ----------- sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expIndices = new int[]{0, 2};
        Assertions.assertArrayEquals(expIndices, A.argmax());

        // ----------- sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expIndices = new int[]{};
        Assertions.assertArrayEquals(expIndices, A.argmax());
    }
}
