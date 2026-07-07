package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MatrixTriangularTests {
    double[][] aEntries;
    Matrix A;

    @Test
    void triangularTestCase() {
        // ----------------- sub-case 1 -----------------
        aEntries = new double[][]{{1, 3, 4}, {4, 5, 6}};
        A = new Matrix(aEntries);
        assertFalse(A.isTri());
        assertFalse(A.isTriL());
        assertFalse(A.isTriU());
        assertFalse(A.isDiag());

        // ----------------- sub-case 2 -----------------
        aEntries = new double[][]{
                {1, 3, 4},
                {4, 5, 6},
                {9, 13.334, -8.123}};
        A = new Matrix(aEntries);
        assertFalse(A.isTri());
        assertFalse(A.isTriL());
        assertFalse(A.isTriU());
        assertFalse(A.isDiag());

        // ----------------- sub-case 3 -----------------
        aEntries = new double[][]{
                {1, 0, 0},
                {4, 5, 0},
                {9, 13.334, -8.123}};
        A = new Matrix(aEntries);
        assertTrue(A.isTri());
        assertTrue(A.isTriL());
        assertFalse(A.isTriU());
        assertFalse(A.isDiag());

        // ----------------- sub-case 4 -----------------
        aEntries = new double[][]{
                {1, 3, 4},
                {0, 5, 6},
                {0, 0, -8.123}};
        A = new Matrix(aEntries);
        assertTrue(A.isTri());
        assertFalse(A.isTriL());
        assertTrue(A.isTriU());
        assertFalse(A.isDiag());

        // ----------------- sub-case 5 -----------------
        aEntries = new double[][]{
                {1, 0, 0},
                {0, 5, 0},
                {0, 0, -8.123}};
        A = new Matrix(aEntries);
        assertTrue(A.isTri());
        assertTrue(A.isTriL());
        assertTrue(A.isTriU());
        assertTrue(A.isDiag());
    }
}
