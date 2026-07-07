package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixVectorCheckTests {
    double[][] aEntries;
    Matrix A;

    @Test
    void isVectorTestCase() {
        // ------------------ sub-case 1 ------------------
        aEntries = new double[][]{{1.123, 5325.123}, {1.566, -2354.5767}};
        A = new Matrix(aEntries);
        assertFalse(A.isVector());

        // ------------------ sub-case 2 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}, {1.566, -2354.5767, 9}};
        A = new Matrix(aEntries);
        assertFalse(A.isVector());

        // ------------------ sub-case 3 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}};
        A = new Matrix(aEntries);
        assertTrue(A.isVector());

        // ------------------ sub-case 4 ------------------
        aEntries = new double[][]{{1.123}};
        A = new Matrix(aEntries);
        assertTrue(A.isVector());

        // ------------------ sub-case 5 ------------------
        aEntries = new double[][]{{1.123}, {-9234}, {-7234.23}, {0.3242}, {1}};
        A = new Matrix(aEntries);
        assertTrue(A.isVector());
    }

    @Test
    void vectorTypeTestCase() {
        // ------------------ sub-case 1 ------------------
        aEntries = new double[][]{{1.123, 5325.123}, {1.566, -2354.5767}};
        A = new Matrix(aEntries);
        assertEquals(-1, A.vectorType());

        // ------------------ sub-case 2 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}, {1.566, -2354.5767, 9}};
        A = new Matrix(aEntries);
        assertEquals(-1, A.vectorType());

        // ------------------ sub-case 3 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}};
        A = new Matrix(aEntries);
        assertEquals(1, A.vectorType());

        // ------------------ sub-case 4 ------------------
        aEntries = new double[][]{{1.123}};
        A = new Matrix(aEntries);
        assertEquals(0, A.vectorType());

        // ------------------ sub-case 5 ------------------
        aEntries = new double[][]{{1.123}, {-9234}, {-7234.23}, {0.3242}, {1}};
        A = new Matrix(aEntries);
        assertEquals(2, A.vectorType());
    }
}
