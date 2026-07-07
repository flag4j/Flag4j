package org.flag4j.linalg.ops.common.real;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.MatrixNorms;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixNormTests {

    double[][] aEntries;
    Matrix A;
    double expNorm;

    @Test
    void maxNormTestCase() {
        // ---------------- sub-case 1  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123, -9.1}, {-932.45, 551.35, -0.92342, 124.5}};
        A = new Matrix(aEntries);
        expNorm = 932.45;

        assertEquals(expNorm, MatrixNorms.maxNorm(A));
    }


    @Test
    void infNormTestCase() {
        // ---------------- sub-case 1  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123, -9.1}, {-932.45, 551.35, -0.92342, 124.5}};
        A = new Matrix(aEntries);
        expNorm = 1609.2234200000003;

        assertEquals(expNorm, MatrixNorms.infNorm(A));
    }


    @Test
    void lpNormTestCase() {
        // ---------------- sub-case 1  ----------------
        aEntries = new double[][]{
                {1.1234, 99.234, 0.000123, -9.1},
                {-932.45, 551.35, -0.92342, 124.5}};
        A = new Matrix(aEntries);
        expNorm = 1094.9348777384303;

        assertEquals(expNorm, MatrixNorms.norm(A), 1.0e-12);

        // ---------------- sub-case 2  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);
        expNorm = 1094.7776004801563;

        assertEquals(expNorm, MatrixNorms.norm(A));

        // ---------------- sub-case 3  ----------------
        aEntries = new double[][]{
                {1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);
        expNorm = 1089.5874942580217;

        assertEquals(expNorm, MatrixNorms.inducedNorm(A,2), 1.0e-12);

        // ---------------- sub-case 4  ----------------
        aEntries = new double[][]{
                {1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);
        expNorm = 1057.0184;

        assertEquals(expNorm, MatrixNorms.inducedNorm(A,1));

        // ---------------- sub-case 5  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);
        expNorm = 1094.7776004801563;

        assertEquals(expNorm, MatrixNorms.norm(A,2, 2));

        // ---------------- sub-case 6  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);
        expNorm = 1002.6438019443739;

        assertEquals(expNorm, MatrixNorms.norm(A,2, 3));

        // ---------------- sub-case 7  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);
        expNorm = 1501.7189796784505;

        assertEquals(expNorm, MatrixNorms.norm(A,2, 1));

        // ---------------- sub-case 8  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()-> MatrixNorms.inducedNorm(A,0));

        // ---------------- sub-case 10  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()-> MatrixNorms.norm(A, 0, 1));

        // ---------------- sub-case 11  ----------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123}, {-932.45, 551.35, -0.92342}, {123.445, 0.00013, 0}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()-> MatrixNorms.norm(A,1, 0));
    }
}
