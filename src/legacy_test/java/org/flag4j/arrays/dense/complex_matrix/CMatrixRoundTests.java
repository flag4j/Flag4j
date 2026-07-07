package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CMatrixRoundTests {

    int tol;
    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void simpleRoundTestCase() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(23.5, Double.NEGATIVE_INFINITY), new Complex128(56, -93.1), new Complex128(3.455, 1.54)},
                {new Complex128(5, Double.NaN), new Complex128(-9854.333, 0.000003), new Complex128(Double.POSITIVE_INFINITY, 9.99)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(24, Double.NEGATIVE_INFINITY), new Complex128(56, -93), new Complex128(3, 2)},
                {new Complex128(5, Double.NaN), new Complex128(-9854, 00), new Complex128(Double.POSITIVE_INFINITY, 10)}};
        exp = new CMatrix(expEntries);

        CMatrix B = A.round();

        assertEquals(exp.shape, B.shape);

        for(int i = 0; i<exp.data.length; i++) {
            if(!exp.data[i].isNaN()) {
                assertEquals(exp.data[i], B.data[i]);
            } else {
                double expRe = (exp.data[i]).re;
                double expIm = (exp.data[i]).im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN((B.data[i]).re));
                } else {
                    assertEquals(expRe, (B.data[i]).re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN((B.data[i]).im));
                } else {
                    assertEquals(expIm, (B.data[i]).im);
                }
            }
        }
    }


    @Test
    void roundTestCase() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(23.5884, Double.NEGATIVE_INFINITY), new Complex128(56, -93.134), new Complex128(3.4557734, 1.54)},
                {new Complex128(5.0043, Double.NaN), new Complex128(-9854.333, 0.000003), new Complex128(Double.POSITIVE_INFINITY, 9.7999)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(23.588, Double.NEGATIVE_INFINITY), new Complex128(56, -93.134), new Complex128(3.456, 1.54)},
                {new Complex128(5.004, Double.NaN), new Complex128(-9854.333, 0), new Complex128(Double.POSITIVE_INFINITY, 9.8)}};
        exp = new CMatrix(expEntries);
        int tol = 3;
        CMatrix B = A.round(tol);

        assertEquals(exp.shape, B.shape);

        for(int i = 0; i<exp.data.length; i++) {
            if(!exp.data[i].isNaN()) {
                assertEquals(exp.data[i], B.data[i]);
            } else {
                double expRe = exp.data[i].re;
                double expIm = exp.data[i].im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN(B.data[i].re));
                } else {
                    assertEquals(expRe, B.data[i].re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN((B.data[i]).im));
                } else {
                    assertEquals(expIm, (B.data[i]).im);
                }
            }
        }
    }


    @Test
    void roundToZeroTestCase() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(23.5, Double.NEGATIVE_INFINITY), new Complex128(0, -0.5e-16), new Complex128(3.455, 1.54)},
                {new Complex128(5, Double.NaN), new Complex128(-34.5e-54, 34.5e-24), new Complex128(Double.POSITIVE_INFINITY, 9.99)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(23.5, Double.NEGATIVE_INFINITY), new Complex128(0), new Complex128(3.455, 1.54)},
                {new Complex128(5, Double.NaN), new Complex128(0), new Complex128(Double.POSITIVE_INFINITY, 9.99)}};
        exp = new CMatrix(expEntries);

        CMatrix B = A.roundToZero(1.0e-8);
        assertEquals(exp.shape, B.shape);

        for(int i = 0; i<exp.data.length; i++) {
            if(!exp.data[i].isNaN()) {
                assertEquals(exp.data[i], B.data[i]);
            } else {
                double expRe = (exp.data[i]).re;
                double expIm = (exp.data[i]).im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN((B.data[i]).re));
                } else {
                    assertEquals(expRe, (B.data[i]).re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN((B.data[i]).im));
                } else {
                    assertEquals(expIm, (B.data[i]).im);
                }
            }
        }
    }


    @Test
    void roundToZeroTolTestCase() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(23.5, Double.NEGATIVE_INFINITY), new Complex128(0, -0.5e-12), new Complex128(1, 1.54)},
                {new Complex128(5, Double.NaN), new Complex128(-34.5e-14, 34.5e-14), new Complex128(Double.POSITIVE_INFINITY, 9.99)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(23.5, Double.NEGATIVE_INFINITY), new Complex128(0), new Complex128(0)},
                {new Complex128(5, Double.NaN), new Complex128(0), new Complex128(Double.POSITIVE_INFINITY, 9.99)}};
        exp = new CMatrix(expEntries);
        double tol = 4.3;

        CMatrix B = A.roundToZero(tol);

        assertEquals(exp.shape, B.shape);

        for(int i = 0; i<exp.data.length; i++) {
            if(!exp.data[i].isNaN()) {
                assertEquals(exp.data[i], B.data[i]);
            } else {
                double expRe = (exp.data[i]).re;
                double expIm = (exp.data[i]).im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN((B.data[i]).re));
                } else {
                    assertEquals(expRe, (B.data[i]).re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN((B.data[i]).im));
                } else {
                    assertEquals(expIm, (B.data[i]).im);
                }
            }
        }
    }
}
