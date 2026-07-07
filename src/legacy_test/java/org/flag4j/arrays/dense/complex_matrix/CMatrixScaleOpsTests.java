package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixScaleOpsTests {

    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realScaleMultTestCase() {
        double scal;

        // ---------------------- sub-case 1 ----------------------
        scal = 79.3419;
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).mult(scal), new Complex128(32.4).mult(scal), new Complex128(394728.1).mult(scal)},
                {new Complex128(-9841, -85.13).mult(scal), new Complex128(0, 84.1).mult(scal), new Complex128(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));

        // ---------------------- sub-case 2 ----------------------
        scal = -2179.3419;
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).mult(scal), new Complex128(32.4).mult(scal), new Complex128(394728.1).mult(scal)},
                {new Complex128(-9841, -85.13).mult(scal), new Complex128(0, 84.1).mult(scal), new Complex128(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));
    }


    @Test
    void complexScaleMultTestCase() {
        Complex128 scal;

        // ---------------------- sub-case 1 ----------------------
        scal = new Complex128(9234.1, -923.1);
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).mult(scal), new Complex128(32.4).mult(scal), new Complex128(394728.1).mult(scal)},
                {new Complex128(-9841, -85.13).mult(scal), new Complex128(0, 84.1).mult(scal), new Complex128(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));

        // ---------------------- sub-case 2 ----------------------
        scal = new Complex128(-0.000234, -923.1);
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).mult(scal), new Complex128(32.4).mult(scal), new Complex128(394728.1).mult(scal)},
                {new Complex128(-9841, -85.13).mult(scal), new Complex128(0, 84.1).mult(scal), new Complex128(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));
    }


    @Test
    void realScaleDivTestCase() {
        double scal;

        // ---------------------- sub-case 1 ----------------------
        scal = 79.3419;
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).div(scal), new Complex128(32.4).div(scal), new Complex128(394728.1).div(scal)},
                {new Complex128(-9841, -85.13).div(scal), new Complex128(0, 84.1).div(scal), new Complex128(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));

        // ---------------------- sub-case 2 ----------------------
        scal = -2179.3419;
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).div(scal), new Complex128(32.4).div(scal), new Complex128(394728.1).div(scal)},
                {new Complex128(-9841, -85.13).div(scal), new Complex128(0, 84.1).div(scal), new Complex128(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));
    }


    @Test
    void complexScaleDivTestCase() {
        Complex128 scal;

        // ---------------------- sub-case 1 ----------------------
        scal = new Complex128(9234.1, -923.1);
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).div(scal), new Complex128(32.4).div(scal), new Complex128(394728.1).div(scal)},
                {new Complex128(-9841, -85.13).div(scal), new Complex128(0, 84.1).div(scal), new Complex128(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));

        // ---------------------- sub-case 2 ----------------------
        scal = new Complex128(-0.000234, -923.1);
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).div(scal), new Complex128(32.4).div(scal), new Complex128(394728.1).div(scal)},
                {new Complex128(-9841, -85.13).div(scal), new Complex128(0, 84.1).div(scal), new Complex128(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));
    }
}

