package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixUnaryOpsTests {

    Complex128[][] aEntries, expEntries;
    double[][] expReEntries;

    CMatrix A, exp;
    Matrix expRe;
    Complex128 expComplex;


    @Test
    void sumTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expComplex = new Complex128(234.66, -9923.1).add(new Complex128(32.4)).add(new Complex128(394728.1))
                .add(new Complex128(-9841, -85.13)).add(new Complex128(0, 84.1)).add(new Complex128(-5.234, 234));

        assertEquals(expComplex, A.sum());
    }


    @Test
    void sqrtTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).sqrt(), new Complex128(32.4).sqrt(), new Complex128(394728.1).sqrt()},
                {new Complex128(-9841, -85.13).sqrt(), new Complex128(0, 84.1).sqrt(), new Complex128(-5.234, 234).sqrt()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sqrt());
    }


    @Test
    void absTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expReEntries = new double[][]{
                {new Complex128(234.66, -9923.1).mag(), new Complex128(32.4).mag(), new Complex128(394728.1).mag()},
                {new Complex128(-9841, -85.13).mag(), new Complex128(0, 84.1).mag(), new Complex128(-5.234, 234).mag()}};
        expRe = new Matrix(expReEntries);

        assertEquals(expRe, A.abs());
    }


    @Test
    void transposeTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(-9841, -85.13)},
                {new Complex128(32.4), new Complex128(0, 84.1)},
                {new Complex128(394728.1), new Complex128(-5.234, 234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.T());


        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)},
                {new Complex128(234.56, 9.4), new Complex128(0.43467, 5.2), new Complex128(234.7, -0.412)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(-9841, -85.13), new Complex128(234.56, 9.4)},
                {new Complex128(32.4), new Complex128(0, 84.1), new Complex128(0.43467, 5.2)},
                {new Complex128(394728.1), new Complex128(-5.234, 234), new Complex128(234.7, -0.412)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.T());

        // -------------------- sub-case 3 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1)},
                {new Complex128(234.56, 9.4), new Complex128(0.43467, 5.2)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(-9841, -85.13), new Complex128(234.56, 9.4)},
                {new Complex128(32.4), new Complex128(0, 84.1), new Complex128(0.43467, 5.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.T());
    }


    @Test
    void hermitianTransposeTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, 9923.1), new Complex128(-9841, 85.13)},
                {new Complex128(32.4), new Complex128(0, -84.1)},
                {new Complex128(394728.1), new Complex128(-5.234, -234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.H());

        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)},
                {new Complex128(234.56, 9.4), new Complex128(0.43467, 5.2), new Complex128(234.7, -0.412)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, 9923.1), new Complex128(-9841, 85.13), new Complex128(234.56, -9.4)},
                {new Complex128(32.4), new Complex128(0, -84.1), new Complex128(0.43467, -5.2)},
                {new Complex128(394728.1), new Complex128(-5.234, -234), new Complex128(234.7, 0.412)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.H());

        // -------------------- sub-case 3 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1)},
                {new Complex128(234.56, 9.4), new Complex128(0.43467, 5.2)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, 9923.1), new Complex128(-9841, 85.13), new Complex128(234.56, -9.4)},
                {new Complex128(32.4), new Complex128(0, -84.1), new Complex128(0.43467, -5.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.H());
    }


    @Test
    void recipTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1).invert(), new Complex128(32.4).invert(), new Complex128(394728.1).invert()},
                {new Complex128(-9841, -85.13).invert(), new Complex128(0, 84.1).invert(), new Complex128(-5.234, 234).invert()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.recip());
    }


    @Test
    void conjTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, -85.13), new Complex128(0, 84.1), new Complex128(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, 9923.1), new Complex128(32.4), new Complex128(394728.1)},
                {new Complex128(-9841, 85.13), new Complex128(0, -84.1), new Complex128(-5.234, -234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.conj());

        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.66, 9923.1), new Complex128(-9841, 85.13)},
                {new Complex128(32.4), new Complex128(0, -84.1)},
                {new Complex128(394728.1), new Complex128(-5.234, -234)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.66, -9923.1), new Complex128(-9841, -85.13)},
                {new Complex128(32.4), new Complex128(0, 84.1)},
                {new Complex128(394728.1), new Complex128(-5.234, 234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.conj());
    }
}
