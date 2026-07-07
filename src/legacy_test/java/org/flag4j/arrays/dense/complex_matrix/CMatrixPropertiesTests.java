package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixPropertiesTests {

    Complex128[][] entriesA;
    CMatrix A;
    boolean expBoolResult;


    @Test
    void isIdentityTestCase() {
        // --------------- sub-case 1 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2, 123.45), new Complex128(3, -4.551)},
                {new Complex128(-0.442, 12.34), new Complex128(13.5), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(90, -43.18)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 2 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2, 123.45), new Complex128(3, -4.551)},
                {new Complex128(-0.442, 12.34), new Complex128(13.5), new Complex128(35.6)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 3 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1, 3.1335), new Complex128(2, 566.72)},
                {new Complex128(-0.442, 67.105), new Complex128(13.5, -78.431)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 4 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0)},
                {new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 5 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(1), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 6 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 6 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 7 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(1), new Complex128(0), new Complex128(0)},
                {new Complex128(1), new Complex128(0), new Complex128(1), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 8 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 9 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0, 0.3232), new Complex128(0)},
                {new Complex128(0), new Complex128(1), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- sub-case 10 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(1, -5), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());
    }


    @Test
    void isRealTestCase() {
        // --------------- sub-case 1 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2, 123.45), new Complex128(3, -4.551)},
                {new Complex128(-0.442, 12.34), new Complex128(13.5), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(90, -43.18)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isReal());


        // --------------- sub-case 2 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3)},
                {new Complex128(-0.442), new Complex128(13.5), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isReal());

        // --------------- sub-case 2 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3)},
                {new Complex128(-0.442), new Complex128(13.5, 1.4), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isReal());

        // --------------- sub-case 3 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3)},
                {new Complex128(-0.442), new Complex128(-13.5), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(0, 1.90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isReal());
    }


    @Test
    void isComplexTestCase() {
        // --------------- sub-case 1 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2, 123.45)},
                {new Complex128(-0.442, 12.34), new Complex128(13.5)},
                {new Complex128(0.4441), new Complex128(6)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isComplex());


        // --------------- sub-case 2 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3)},
                {new Complex128(-0.442), new Complex128(13.5), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isComplex());

        // --------------- sub-case 2 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3)},
                {new Complex128(-0.442), new Complex128(13.5, 1.4), new Complex128(35.6)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isComplex());

        // --------------- sub-case 3 ---------------
        entriesA = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3)},
                {new Complex128(-0.442), new Complex128(-13.5), new Complex128(35.6)},
                {new Complex128(0.4441), new Complex128(6), new Complex128(0, 1.90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isComplex());
    }
}
