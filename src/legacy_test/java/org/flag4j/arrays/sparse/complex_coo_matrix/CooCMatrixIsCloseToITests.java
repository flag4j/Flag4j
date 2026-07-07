package org.flag4j.arrays.sparse.complex_coo_matrix;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CooCMatrixIsCloseToITests {

    @Test
    void testCooCMatrixIsCloseToI() {
        Shape aShape;
        int[] aRowIndices, aColIndices;
        Complex128[] aData;
        CooCMatrix a;

        // ---------------------- sub-case 1 ----------------------
        aShape = new Shape(50, 12);
        aRowIndices = new int[]{0, 5, 14, 23, 49};
        aColIndices = new int[]{1, 3, 3,  1,  2};
        aData = new Complex128[]{new Complex128(1, 234), new Complex128(3, 456), new Complex128(5, 789),
        new Complex128(7, 89), new Complex128(9, 99)};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isCloseToI());

        // ---------------------- sub-case 2 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 12, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new Complex128[]{Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isCloseToI());

        // ---------------------- sub-case 3 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new Complex128[]{Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                new Complex128(5, -1)};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isCloseToI());

        // ---------------------- sub-case 4 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new Complex128[]{Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isCloseToI());

        // ---------------------- sub-case 5 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new Complex128[]{Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, new Complex128(1.000000000000001, -15.1e-56), Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE,
                Complex128.ONE, Complex128.ONE, Complex128.ONE, new Complex128(0.99999999, 1.4e-14), Complex128.ONE, Complex128.ONE,
                Complex128.ONE};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isCloseToI());

        // ---------------------- sub-case 6 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new Complex128[]{Complex128.ONE, Complex128.ONE, Complex128.ONE, new Complex128(1.24e-16, 1.2e-12),
                Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                new Complex128(1.000000000000001, -15.1e-56), Complex128.ONE, Complex128.ONE,
                new Complex128(1.0e-18, 25.6e-21), Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE, Complex128.ONE,
                new Complex128(0.99999999, 1.4e-14), Complex128.ONE, Complex128.ONE, Complex128.ONE};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isCloseToI());
    }
}
