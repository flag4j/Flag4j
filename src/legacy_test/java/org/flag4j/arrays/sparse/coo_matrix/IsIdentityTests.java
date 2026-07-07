package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IsIdentityTests {

    @Test
    void testCooMatrixIsCloseToI() {
        Shape aShape;
        int[] aRowIndices, aColIndices;
        double[] aData;
        CooMatrix a;

        // ---------------------- sub-case 1 ----------------------
        aShape = new Shape(50, 12);
        aRowIndices = new int[]{0, 5, 14, 23, 49};
        aColIndices = new int[]{1, 3, 3,  1,  2};
        aData = new double[]{1, 3, 5, 7, 9};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isCloseToI());

        // ---------------------- sub-case 2 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 12, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new double[]{1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isCloseToI());

        // ---------------------- sub-case 3 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new double[]{1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
                5};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isCloseToI());

        // ---------------------- sub-case 4 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new double[]{1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isCloseToI());

        // ---------------------- sub-case 5 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new double[]{1, 1, 1, 1, 1, 1,
                1, 1, 1.000000000000001, 1, 1,
                1, 1,
                1, 1, 1, 0.99999999, 1, 1,
                1};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isCloseToI());

        // ---------------------- sub-case 6 ----------------------
        aShape = new Shape(20, 20);
        aRowIndices = new int[]{0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aColIndices = new int[]{0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        aData = new double[]{1, 1, 1, 1.24e-16,
                1, 1, 1, 1, 1,
                1.00000000000000001, 1, 1,
                1.0e-18, 1, 1, 1, 1, 1,
                0.9999999999999, 1, 1, 1};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isCloseToI());
    }
}
