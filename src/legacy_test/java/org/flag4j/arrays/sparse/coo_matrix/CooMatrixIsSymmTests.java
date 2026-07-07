package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CooMatrixIsSymmTests {

    @Test
    void isSymmetricTests() {
        Shape aShape;
        int[] aRowIndices, aColIndices;
        double[] aData;
        CooMatrix a;

        // -------------------- sub-case 1 --------------------
        aShape = new Shape(51, 51);
        aData = new double[]{0.711, 0.875, 0.057, 0.207,
                0.885, 0.939, 0.869, 0.562,
                0.94, 0.193, 0.727, 0.938};
        aRowIndices = new int[]{0, 1, 3, 4, 17, 22, 23, 29, 38, 41, 50, 50};
        aColIndices = new int[]{30, 25, 49, 11, 22, 37, 50, 15, 25, 4, 21, 32};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isSymmetric());
        assertFalse(a.isHermitian());

        // -------------------- sub-case 2 --------------------
        aShape = new Shape(51, 51);
        aData = new double[]{0.315, 0.155, 
                0.345, 0.155, 
                0.347, 0.315, 
                0.345, 0.347, 
                0.119};
        aRowIndices = new int[]{0, 6, 10, 14, 28, 29, 40, 42, 49};
        aColIndices = new int[]{29, 14, 40, 6, 42, 0, 10, 28, 49};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 3 --------------------
        aShape = new Shape(51, 51);
        aData = new double[]{0.38, 0.38, 
                0.456, 0.843, 
                0.839, 0.533, 
                0.718, 0.533, 
                0.456, 0.718, 
                0.839, 0.843};
        aRowIndices = new int[]{1, 8, 9, 17, 26, 29, 32, 34, 36, 38, 40, 44};
        aColIndices = new int[]{8, 1, 36, 44, 40, 34, 38, 29, 9, 32, 26, 17};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 4 --------------------
        aShape = new Shape(12, 12);
        aData = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertTrue(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 5 --------------------
        aShape = new Shape(12, 17);
        aData = new double[]{0.305, 0.599, 
                0.02, 0.549, 
                0.842, 0.48, 
                0.166};
        aRowIndices = new int[]{0, 0, 0, 0, 3, 3, 8};
        aColIndices = new int[]{2, 6, 11, 12, 5, 8, 0};
        a = new CooMatrix(aShape, aData, aRowIndices, aColIndices);

        assertFalse(a.isSymmetric());
        assertFalse(a.isHermitian());
    }
}
