package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CooCMatrixIsSymmTests {

    @Test
    void isSymmetricTests() {
        Shape aShape;
        int[] aRowIndices, aColIndices;
        Complex128[] aData;
        CooCMatrix a;

        // -------------------- sub-case 1 --------------------
        aShape = new Shape(51, 51);
        aData = new Complex128[]{new Complex128(0.711, 0.3), new Complex128(0.875, 0.657),
                new Complex128(0.057, 0.164), new Complex128(0.207, 0.887),
                new Complex128(0.885, 0.926), new Complex128(0.939, 0.405),
                new Complex128(0.869, 0.506), new Complex128(0.562, 0.55),
                new Complex128(0.94, 0.756), new Complex128(0.193, 0.037),
                new Complex128(0.727, 0.541), new Complex128(0.938, 0.119)};
        aRowIndices = new int[]{0, 1, 3, 4, 17, 22, 23, 29, 38, 41, 50, 50};
        aColIndices = new int[]{30, 25, 49, 11, 22, 37, 50, 15, 25, 4, 21, 32};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);
        
        assertFalse(a.isSymmetric());
        assertFalse(a.isHermitian());

        // -------------------- sub-case 2 --------------------
        aShape = new Shape(51, 51);
        aData = new Complex128[]{new Complex128(0.315, 0.311), new Complex128(0.155, 0.236), new Complex128(0.345, 0.92), new Complex128(0.155, 0.236), new Complex128(0.347, 0.256), new Complex128(0.315, 0.311), new Complex128(0.345, 0.92), new Complex128(0.347, 0.256), new Complex128(0.119, 0.913)};
        aRowIndices = new int[]{0, 6, 10, 14, 28, 29, 40, 42, 49};
        aColIndices = new int[]{29, 14, 40, 6, 42, 0, 10, 28, 49};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);
        
        assertTrue(a.isSymmetric());
        assertFalse(a.isHermitian());

        // -------------------- sub-case 3 --------------------
        aShape = new Shape(51, 51);
        aData = new Complex128[]{new Complex128(0.38, -0.82), new Complex128(0.38, 0.82), new Complex128(0.456, -0.305), new Complex128(0.843, -0.768), new Complex128(0.839, -0.306), new Complex128(0.533, -0.878), new Complex128(0.718, -0.497), new Complex128(0.533, 0.878), new Complex128(0.456, 0.305), new Complex128(0.718, 0.497), new Complex128(0.839, 0.306), new Complex128(0.843, 0.768)};
        aRowIndices = new int[]{1, 8, 9, 17, 26, 29, 32, 34, 36, 38, 40, 44};
        aColIndices = new int[]{8, 1, 36, 44, 40, 34, 38, 29, 9, 32, 26, 17};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);
        
        assertFalse(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 4 --------------------
        aShape = new Shape(12, 12);
        aData = new Complex128[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);
        
        assertTrue(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 5 --------------------
        aShape = new Shape(12, 17);
        aData = new Complex128[]{new Complex128(0.305, 0.406), new Complex128(0.599, 0.739), new Complex128(0.02, 0.945), new Complex128(0.549, 0.64), new Complex128(0.842, 0.842), new Complex128(0.48, 0.124), new Complex128(0.166, 0.889)};
        aRowIndices = new int[]{0, 0, 0, 0, 3, 3, 8};
        aColIndices = new int[]{2, 6, 11, 12, 5, 8, 0};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);
        
        assertFalse(a.isSymmetric());
        assertFalse(a.isHermitian());
    }
}
