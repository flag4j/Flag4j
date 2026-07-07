package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CsrCMatrixIsSymmTests {

    @Test
    void isSymmetricTests() {
        Shape aShape;
        int[] aRowPointers, aColIndices;
        Complex128[] aData;
        CsrCMatrix a;

        // -------------------- sub-case 1 --------------------
        aShape = new Shape(51, 51);
        aData = new Complex128[]{new Complex128(0.711, 0.3), new Complex128(0.875, 0.657), new Complex128(0.057, 0.164), new Complex128(0.207, 0.887), new Complex128(0.885, 0.926), new Complex128(0.939, 0.405), new Complex128(0.869, 0.506), new Complex128(0.562, 0.55), new Complex128(0.94, 0.756), new Complex128(0.193, 0.037), new Complex128(0.727, 0.541), new Complex128(0.938, 0.119)};
        aRowPointers = new int[]{0, 1, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12};
        aColIndices = new int[]{30, 25, 49, 11, 22, 37, 50, 15, 25, 4, 21, 32};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        assertFalse(a.isSymmetric());
        assertFalse(a.isHermitian());

        // -------------------- sub-case 2 --------------------
        aShape = new Shape(51, 51);
        aData = new Complex128[]{new Complex128(0.315, 0.311), new Complex128(0.155, 0.236), new Complex128(0.345, 0.92), new Complex128(0.155, 0.236), new Complex128(0.347, 0.256), new Complex128(0.315, 0.311), new Complex128(0.345, 0.92), new Complex128(0.347, 0.256), new Complex128(0.119, 0.913)};
        aRowPointers = new int[]{0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9};
        aColIndices = new int[]{29, 14, 40, 6, 42, 0, 10, 28, 49};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        assertTrue(a.isSymmetric());
        assertFalse(a.isHermitian());

        // -------------------- sub-case 3 --------------------
        aShape = new Shape(51, 51);
        aData = new Complex128[]{new Complex128(0.38, -0.82), new Complex128(0.38, 0.82), new Complex128(0.456, -0.305), new Complex128(0.843, -0.768), new Complex128(0.839, -0.306), new Complex128(0.533, -0.878), new Complex128(0.718, -0.497), new Complex128(0.533, 0.878), new Complex128(0.456, 0.305), new Complex128(0.718, 0.497), new Complex128(0.839, 0.306), new Complex128(0.843, 0.768)};
        aRowPointers = new int[]{0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12};
        aColIndices = new int[]{8, 1, 36, 44, 40, 34, 38, 29, 9, 32, 26, 17};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        assertFalse(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 4 --------------------
        aShape = new Shape(12, 12);
        aData = new Complex128[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        assertTrue(a.isSymmetric());
        assertTrue(a.isHermitian());

        // -------------------- sub-case 5 --------------------
        aShape = new Shape(12, 17);
        aData = new Complex128[]{new Complex128(0.305, 0.406), new Complex128(0.599, 0.739), new Complex128(0.02, 0.945), new Complex128(0.549, 0.64), new Complex128(0.842, 0.842), new Complex128(0.48, 0.124), new Complex128(0.166, 0.889)};
        aRowPointers = new int[]{0, 4, 4, 4, 6, 6, 6, 6, 6, 7, 7, 7, 7};
        aColIndices = new int[]{2, 6, 11, 12, 5, 8, 0};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        assertFalse(a.isSymmetric());
        assertFalse(a.isHermitian());
    }
}
