package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.MatrixNorms;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixNormTests {

    @Test
    void csrLpqNorms() {
        Shape aShape;
        double[] aData;
        int[] aRowPointers, aColIndices;
        CsrMatrix a;
        double exp, p, q;

        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(32, 32);
        aData = new double[]{0.59873, 0.14037, 0.51302, 0.81953, 0.85602, 0.26156, 0.09093, 0.47848, 0.49457, 0.22042};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 6, 7, 7, 8, 8, 9, 10, 10, 10, 10};
        aColIndices = new int[]{6, 8, 9, 27, 1, 2, 4, 31, 5, 18};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 1;
        q = 1;
        exp = 4.473629999999999;

        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(32, 32);
        aData = new double[]{0.25735, 0.03955, 0.30834, 0.78676, 0.06766, 0.86635, 0.29043, 0.6284, 0.87736, 0.9143};
        aRowPointers = new int[]{0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 6, 6, 7, 7, 8, 8, 8, 10, 10, 10, 10, 10, 10};
        aColIndices = new int[]{9, 3, 9, 17, 8, 25, 9, 29, 24, 29};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 1;
        q = 2;
        exp = 2.2931029222867427;

        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(32, 32);
        aData = new double[]{0.01011, 0.30401, 0.44966, 0.91805, 0.74716, 0.79202, 0.48672, 0.9156, 0.26773, 0.19309};
        aRowPointers = new int[]{0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10};
        aColIndices = new int[]{2, 1, 6, 22, 4, 29, 31, 29, 19, 4};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 2;
        q = 1;
        exp = 4.418614617567907;
        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 4 -----------------------
        aShape = new Shape(32, 32);
        aData = new double[]{0.26041, 0.0959, 0.91373, 0.93231, 0.1104, 0.02363, 0.94682, 0.72971, 0.98633, 0.96019};
        aRowPointers = new int[]{0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 10, 10};
        aColIndices = new int[]{22, 8, 5, 8, 16, 6, 29, 27, 5, 6};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 4.12;
        q = 9.3;
        exp = 1.1861666012978695;

        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 5 -----------------------
        aShape = new Shape(32, 32);
        aData = new double[]{0.57274, 0.87413, 0.52516, 0.08473, 0.55946, 0.6395, 0.10405, 0.78955, 0.68463, 0.48435};
        aRowPointers = new int[]{0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10};
        aColIndices = new int[]{3, 4, 14, 10, 31, 5, 0, 26, 24, 8};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 0;
        q = 0;

        CsrMatrix finalA = a;
        double finalP = p;
        double finalQ = q;
        assertThrows(IllegalArgumentException.class, () -> MatrixNorms.norm(finalA, finalP, finalQ));
    }
}
