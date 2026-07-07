package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixGetSliceTest {

    @Test
    void getSliceTests() {
        int rowStart, rowEnd, colStart, colEnd;
        int[] aRowPointers, aColIndices, expRowPointers, expColIndices;
        double[] aData, expData;
        Shape aShape, expShape;
        CsrMatrix a, exp;

        // -------------------- sub-case 1 --------------------
        rowStart = 0;
        rowEnd = 15;
        colStart = 0;
        colEnd = 156;

        aShape = new Shape(162, 525);
        aData = new double[]{0.00689, 0.47811, 0.22132, 0.95089, 0.17458, 0.96716, 0.38071, 0.32721, 0.70462};
        aRowPointers = new int[]{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
        aColIndices = new int[]{211, 479, 403, 342, 499, 197, 187, 256, 302};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(15, 156);
        expData = new double[]{};
        expRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 2 --------------------
        rowStart = 15;
        rowEnd = 25;
        colStart = 6;
        colEnd = 24;

        aShape = new Shape(25, 35);
        aData = new double[]{0.21399, 0.92765, 0.47011, 0.39667, 0.72502, 0.97094, 0.47102, 0.14218, 0.44779};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 5, 7, 8, 8, 8, 8, 9, 9, 9, 9};
        aColIndices = new int[]{14, 6, 18, 19, 0, 21, 27, 14, 11};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(10, 18);
        expData = new double[]{0.97094, 0.14218, 0.44779};
        expRowPointers = new int[]{0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        expColIndices = new int[]{15, 8, 5};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 3 --------------------
        rowStart = 8;
        rowEnd = 9;
        colStart = 18;
        colEnd = 21;

        aShape = new Shape(33, 21);
        aData = new double[]{0.85158, 0.21804, 0.43476, 0.49103, 0.96505, 0.23558, 0.52658};
        aRowPointers = new int[]{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 7, 7, 7, 7, 7};
        aColIndices = new int[]{18, 0, 0, 8, 7, 14, 17};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(1, 3);
        expData = new double[]{};
        expRowPointers = new int[]{0, 0};
        expColIndices = new int[]{};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);
        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));


        // -------------------- sub-case 4 --------------------
        rowStart = 5;
        rowEnd = 22;
        colStart = 0;
        colEnd = 55;

        aShape = new Shape(55, 55);
        aData = new double[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(17, 55);
        expData = new double[]{};
        expRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);
        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 4 --------------------
        aShape = new Shape(55, 55);
        aData = new double[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        CsrMatrix finalA = a;
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(-1, 2, 4, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 2, -4, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(4, 2, 4, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 2, 4, 1));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 56, 0, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 2, 0, 514));
    }
}
