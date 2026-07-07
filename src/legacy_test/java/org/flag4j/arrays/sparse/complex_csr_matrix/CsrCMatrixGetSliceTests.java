package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixGetSliceTests {

    @Test
    void getSliceTests() {
        int rowStart, rowEnd, colStart, colEnd;
        int[] aRowPointers, aColIndices, expRowPointers, expColIndices;
        Complex128[] aData, expData;
        Shape aShape, expShape;
        CsrCMatrix a, exp;

        // -------------------- sub-case 1 --------------------
        rowStart = 0;
        rowEnd = 15;
        colStart = 0;
        colEnd = 156;

        aShape = new Shape(162, 525);
        aData = new Complex128[]{new Complex128(0.78035, 0.12308), new Complex128(0.69964, 0.39359), new Complex128(0.71946, 0.23139), new Complex128(0.03003, 0.24849), new Complex128(0.75854, 0.87197), new Complex128(0.6154, 0.79933), new Complex128(0.22866, 0.81778), new Complex128(0.62108, 0.02225), new Complex128(0.12051, 0.77826)};
        aRowPointers = new int[]{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
        aColIndices = new int[]{377, 323, 104, 450, 260, 373, 314, 507, 383};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(15, 156);
        expData = new Complex128[]{};
        expRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{};
        exp = new CsrCMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 2 --------------------
        rowStart = 15;
        rowEnd = 25;
        colStart = 6;
        colEnd = 24;

        aShape = new Shape(25, 35);
        aData = new Complex128[]{new Complex128(0.58041, 0.13741), new Complex128(0.31126, 0.11722), new Complex128(0.24317, 0.64169), new Complex128(0.37413, 0.97784), new Complex128(0.00337, 0.35128), new Complex128(0.76686, 0.96587), new Complex128(0.1325, 0.53405), new Complex128(0.17819, 0.94705), new Complex128(0.20021, 0.52085)};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 7, 8, 8, 8, 9, 9, 9, 9};
        aColIndices = new int[]{24, 34, 10, 21, 27, 11, 13, 29, 15};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(10, 18);
        expData = new Complex128[]{new Complex128(0.24317, 0.64169), new Complex128(0.37413, 0.97784), new Complex128(0.76686, 0.96587), new Complex128(0.1325, 0.53405), new Complex128(0.20021, 0.52085)};
        expRowPointers = new int[]{0, 0, 2, 4, 4, 4, 4, 5, 5, 5, 5};
        expColIndices = new int[]{4, 15, 5, 7, 9};
        exp = new CsrCMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 3 --------------------
        rowStart = 8;
        rowEnd = 9;
        colStart = 18;
        colEnd = 21;

        aShape = new Shape(33, 21);
        aData = new Complex128[]{new Complex128(0.38171, 0.9855), new Complex128(0.97057, 0.96037), new Complex128(0.87573, 0.34079), new Complex128(0.37559, 0.59789), new Complex128(0.87787, 0.76645), new Complex128(0.57899, 0.49322), new Complex128(0.31817, 0.39966)};
        aRowPointers = new int[]{0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 7, 7, 7};
        aColIndices = new int[]{9, 13, 16, 8, 11, 2, 1};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(1, 3);
        expData = new Complex128[]{};
        expRowPointers = new int[]{0, 0};
        expColIndices = new int[]{};
        exp = new CsrCMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 4 --------------------
        rowStart = 5;
        rowEnd = 22;
        colStart = 0;
        colEnd = 55;

        aShape = new Shape(55, 55);
        aData = new Complex128[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        expShape = new Shape(17, 55);
        expData = new Complex128[]{};
        expRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{};
        exp = new CsrCMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.getSlice(rowStart, rowEnd, colStart, colEnd));

        // -------------------- sub-case 4 --------------------
        aShape = new Shape(55, 55);
        aData = new Complex128[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        CsrCMatrix finalA = a;
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(-1, 2, 4, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 2, -4, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(4, 2, 4, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 2, 4, 1));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 56, 0, 5));
        assertThrows(IllegalArgumentException.class, ()-> finalA.getSlice(1, 2, 0, 514));
    }
}
