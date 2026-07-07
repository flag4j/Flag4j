package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.exceptions.NDArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrElemMultTests {

    @Test
    void elemMultTests() {
        Shape aShape, bShape, expShape;
        double[] aData, bData, expData;
        int[] aRowPointers, bRowPointers, expRowPointers;
        int[] aColIndices, bColIndices, expColIndices;
        CsrMatrix a, b, exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(12, 12);
        aData = new double[]{0.02234, 0.43189, 0.87963, 0.79995, 0.72224, 0.74126, 0.44675, 0.44917, 0.69626, 0.81694, 0.32876, 0.02551, 0.75703, 0.58045, 0.64317, 0.23147, 0.52038, 0.37257, 0.88039, 0.48479, 0.02311, 0.72269, 0.80202, 0.0231, 0.12105, 0.47067, 0.2849, 0.3631, 0.70669};
        aRowPointers = new int[]{0, 4, 4, 6, 12, 14, 15, 17, 18, 24, 25, 28, 29};
        aColIndices = new int[]{0, 1, 2, 9, 1, 7, 1, 2, 3, 7, 8, 11, 1, 7, 2, 9, 11, 7, 0, 2, 4, 5, 6, 9, 1, 6, 9, 10, 4};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        bShape = new Shape(12, 12);
        bData = new double[]{0.60519, 0.32299, 0.80625, 0.28575, 0.34192, 0.89753, 0.92196, 0.23285, 0.32287, 0.43847, 0.70881, 0.79414, 0.71232, 0.8306, 0.09576, 0.10497, 0.99244, 0.27191, 0.65053, 0.16913, 0.27482, 0.14437, 0.36674, 0.07254, 0.97292, 0.47324, 0.09344, 0.26302, 0.2397};
        bRowPointers = new int[]{0, 3, 3, 4, 6, 9, 10, 14, 17, 20, 25, 29, 29};
        bColIndices = new int[]{0, 6, 10, 2, 3, 8, 1, 4, 11, 2, 3, 4, 7, 9, 0, 4, 6, 0, 6, 10, 0, 2, 6, 8, 9, 4, 7, 10, 11};
        b = new CsrMatrix(bShape, bData, bRowPointers, bColIndices);

        expShape = new Shape(12, 12);
        expData = new double[]{0.0135199446, 0.2380652192, 0.29507196280000003, 0.6979513788, 0.2820107499, 0.192258982, 0.23938684489999998, 0.5217380706, 0.09550256199999999};
        expRowPointers = new int[]{0, 1, 1, 1, 3, 4, 5, 6, 6, 8, 8, 9, 9};
        expColIndices = new int[]{0, 3, 8, 1, 2, 9, 0, 6, 10};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.elemMult(b));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(14, 16);
        aData = new double[]{0.70348, 0.43878, 0.5199, 0.78916, 0.634, 0.24166, 0.14356, 0.99185, 0.90768};
        aRowPointers = new int[]{0, 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9};
        aColIndices = new int[]{2, 5, 15, 4, 8, 12, 13, 6, 3};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        bShape = new Shape(14, 16);
        bData = new double[]{0.88271, 0.31581, 0.41842, 0.67326};
        bRowPointers = new int[]{0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 4};
        bColIndices = new int[]{14, 11, 8, 1};
        b = new CsrMatrix(bShape, bData, bRowPointers, bColIndices);

        expShape = new Shape(14, 16);
        expData = new double[]{};
        expRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.elemMult(b));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(16, 5);
        aData = new double[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        a = new CsrMatrix(aShape, aData, aRowPointers, aColIndices);

        bShape = new Shape(16, 5);
        bData = new double[]{};
        bRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        bColIndices = new int[]{};
        b = new CsrMatrix(bShape, bData, bRowPointers, bColIndices);

        expShape = new Shape(16, 5);
        expData = new double[]{};
        expRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{};
        exp = new CsrMatrix(expShape, expData, expRowPointers, expColIndices);

        assertEquals(exp, a.elemMult(b));

        // ------------------- sub-case 4 -------------------
        aShape = new Shape(16, 5);
        a = new CsrMatrix(aShape);
        bShape = new Shape(16, 4);
        b = new CsrMatrix(bShape);

        CsrMatrix finalA = a;
        CsrMatrix finalB = b;
        assertThrows(NDArrayShapeException.class, ()-> finalA.elemMult(finalB));

        aShape = new Shape(15156, 95314);
        a = new CsrMatrix(aShape);
        bShape = new Shape(132, 235);
        b = new CsrMatrix(bShape);

        CsrMatrix finalA1 = a;
        CsrMatrix finalB1 = b;
        assertThrows(NDArrayShapeException.class, ()-> finalA1.elemMult(finalB1));
    }
}
