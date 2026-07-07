package org.flag4j.linalg.ops.sparse.coo.complex;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatMult.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexSparseMatMultTests {

    Complex128[][] expEntries;
    Complex128[] aEntries, bEntries, bVectorEntries, act;

    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, indices;

    Shape aShape, bShape;

    CooCMatrix A, B;
    CooCVector bVector;
    CMatrix exp;

    @Test
    void matMultTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new Complex128[]{new Complex128("1+3.45i"), new Complex128("9.43-8j")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{2, 1};
        aShape = new Shape(4, 3);
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[]{new Complex128("1.34+13.4i"), new Complex128("9234")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("9234.0+31857.300000000003i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        act = new Complex128[A.numRows*B.numCols];
        standard(A.data, A.rowIndices, A.colIndices, A.shape,
                B.data, B.rowIndices, B.colIndices, B.shape, act);
        assertArrayEquals(exp.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentStandard(A.data, A.rowIndices, A.colIndices, A.shape,
                B.data, B.rowIndices, B.colIndices, B.shape, act);
        assertArrayEquals(exp.data, act);
    }


    @Test
    void matVecMultTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new Complex128[]{new Complex128("1+3.45i"), new Complex128("7.9-105j"), new Complex128("9.43-8j")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 0, 1};
        aShape = new Shape(4, 3);
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[]{new Complex128("1.34+13.4i")};
        indices = new int[]{1};
        bVector = new CooCVector(3, bEntries, indices);

        expEntries = new Complex128[][]{{new Complex128("0.0")},
                {new Complex128("0.0")},
                {new Complex128("119.8362+115.642i")},
                {new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        Complex128[] act = new Complex128[A.numRows];
        standardVector(A.data, A.rowIndices, A.colIndices, A.shape,
                bVector.data, bVector.indices, act);
        assertArrayEquals(exp.data, act);

        act = new Complex128[A.numRows];
        concurrentStandardVector(A.data, A.rowIndices, A.colIndices, A.shape,
                bVector.data, bVector.indices, act);
        assertArrayEquals(exp.data, act);
    }
}
