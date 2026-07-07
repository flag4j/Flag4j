package org.flag4j.linalg.ops.sparse.coo.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.sparse.coo.real.RealSparseMatMult.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealSparseMatMultTests {
    double[][] expEntries;
    double[] aEntries, bEntries, bVectorEntries;

    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, indices;

    Shape aShape, bShape;

    CooMatrix A, B;
    CooVector bVector;
    Matrix exp;

    @Test
    void matMultTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new double[]{1, 9.43};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{2, 1};
        aShape = new Shape(4, 3);
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{1.34, 9234};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0, 9234},
                {0, 0},
                {0, 0},
                {0, 0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.data, standard(A.data, A.rowIndices, A.colIndices, A.shape,
                B.data, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(exp.data, concurrentStandard(A.data, A.rowIndices, A.colIndices, A.shape,
                B.data, B.rowIndices, B.colIndices, B.shape));
    }


    @Test
    void matVecMultTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new double[]{1, 7.9, 9.43};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 0, 1};
        aShape = new Shape(4, 3);
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{1.34};
        indices = new int[]{1};
        bVector = CooVector.unsafeMake(3, bEntries, indices);

        expEntries = new double[][]{{0}, {0}, {12.6362}, {0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.data, standardVector(A.data, A.rowIndices, A.colIndices, A.shape,
                bVector.data, bVector.indices));
        assertArrayEquals(exp.data, concurrentStandardVector(A.data, A.rowIndices, A.colIndices, A.shape,
                bVector.data, bVector.indices));
    }
}








