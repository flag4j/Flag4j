package org.flag4j.linalg.ops.dense_sparse.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatrixMultTranspose.multTranspose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseSparseMatMultTransposeTests {
    double[] bEntries;
    int[] rowIndices, colIndices;
    CooMatrix B;
    Shape bShape;

    double[][] aEntries;
    Matrix A, exp;

    @Test
    void matMultTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 2};
        bShape = new Shape(2, 3);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        exp = A.mult(new CooMatrix(bShape.swapAxes(0, 1), bEntries, colIndices, rowIndices));

        assertArrayEquals(exp.data, multTranspose(A.data, A.shape, B.data, B.rowIndices, B.colIndices, B.shape));
    }
}
