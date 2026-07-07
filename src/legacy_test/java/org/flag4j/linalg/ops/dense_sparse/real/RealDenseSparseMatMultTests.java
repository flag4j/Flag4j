package org.flag4j.linalg.ops.dense_sparse.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatMult.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseSparseMatMultTests {
    double[] bEntries;
    int[] rowIndices, colIndices;
    CooMatrix B;
    Shape bShape;

    double[][] aEntries, expEntries;
    Matrix A, exp;

    @Test
    void matMultTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7375568794, 0.00143541},
                {-515.255376035, -10.7763114},
                {-0.00012148943299999999, 0.0},
                {-11.4330901794, -115804.09409999999}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.data, standard(A.data, A.shape, B.data, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(exp.data, concurrentStandard(A.data, A.shape, B.data, B.rowIndices, B.colIndices, B.shape));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(5, 3);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{0.0, 0.0, 0.0},
                {-1.04985560794, -92.7375568794, -0.00011494769430000002},
                {-10881.6915, 6434.2545, -10.7763114},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.data, standard(B.data, B.rowIndices, B.colIndices, B.shape, A.data, A.shape));
        assertArrayEquals(exp.data, concurrentStandard(B.data, B.rowIndices, B.colIndices, B.shape, A.data, A.shape));
    }

    @Test
    void matVecMultTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 0};
        bShape = new Shape(3, 1);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7361214694},
                {-526.0316874350001},
                {-0.00012148943299999999},
                {-115815.52719017939}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.data, standardVector(A.data, A.shape, B.data, B.rowIndices));
        assertArrayEquals(exp.data, concurrentStandardVector(A.data, A.shape, B.data, B.rowIndices));
        assertArrayEquals(exp.data, blockedVector(A.data, A.shape, B.data, B.rowIndices));
        assertArrayEquals(exp.data, concurrentBlockedVector(A.data, A.shape, B.data, B.rowIndices));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45},
                {123.445},
                {78.234}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(5, 4);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{0.0},
                {-1.04985560794},
                {-10881.6915},
                {0.0},
                {0.0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.data, standardVector(B.data, B.rowIndices, B.colIndices, B.shape, A.data, A.shape));
        assertArrayEquals(exp.data, concurrentStandardVector(B.data, B.rowIndices, B.colIndices, B.shape, A.data, A.shape));
    }
}
