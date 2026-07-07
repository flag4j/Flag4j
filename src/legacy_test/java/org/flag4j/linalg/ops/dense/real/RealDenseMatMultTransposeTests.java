package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseMatMultTransposeTests {
    double[][] entriesA, entriesB;
    Matrix A, B;
    double[] exp, act;

    @Test
    void squareTestCase() {
        entriesA = new double[][]{{1.1119431, 2.1, -3}, {1.33, -9.44, 1233.4}, {0.001, 0, -9.3}};
        entriesB = new double[][]{{10.3, 23, 4}, {1334.5, -13.4, 0}, {0.0013, 1, 34.5}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);
        exp = A.mult(B.T()).data;

        // ------------ sub-case 1 ------------
        act = RealDenseMatMultTranspose.multTranspose(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = RealDenseMatMultTranspose.multTransposeBlocked(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = RealDenseMatMultTranspose.multTransposeConcurrent(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = RealDenseMatMultTranspose.multTransposeBlockedConcurrent(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new double[][]{
                {1.1119431, 2.1, -3},
                {1.33, -9.44, 1233.4},
                {0.001, 0, -9.3},
                {9.444, 13.4, -93.4}};
        entriesB = new double[][]{
                {10.3, 23, 4, 16.4, 7.8, 1.233},
                {1334.5, -13.4, 0, 11.3346, -0.3331334, 6.245},
                {0.0013, 1, 34.5, 4.566, -581.4, 0}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB).T();
        exp = A.mult(B.T()).data;

        // ------------ sub-case 1 ------------
        act = RealDenseMatMultTranspose.multTranspose(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = RealDenseMatMultTranspose.multTransposeBlocked(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = RealDenseMatMultTranspose.multTransposeConcurrent(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = RealDenseMatMultTranspose.multTransposeBlockedConcurrent(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);
    }
}
