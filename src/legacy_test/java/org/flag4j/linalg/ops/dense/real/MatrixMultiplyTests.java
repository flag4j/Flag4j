package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MatrixMultiplyTests {
    double[][] entriesA, entriesB;
    Matrix A, B;
    double[] exp, act;

    @Test
    void squareTestCase() {
        entriesA = new double[][]{{1.1119431, 2.1, -3}, {1.33, -9.44, 1233.4}, {0.001, 0, -9.3}};
        entriesB = new double[][]{{10.3, 23, 4}, {1334.5, -13.4, 0}, {0.0013, 1, 34.5}};
        exp = new double[]
                {2813.89911393, -5.565308700000003, -99.0522276,
                -12582.377579999998, 1390.486, 42557.62,
                        -0.00179, -9.277000000000001, -320.846};

        A = new Matrix(entriesA);
        B = new Matrix(entriesB);

        // ------------ sub-case 1 ------------
        act = RealDenseMatMult.standard(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = RealDenseMatMult.reordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = RealDenseMatMult.blocked(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = RealDenseMatMult.blockedReordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 5 ------------
        act = RealDenseMatMult.concurrentStandard(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 6 ------------
        act = RealDenseMatMult.concurrentReordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 7 ------------
        act = RealDenseMatMult.concurrentBlocked(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 8 ------------
        act = RealDenseMatMult.concurrentBlockedReordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new double[][]{{1.1119431, 2.1, -3}, {1.33, -9.44, 1233.4},
                {0.001, 0, -9.3}, {9.444, 13.4, -93.4}};
        entriesB = new double[][]{{10.3, 23, 4, 16.4, 7.8, 1.233},
                {1334.5, -13.4, 0, 11.3346, -0.3331334, 6.245},
                {0.0013, 1, 34.5, 4.566, -581.4, 0}};
        exp = new double[]
                {2813.89911393, -5.565308700000003, -99.0522276, 28.340526839999995, 1752.1735760399997,
                        14.485525842300001, -12582.377579999998, 1390.486, 42557.62, 5546.517776000001,
                        -717085.241220704, -57.312909999999995, -0.00179, -9.277000000000001,
                        -320.846, -42.4474, 5407.027800000001, 0.0012330000000000002, 17979.45178, -55.74799999999999,
                        -3184.5240000000003, -119.69916, 54371.95921244, 95.32745200000001};

        A = new Matrix(entriesA);
        B = new Matrix(entriesB);

        // ------------ sub-case 1 ------------
        act = RealDenseMatMult.standard(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = RealDenseMatMult.reordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = RealDenseMatMult.blocked(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = RealDenseMatMult.blockedReordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 5 ------------
        act = RealDenseMatMult.concurrentStandard(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 6 ------------
        act = RealDenseMatMult.concurrentReordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 7 ------------
        act = RealDenseMatMult.concurrentBlocked(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 8 ------------
        act = RealDenseMatMult.concurrentBlockedReordered(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void columnVectorTestCase() {
        entriesA = new double[][]{{1.1119431, 2.1, -3}, {1.33, -9.44, 1233.4},
                {0.001, 0, -9.3}, {9.444, 13.4, -93.4}};
        entriesB = new double[][]{{981.3371}, {1.69063}, {-7.441}};
        exp = new double[]{1117.06434011901, -7888.510604200001, 70.18263710000001, 9985.3914144};

        A = new Matrix(entriesA);
        B = new Matrix(entriesB);

        // ------------ sub-case 1 ------------
        act = RealDenseMatMult.standardVector(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = RealDenseMatMult.blockedVector(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = RealDenseMatMult.concurrentStandardVector(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = RealDenseMatMult.concurrentBlockedVector(A.data, A.shape, B.data, B.shape);
        assertArrayEquals(exp, act);
    }
}
