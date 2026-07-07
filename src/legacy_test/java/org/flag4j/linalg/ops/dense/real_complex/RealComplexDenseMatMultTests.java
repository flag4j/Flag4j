package org.flag4j.linalg.ops.dense.real_complex;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseMatMult.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealComplexDenseMatMultTests {
    double[][] aEntries;
    Complex128[][] expCEntries;

    Matrix A;
    CMatrix expC;

    @Test
    void matMultTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("1.8715844-91.6141568794i"), new Complex128("8768.731856002458-10.238294909999999i")},
                {new Complex128("-1553.4617-1447.705376035i"), new Complex128("37924.640881531595+8428.0382634i")},
                {new Complex128("205.65936999999997+123.444878510567i"), new Complex128("1419.6289704199999-1126.188735i")},
                {new Complex128("130.337844+66.8009098206i"), new Complex128("1978.9472913999998-846470.621682i")}};
        expC = new CMatrix(expCEntries);

        Complex128[] act = new Complex128[A.numRows*B.numCols];

        standard(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        reordered(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        blocked(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        blockedReordered(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentStandard(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentReordered(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentBlocked(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentBlockedReordered(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("-10721.303415600001+8507.86475i"), new Complex128("6505.848844-4930.73205i"), new Complex128("-10.619125082+8.42448366i")},
                {new Complex128("-82273.7933-1.04985560794i"), new Complex128("48647.8159-92.7375568794i"), new Complex128("-81.47704028-0.00011494769430000002i")},
                {new Complex128("-0.018649000000000002-79472.71350000001i"), new Complex128("0.011027000000000002+46991.56050000001i"), new Complex128("-0.0000184684-78.7030866i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[B.numRows*A.numCols];
        standard(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        reordered(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        blocked(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        blockedReordered(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        concurrentStandard(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        concurrentReordered(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        concurrentReordered(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows*A.numCols];
        concurrentBlockedReordered(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);
    }


    @Test
    void matVecMultRealComplexTestCase() {
        Complex128[][] bEntries;
        Complex128[] act;
        CMatrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i")},
                {new Complex128("-0.0-0.9345341i")},
                {new Complex128("0.0")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("1.8715844-91.6141568794i")},
                {new Complex128("-1553.4617-1447.705376035i")},
                {new Complex128("205.65936999999997+123.444878510567i")},
                {new Complex128("130.337844+66.8009098206i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[A.numRows*B.numCols];
        standardVector(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        blockedVector(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentStandardVector(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numCols];
        concurrentBlockedVector(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("-10721.303415600001+8507.86475i")},
                {new Complex128("-82273.7933-1.04985560794i")},
                {new Complex128("-0.018649000000000002-79472.71350000001i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[B.numRows];
        standardVector(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows];
        blockedVector(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows];
        concurrentStandardVector(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[B.numRows];
        concurrentBlockedVector(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);
    }
}
