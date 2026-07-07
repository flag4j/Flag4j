package org.flag4j.linalg.ops.dense.complex;

import org.flag4j.linalg.ops.dense.ring_ops.DenseRingHermitianTranspose;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense.DenseTranspose.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexDenseTransposeTests {
    Complex128[] A;
    Complex128[] expTranspose, expTransposeH;
    Complex128[] act;
    int numRows, numCols;


    @Test
    void transposeTestCase() {
        // ------------- sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new Complex128[]{
                new Complex128(1), new Complex128(2), new Complex128(3),
                new Complex128(4), new Complex128(5), new Complex128(6),
                new Complex128(7), new Complex128(8), new Complex128(9),
                new Complex128(10), new Complex128(11), new Complex128(12)
        };
        
        expTranspose = new Complex128[]{
                new Complex128(1), new Complex128(5), new Complex128(9),
                new Complex128(2), new Complex128(6), new Complex128(10),
                new Complex128(3), new Complex128(7), new Complex128(11),
                new Complex128(4), new Complex128(8), new Complex128(12)};
        expTransposeH = new Complex128[]{
                new Complex128(1).conj(), new Complex128(5).conj(), new Complex128(9).conj(),
                new Complex128(2).conj(), new Complex128(6).conj(), new Complex128(10).conj(),
                new Complex128(3).conj(), new Complex128(7).conj(), new Complex128(11).conj(),
                new Complex128(4).conj(), new Complex128(8).conj(), new Complex128(12).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols, null));

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTransposeH, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        // ------------- sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new Complex128[]{new Complex128(1), new Complex128(2), new Complex128(3),
                new Complex128(4), new Complex128(5), new Complex128(6),
                new Complex128(7), new Complex128(8), new Complex128(9),
                new Complex128(10), new Complex128(11), new Complex128(12)};
        expTranspose = new Complex128[]{
                new Complex128(1), new Complex128(3), new Complex128(5),
                new Complex128(7), new Complex128(9), new Complex128(11),
                new Complex128(2), new Complex128(4), new Complex128(6),
                new Complex128(8), new Complex128(10), new Complex128(12)};
        expTransposeH = new Complex128[]{
                new Complex128(1).conj(), new Complex128(3).conj(), new Complex128(5).conj(),
                new Complex128(7).conj(), new Complex128(9).conj(), new Complex128(11).conj(),
                new Complex128(2).conj(), new Complex128(4).conj(), new Complex128(6).conj(),
                new Complex128(8).conj(), new Complex128(10).conj(), new Complex128(12).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols, null));

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTransposeH, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        // ------------- sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new Complex128[]{new Complex128(1), new Complex128(2), new Complex128(3),
                new Complex128(4), new Complex128(5), new Complex128(6),
                new Complex128(7), new Complex128(8), new Complex128(9),
                new Complex128(10), new Complex128(11), new Complex128(12)};
        expTranspose = new Complex128[]{
                new Complex128(1), new Complex128(2), new Complex128(3),
                new Complex128(4), new Complex128(5), new Complex128(6),
                new Complex128(7), new Complex128(8), new Complex128(9),
                new Complex128(10), new Complex128(11), new Complex128(12)};
        expTransposeH = new Complex128[]{
                new Complex128(1).conj(), new Complex128(2).conj(), new Complex128(3).conj(),
                new Complex128(4).conj(), new Complex128(5).conj(), new Complex128(6).conj(),
                new Complex128(7).conj(), new Complex128(8).conj(), new Complex128(9).conj(),
                new Complex128(10).conj(), new Complex128(11).conj(), new Complex128(12).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols, null));

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTransposeH, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        // ------------- sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new Complex128[]{
                new Complex128(1), new Complex128(2), new Complex128(3),
                new Complex128(4), new Complex128(5), new Complex128(6),
                new Complex128(7), new Complex128(8), new Complex128(9)};
        expTranspose = new Complex128[]{
                new Complex128(1), new Complex128(4), new Complex128(7),
                new Complex128(2), new Complex128(5), new Complex128(8),
                new Complex128(3), new Complex128(6), new Complex128(9)};
        expTransposeH = new Complex128[]{
                new Complex128(1).conj(), new Complex128(4).conj(), new Complex128(7).conj(),
                new Complex128(2).conj(), new Complex128(5).conj(), new Complex128(8).conj(),
                new Complex128(3).conj(), new Complex128(6).conj(), new Complex128(9).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols, null));

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTransposeH, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        // ------------- sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new Complex128[]{new Complex128(1.13)};
        expTranspose = new Complex128[]{new Complex128(1.13)};
        expTransposeH = new Complex128[]{new Complex128(1.13).conj()};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols, null));
        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols, null));

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTransposeH, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.standardMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);

        act = new Complex128[A.length];
        DenseRingHermitianTranspose.blockedMatrixConcurrentHerm(A, numRows, numCols, act);
        assertArrayEquals(expTranspose, act);
    }
}
