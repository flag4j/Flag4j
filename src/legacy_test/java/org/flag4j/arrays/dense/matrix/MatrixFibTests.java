package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixFibTests {
    double[][] aEntries, bEntries;
    double[] bSparseEntries;
    Matrix A, B;
    CooMatrix BSparse;
    Shape sparseShape;
    int[] rowIndices, colIndices;
    Double exp;

    @Test
    void matrixFibTestCase() {
        // ------------------------- sub-case 1 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{104, 0, 8.23},
                {0.11135, -8, -3.123},
                {-9.8, 109.4, 4},
                {0, 0, 1}};
        B = new Matrix(bEntries);
        exp = -639.980377;

        assertEquals(exp, A.fib(B));

        // ------------------------- sub-case 2 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{104, 0, 8.23},
                {0.11135, -8, -3.123},
                {-9.8, 109.4, 4}};
        B = new Matrix(bEntries);

        assertThrows(LinearAlgebraException.class, ()->A.fib(B));


        // ------------------------- sub-case 3 -------------------------
        aEntries = new double[][]{{1, -212},
                {100.9, 80.1},
                {12.3, 0},
                {9, 6}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{104, 0, 8.23},
                {0.11135, -8, -3.123},
                {-9.8, 109.4, 4},
                {0, 0, 1}};
        B = new Matrix(bEntries);

        assertThrows(LinearAlgebraException.class, ()->A.fib(B));
    }
}
