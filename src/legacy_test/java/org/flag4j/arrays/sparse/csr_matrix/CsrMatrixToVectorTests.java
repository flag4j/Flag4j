package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrMatrixToVectorTests {
    CsrMatrix A;
    double[][] aEntries;

    CooVector exp;
    double[] expEntries;

    @Test
    void toVectorTests() {
        // ------------------------- sub-case 1 -------------------------
        aEntries = new double[][]{{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2}};
        A = new Matrix(aEntries).toCsr();
        expEntries = new double[]{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2};
        exp = new Vector(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- sub-case 2 -------------------------
        aEntries = new double[][]{{-9.2512}, {0}, {0},
                {0}, {0}, {2.2516},
                {0}, {0}, {6.2}};
        A = new Matrix(aEntries).toCsr();
        expEntries = new double[]{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2};
        exp = new Vector(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- sub-case 3 -------------------------
        aEntries = new double[][]{
                {-9.2512, 0},
                {0, 0},
                {0, 2.2516},
                {0, 0},
                {6.2, 0}};
        A = new Matrix(aEntries).toCsr();
        expEntries = new double[]{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2, 0};
        exp = new Vector(expEntries).toCoo();

        assertEquals(exp, A.toVector());
    }
}
