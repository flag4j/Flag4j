package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RealCsrCsrMatMultTests {

    static CsrMatrix A;
    static Matrix aDense;
    static double[][] aEntries;
    static CsrMatrix B;
    static Matrix bDense;
    static double[][] bEntries;
    static Matrix exp;
    static CsrMatrix expCsr;

    private static void build(boolean... args) {
        aDense = new Matrix(aEntries);
        A = aDense.toCsr();
        bDense = new Matrix(bEntries);
        B = bDense.toCsr();
        if(args.length == 0 || args[0]) {
            exp = aDense.mult(bDense);
            expCsr = exp.toCsr();
        }
    }


    @Test
    void multTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 80.1},
                {0, 1.41, 0, 15.5, 0, 0},
                {0, -9.25, 0, 0, 0, 0},
                {0, 0, 0, 0, -999.1155, 1}};
        bEntries = new double[][]{
                {1.45, 0},
                {0, 0.3265},
                {0, 0},
                {0, 0.36597},
                {0.18312, 0},
                {0.40715, 0}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0},
                {-77.3, 0, 0, 0},
                {0, 0, 803.2, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, -9.345, 0, 0},
                {0, 0, 1.45, 0},
                {345, 2.4, 0, 0},
                {0, 0, 4.45, 0},
                {0, 0, 0, 1},};
        bEntries = new double[][]{
                {12.5, 0},
                {0, -9.215},
                {0, 0},
                {1, 0}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0},
                {-77.3, 0, 0, 0},
                {0, 0, 803.2, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, -9.345, 0, 0},
                {0, 0, 1.45, 0},
                {345, 2.4, 0, 0},
                {0, 0, 4.45, 0},
                {0, 0, 0, 1},};
        bEntries = new double[][]{
                {0, 0.90836},
                {0, 0},
                {0.23691, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }

    @Test
    void multAsCsrTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0, 0, 94.0149, 0, 0},
                {0, 1.4, 0, 1.3, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 4, 0, 5, 0, 0, 0, -9.34, 0},
                {0, 0, 0, 0, 0, 30.2, 0, 0, 0, 0}};
        bEntries = new double[][]{
                {0, 0, 0, 0, 0, 2.34, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 25.2, 0, 0, 0, 0, 0, 0},
                {1.23, 0, -99.141, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 690781.3, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30.13, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0.0002, 0, -81.3, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.1, 0},
                {0, 0, 0, 0, 4.34, 0, 0, 0, 0, 0, 0, 0},
                {0, 832.1, 241, 2, 5.4, 0, 0, 0, 0, 0, 4.1, 0}};
        build();

        assertEquals(expCsr, A.mult2Csr(B));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{
                {-9.31, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 15415.23, 0, 0},
                {0, 0, 0, 2.5, 0},
                {0, 0, 0, 0, 0},
                {0, -1986.5, 105.3, 0, 0.009034},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, -92.2, 0, 0},
                {20.5, 0, 0, 0, 0},
        };
        bEntries = new double[][]{
                {0, 0, 4.1},
                {6.025, 0, 0},
                {0, 0, 0},
                {0, 0, 0},
                {0, 5, 0}};
        build();

        assertEquals(expCsr, A.mult2Csr(B));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new double[][]{
                {-9.31, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 15415.23, 0, 0},
                {0, 0, 0, 2.5, 0},
                {0, 0, 0, 0, 0},
                {0, -1986.5, 105.3, 0, 0.009034},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, -92.2, 0, 0},
                {20.5, 0, 0, 0, 0},
        };
        bEntries = new double[][]{
                {0, 0, 4.1},
                {6.025, 0, 0},
                {0, 0, 0},
                {0, 0, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult2Csr(B));

        // ---------------------- sub-case 4 ----------------------
        aEntries = new double[][]{
                {-9.31, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 15415.23, 0, 0},
                {0, 0, 0, 2.5, 0},
                {0, 0, 0, 0, 0},
                {0, -1986.5, 105.3, 0, 0.009034},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, -92.2, 0, 0},
                {20.5, 0, 0, 0, 0},
        };
        bEntries = new double[][]{
                {0, 0, 4.1},
                {6.025, 0, 0},
                {0, 0, 0},
                {0, 0, 0},
                {0, 5, 0},
                {0, 0, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult2Csr(B));
    }
}
