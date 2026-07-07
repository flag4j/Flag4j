package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexRealCsrCsrMatMultTests {
    static CsrCMatrix A;
    static CMatrix aDense;
    static Complex128[][] aEntries;
    static CsrMatrix B;
    static Matrix bDense;
    static double[][] bEntries;
    static CMatrix exp;
    static CsrCMatrix expCsr;

    private static void build(boolean... args) {
        aDense = new CMatrix(aEntries);
        A = aDense.toCsr();
        bDense = new Matrix(bEntries);
        B = bDense.toCsr();
        if(args.length ==0 || args[0]) {
            exp = aDense.mult(bDense);
            expCsr = exp.toCsr();
        }
    }


    @Test
    void multTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80.1, 2.5)},
                {new Complex128(0), new Complex128(1.41, -92.2), new Complex128(0), new Complex128(0, 15.5), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.25, 23.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(-999.1155, 2.25), new Complex128(-1, 1)}};
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
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bEntries = new double[][]{
                {12.5, 0},
                {0, -9.215},
                {0, 0},
                {1, 0}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bEntries = new double[][]{
                {0, 0.90836},
                {0, 0},
                {0.23691, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }


    @Test
    void mult2CsrTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80.1, 2.5)},
                {new Complex128(0), new Complex128(1.41, -92.2), new Complex128(0), new Complex128(0, 15.5), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.25, 23.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(-999.1155, 2.25), new Complex128(-1, 1)}};
        bEntries = new double[][]{
                {1.45, 0},
                {0, 0.3265},
                {0, 0},
                {0, 0.36597},
                {0.18312, 0},
                {0.40715, 0}};
        build();

        assertEquals(expCsr, A.mult2Csr(B));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bEntries = new double[][]{
                {12.5, 0},
                {0, -9.215},
                {0, 0},
                {1, 0}};
        build();

        assertEquals(expCsr, A.mult2Csr(B));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bEntries = new double[][]{
                {0, 0.90836},
                {0, 0},
                {0.23691, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult2Csr(B));
    }
}
