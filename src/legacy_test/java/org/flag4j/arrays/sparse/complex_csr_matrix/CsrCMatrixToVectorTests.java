package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrCMatrixToVectorTests {
    CsrCMatrix A;
    Complex128[][] aEntries;

    CooCVector exp;
    Complex128[] expEntries;

    @Test
    void toVectorTests() {
        // ------------------------- sub-case 1 -------------------------
        aEntries = new Complex128[][]{{new Complex128(1.23, -9.25), Complex128.ZERO, Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(1.526, -3.1),
                Complex128.ZERO, Complex128.ZERO, new Complex128(0, 1.2)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[]{new Complex128(1.23, -9.25), Complex128.ZERO, Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(1.526, -3.1),
                Complex128.ZERO, Complex128.ZERO, new Complex128(0, 1.2)};
        exp = new CVector(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- sub-case 2 -------------------------
        aEntries = new Complex128[][]{{new Complex128(1.23, -9.25)}, {Complex128.ZERO}, {Complex128.ZERO},
                {Complex128.ZERO}, {Complex128.ZERO}, {new Complex128(1.526, -3.1)},
                {Complex128.ZERO}, {Complex128.ZERO}, {new Complex128(0, 1.2)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[]{new Complex128(1.23, -9.25), Complex128.ZERO, Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(1.526, -3.1),
                Complex128.ZERO, Complex128.ZERO, new Complex128(0, 1.2)};
        exp = new CVector(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- sub-case 3 -------------------------
        aEntries = new Complex128[][]{
                {new Complex128(1.23, -9.25), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128(1.526, -3.1)},
                {Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0, 1.2), Complex128.ZERO}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[]{new Complex128(1.23, -9.25), Complex128.ZERO, Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(1.526, -3.1),
                Complex128.ZERO, Complex128.ZERO, new Complex128(0, 1.2), Complex128.ZERO};
        exp = new CVector(expEntries).toCoo();

        assertEquals(exp, A.toVector());
    }
}
