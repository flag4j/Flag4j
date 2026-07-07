package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixGetSetTests {

    static CsrCMatrix A;
    static Complex128[][] aEntries;
    static CsrCMatrix exp;
    static Complex128[][] expEntries;


    @Test
    void getTests() {
        // -------------------------- sub-case 1 --------------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(1, -72.1), new Complex128(0), new Complex128(0), new Complex128(4), new Complex128(0),
                        new Complex128(0)},
                {new Complex128(0), new Complex128(9.1), new Complex128(0), new Complex128(0), new Complex128(-1.4, 34.1), new Complex128(0),
                        new Complex128(0)},
                {new Complex128(0), new Complex128(0, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(4, -3), new Complex128(0), new Complex128(801.4, 15),
                        new Complex128(-15, 1)}};
        A = new CMatrix(aEntries).toCsr();

        assertEquals(new Complex128(0), A.get(0, 0));
        assertEquals(new Complex128(1, -72.1), A.get(0, 1));
        assertEquals(new Complex128(0), A.get(0, 2));
        assertEquals(new Complex128(0), A.get(0, 3));
        assertEquals(new Complex128(4), A.get(0, 4));
        assertEquals(new Complex128(0), A.get(0, 5));
        assertEquals(new Complex128(0), A.get(0, 6));

        assertEquals(new Complex128(0), A.get(1, 0));
        assertEquals(new Complex128(9.1), A.get(1, 1));
        assertEquals(new Complex128(0), A.get(1, 2));
        assertEquals(new Complex128(0), A.get(1, 3));
        assertEquals(new Complex128(-1.4, 34.1), A.get(1, 4));
        assertEquals(new Complex128(0), A.get(1, 5));
        assertEquals(new Complex128(0), A.get(1, 6));

        assertEquals(new Complex128(0), A.get(2, 0));
        assertEquals(new Complex128(0, 1.5), A.get(2, 1));
        assertEquals(new Complex128(0), A.get(2, 2));
        assertEquals(new Complex128(0), A.get(2, 3));
        assertEquals(new Complex128(0), A.get(2, 4));
        assertEquals(new Complex128(0), A.get(2, 5));
        assertEquals(new Complex128(0), A.get(2, 6));

        assertEquals(new Complex128(0), A.get(3, 0));
        assertEquals(new Complex128(0), A.get(3, 1));
        assertEquals(new Complex128(0), A.get(3, 2));
        assertEquals(new Complex128(4, -3), A.get(3, 3));
        assertEquals(new Complex128(0), A.get(3, 4));
        assertEquals(new Complex128(801.4, 15), A.get(3, 5));
        assertEquals(new Complex128(-15, 1), A.get(3, 6));

        // -------------------------- sub-case 2 --------------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(-1, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(4, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, 7));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(20, 15));
    }


    @Test
    void setTests() {
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(1, -72.1), new Complex128(0), new Complex128(0), new Complex128(4), new Complex128(0),
                        new Complex128(0)},
                {new Complex128(0), new Complex128(9.1), new Complex128(0), new Complex128(0), new Complex128(-1.4, 34.1), new Complex128(0),
                        new Complex128(0)},
                {new Complex128(0), new Complex128(0, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(4, -3), new Complex128(0), new Complex128(801.4, 15),
                        new Complex128(-15, 1)}};
        A = new CMatrix(aEntries).toCsr();

        // -------------------------- sub-case 1 --------------------------
        expEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(1, -72.1), new Complex128(0), new Complex128(0),
                        new Complex128(4), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(9.1), new Complex128(0), new Complex128(0),
                        new Complex128(9,-1.25), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0, 1.5), new Complex128(0), new Complex128(0),
                        new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(4, -3),
                        new Complex128(0), new Complex128(801.4, 15), new Complex128(-15, 1)}};
        exp = new CMatrix(expEntries).toCsr();
        assertEquals(exp, A.set(new Complex128(9,-1.25), 1, 4));

        // -------------------------- sub-case 2 --------------------------
        expEntries = new Complex128[][]{
                {new Complex128(6, -2), new Complex128(1, -72.1), new Complex128(0), new Complex128(0),
                        new Complex128(4), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(9.1), new Complex128(0), new Complex128(0),
                        new Complex128(-1.4, 34.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0, 1.5), new Complex128(0), new Complex128(0),
                        new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(4, -3),
                        new Complex128(0), new Complex128(801.4, 15), new Complex128(-15, 1)}};
        exp = new CMatrix(expEntries).toCsr();
        assertEquals(exp, A.set(new Complex128(6, -2), 0, 0));

        // -------------------------- sub-case 3 --------------------------
        expEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(1, -72.1), new Complex128(0), new Complex128(0),
                        new Complex128(4), new Complex128(0), new Complex128(5)},
                {new Complex128(0), new Complex128(9.1), new Complex128(0), new Complex128(0),
                        new Complex128(-1.4, 34.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0, 1.5), new Complex128(0), new Complex128(0),
                        new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(4, -3),
                        new Complex128(0), new Complex128(801.4, 15), new Complex128(-15, 1)}};
        exp = new CMatrix(expEntries).toCsr();
        assertEquals(exp, A.set(5.0, 0, 6));

        // -------------------------- sub-case 4 --------------------------
        expEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(1, -72.1), new Complex128(0), new Complex128(0),
                        new Complex128(4), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(9.1), new Complex128(0), new Complex128(0),
                        new Complex128(-1.4, 34.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0, 1.5), new Complex128(0), new Complex128(0),
                        new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(4, -3),
                        new Complex128(-5900.1, 2352.26221), new Complex128(801.4, 15), new Complex128(-15, 1)}};
        exp = new CMatrix(expEntries).toCsr();
        assertEquals(exp, A.set(new Complex128(-5900.1, 2352.26221), 3, 4));

        // -------------------------- sub-case 5 --------------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, -1, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 0, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 4, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 0, 7));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 20, 15));
    }
}
