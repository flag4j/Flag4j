package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixRowColSwapTests {

    CsrCMatrix A;
    CsrCMatrix exp;
    Complex128[][] aEntries;
    Complex128[][] expEntries;

    @Test
    void rowSwapTests() {
        // ---------------------- sub-case new Complex128(1) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60.1, -15), new Complex128(70.2, 15.34),
                        new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80,2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60.1, -15), new Complex128(70.2, 15.34),
                        new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(1, 2));

        // ---------------------- sub-case new Complex128(2) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(2, 1));

        // ---------------------- sub-case new Complex128(3) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(2, 0));

        // ---------------------- sub-case new Complex128(4) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(0, 3));

        // ---------------------- sub-case new Complex128(5) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(3, 2));

        // ---------------------- sub-case new Complex128(6) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(30, 7.2), new Complex128(0), new Complex128(40), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();

        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(-1, 2));
        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(1, 145));
        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(0, 145));
    }


    @Test
    void colSwapTests() {
        // ---------------------- sub-case new Complex128(1) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(40), new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(0),  new Complex128(20, 1.5), new Complex128(0),  new Complex128(10, -9.1), new Complex128(0),  new Complex128(0)},
                {new Complex128(40), new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0)},
                {new Complex128(60, -15), new Complex128(0),  new Complex128(50, 56), new Complex128(0),  new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 3));

        // ---------------------- sub-case new Complex128(2) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(40), new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(0),  new Complex128(20, 1.5), new Complex128(0),  new Complex128(10, -9.1), new Complex128(0),  new Complex128(0)},
                {new Complex128(40), new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0)},
                {new Complex128(60, -15), new Complex128(0),  new Complex128(50, 56), new Complex128(0),  new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(3, 0));

        // ---------------------- sub-case new Complex128(3) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(40), new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(0),  new Complex128(0),  new Complex128(20, 1.5), new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(40), new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(0)},
                {new Complex128(0),  new Complex128(60, -15), new Complex128(50, 56), new Complex128(0),  new Complex128(70, 15.34), new Complex128(0)},
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(1, 3));

        // ---------------------- sub-case new Complex128(4) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(40), new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(0),  new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(10, -9.1)},
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(40), new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(80, 2.1), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) }};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 5));

        // ---------------------- sub-case new Complex128(5) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(40), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1),  new Complex128(0), new Complex128(90)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(0),  new Complex128(0),  new Complex128(20, 1.5), new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(60, -15), new Complex128(50, 56), new Complex128(40), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(0),  new Complex128(80, 2.1), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(90)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(1, 3));

        // ---------------------- sub-case new Complex128(6) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(40), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1),  new Complex128(0), new Complex128(90)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(40), new Complex128(50, 56), new Complex128(0),  new Complex128(70, 15.34), new Complex128(60, -15) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(90), new Complex128(0), new Complex128(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(3, 5));

        // ---------------------- sub-case new Complex128(7) ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(10, -9.1), new Complex128(20, 1.5), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(0),  new Complex128(40), new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1),  new Complex128(0), new Complex128(90)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new Complex128[][]{
                {new Complex128(20, 1.5), new Complex128(10, -9.1), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(30, 7.2), new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(0) },
                {new Complex128(40), new Complex128(0),  new Complex128(50, 56), new Complex128(60, -15), new Complex128(70, 15.34), new Complex128(0) },
                {new Complex128(0),  new Complex128(0),  new Complex128(0),  new Complex128(80, 2.1),  new Complex128(0), new Complex128(90)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 1));
    }
}
