package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixGetSetTests {

    static CsrMatrix A;
    static double[][] aEntries;
    static CsrMatrix exp;
    static double[][] expEntries;


    @Test
    void getTests() {
        // -------------------------- sub-case 1 --------------------------
        aEntries = new double[][]{
                {0, 1, 0, 0, 4, 0, 0},
                {0, 9.1, 0, 0, -1.4, 0, 0},
                {0, 1.5, 0, 0, 0, 0, 0},
                {0, 0, 0, 4, 0, 801.4, 15}};
        A = new Matrix(aEntries).toCsr();

        assertEquals(0, A.get(0, 0));
        assertEquals(1, A.get(0, 1));
        assertEquals(0, A.get(0, 2));
        assertEquals(0, A.get(0, 3));
        assertEquals(4, A.get(0, 4));
        assertEquals(0, A.get(0, 5));
        assertEquals(0, A.get(0, 6));

        assertEquals(0, A.get(1, 0));
        assertEquals(9.1, A.get(1, 1));
        assertEquals(0, A.get(1, 2));
        assertEquals(0, A.get(1, 3));
        assertEquals(-1.4, A.get(1, 4));
        assertEquals(0, A.get(1, 5));
        assertEquals(0, A.get(1, 6));

        assertEquals(0, A.get(2, 0));
        assertEquals(1.5, A.get(2, 1));
        assertEquals(0, A.get(2, 2));
        assertEquals(0, A.get(2, 3));
        assertEquals(0, A.get(2, 4));
        assertEquals(0, A.get(2, 5));
        assertEquals(0, A.get(2, 6));

        assertEquals(0, A.get(3, 0));
        assertEquals(0, A.get(3, 1));
        assertEquals(0, A.get(3, 2));
        assertEquals(4, A.get(3, 3));
        assertEquals(0, A.get(3, 4));
        assertEquals(801.4, A.get(3, 5));
        assertEquals(15, A.get(3, 6));

        // -------------------------- sub-case 2 --------------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(-1, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(4, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, 7));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(20, 15));
    }


    @Test
    void setTests() {
        aEntries = new double[][]{
                {0, 1, 0, 0, 4, 0, 0},
                {0, 9.1, 0, 0, -1.4, 0, 0},
                {0, 1.5, 0, 0, 0, 0, 0},
                {0, 0, 0, 4, 0, 801.4, 15}};
        A = new Matrix(aEntries).toCsr();

        // -------------------------- sub-case 1 --------------------------
        expEntries = new double[][]{
                {0, 1, 0, 0, 4, 0, 0},
                {0, 9.1, 0, 0, 1004.4, 0, 0},
                {0, 1.5, 0, 0, 0, 0, 0},
                {0, 0, 0, 4, 0, 801.4, 15}};
        exp = new Matrix(expEntries).toCsr();
        assertEquals(exp, A.set(1004.4, 1, 4));

        // -------------------------- sub-case 2 --------------------------
        expEntries = new double[][]{
                {-2, 1, 0, 0, 4, 0, 0},
                {0, 9.1, 0, 0, -1.4, 0, 0},
                {0, 1.5, 0, 0, 0, 0, 0},
                {0, 0, 0, 4, 0, 801.4, 15}};
        exp = new Matrix(expEntries).toCsr();
        assertEquals(exp, A.set(-2.0, 0, 0));

        // -------------------------- sub-case 3 --------------------------
        expEntries = new double[][]{
                {0, 1, 0, 0, 4, 0, 5},
                {0, 9.1, 0, 0, -1.4, 0, 0},
                {0, 1.5, 0, 0, 0, 0, 0},
                {0, 0, 0, 4, 0, 801.4, 15}};
        exp = new Matrix(expEntries).toCsr();
        assertEquals(exp, A.set(5.0, 0, 6));

        // -------------------------- sub-case 4 --------------------------
        expEntries = new double[][]{
                {0, 1, 0, 0, 4, 0, 0},
                {0, 9.1, 0, 0, -1.4, 0, 0},
                {0, 1.5, 0, 0, 0, 0, 0},
                {0, 0, 0, 4, -992.5, 801.4, 15}};
        exp = new Matrix(expEntries).toCsr();
        assertEquals(exp, A.set(-992.5, 3, 4));

        // -------------------------- sub-case 5 --------------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, -1, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 0, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 4, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 0, 7));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 20, 15));
    }
}
