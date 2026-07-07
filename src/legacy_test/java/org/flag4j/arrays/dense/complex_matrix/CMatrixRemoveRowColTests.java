package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixRemoveRowColTests {

    Complex128[][] aEntries, expEntries;
    int index;
    CMatrix A, exp;

    @Test
    void removeRowTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 1;
        expEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeRow(index));

        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 2;
        expEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeRow(index));

        // -------------------- sub-case 3 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = -1;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeRow(index));

        // -------------------- sub-case 4--------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 3;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeRow(index));
    }


    @Test
    void removeRowsTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeRows(0, 1));

        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 2;
        expEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeRows(index));

        // -------------------- sub-case 3 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeRows(-1, 1, 0));

        // -------------------- sub-case 4--------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeRows(0, 1, 4));
    }


    @Test
    void removeColTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 1;
        expEntries = new Complex128[][]{
                {new Complex128(1, 34.3)},
                {new Complex128(85.124, 51)},
                {new Complex128(26.24, 160.5)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeCol(index));

        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 0;
        expEntries = new Complex128[][]{
                {new Complex128(0.44, -9.4)},
                {new Complex128(3)},
                {new Complex128(0, -34.5)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeCol(index));

        // -------------------- sub-case 3 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = -1;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeCol(index));

        // -------------------- sub-case 4--------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        A = new CMatrix(aEntries);
        index = 2;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeCol(index));
    }


    @Test
    void removeColsTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4), new Complex128(3.4, 65.34)},
                {new Complex128(85.124, 51), new Complex128(3), new Complex128(3, 5.56)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5), new Complex128(Double.POSITIVE_INFINITY, 5.2)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(3.4, 65.34)},
                {new Complex128(3, 5.56)},
                {new Complex128(Double.POSITIVE_INFINITY, 5.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeCols(0, 1));

        // -------------------- sub-case 2 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4), new Complex128(3.4, 65.34)},
                {new Complex128(85.124, 51), new Complex128(3), new Complex128(3, 5.56)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5), new Complex128(Double.POSITIVE_INFINITY, 5.2)}};
        A = new CMatrix(aEntries);
        index = 2;
        expEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.removeCols(index));

        // -------------------- sub-case 3 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4), new Complex128(3.4, 65.34)},
                {new Complex128(85.124, 51), new Complex128(3), new Complex128(3, 5.56)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5), new Complex128(Double.POSITIVE_INFINITY, 5.2)}};
        A = new CMatrix(aEntries);
        assertThrows(IllegalArgumentException.class, ()->A.removeCols(-1, 0, 1, 2));

        // -------------------- sub-case 4--------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4), new Complex128(3.4, 65.34)},
                {new Complex128(85.124, 51), new Complex128(3), new Complex128(3, 5.56)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5), new Complex128(Double.POSITIVE_INFINITY, 5.2)}};
        A = new CMatrix(aEntries);
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.removeCols(0, 10, 4));
    }
}
