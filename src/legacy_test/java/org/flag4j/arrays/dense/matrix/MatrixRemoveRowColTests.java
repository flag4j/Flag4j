package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixRemoveRowColTests {
    double[][] aEntries, expEntries;
    Matrix A, exp;
    int index;
    int[] indices;

    @Test
    void removeRowTestCase() {
        // ------------ sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 1;
        expEntries = new double[][]{{1, 2, 3}, {7, 8, 9}, {10, 11, 12}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeRow(index));

        // ------------ sub-case 2 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 3;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeRow(index));

        // ------------ sub-case 3 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = -1;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeRow(index));

        // ------------ sub-case 4 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 5;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeRow(index));
    }


    @Test
    void removeColTestCase() {
        // ------------ sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 1;
        expEntries = new double[][]{{1, 3}, {4, 6}, {7, 9}, {10, 12}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeCol(index));

        // ------------ sub-case 2 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 2;
        expEntries = new double[][]{{1, 2}, {4, 5}, {7, 8}, {10, 11}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeCol(index));

        // ------------ sub-case 3 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = -1;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeCol(index));

        // ------------ sub-case 4 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        index = 5;
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeCol(index));
    }


    @Test
    void removeRowsTestCase() {
        // ------------ sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{1, 2};
        expEntries = new double[][]{{1, 2, 3}, {10, 11, 12}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeRows(indices));

        // ------------ sub-case 2 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{0, 1, 3};
        expEntries = new double[][]{{7, 8, 9}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeRows(indices));

        // ------------ sub-case 3 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{0, -1, 3};
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeRows(indices));

        // ------------ sub-case 4 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{0, 1, 13};
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeRows(indices));
    }


    @Test
    void removeColsTestCase() {
        // ------------ sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{1, 2};
        expEntries = new double[][]{{1}, {4}, {7}, {10}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeCols(indices));

        // ------------ sub-case 2 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{1};
        expEntries = new double[][]{{1, 3}, {4, 6}, {7, 9}, {10, 12}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.removeCols(indices));

        // ------------ sub-case 3 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{0, -1, 3};
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeCols(indices));

        // ------------ sub-case 4 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);
        indices = new int[]{0, 1, 13};
        expEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        exp = new Matrix(expEntries);

        assertThrows(ArrayIndexOutOfBoundsException.class, ()-> A.removeCols(indices));
    }
}
