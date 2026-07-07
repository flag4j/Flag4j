package org.flag4j.arrays.dense.matrix;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixGetTests {

    static double[][] aEntries;
    double[][] expEntries;
    static Matrix A;
    Matrix exp;

    @BeforeAll
    static void setup() {
        aEntries = new double[][]{
                {1.123, 5525, 66.74},
                {-8234.5, 15.22, -84.12},
                {234, 8, 1},
                {-9.451, -45.6, 111.345}};
        A = new Matrix(aEntries);
    }


    @Test
    void getRowTestCase() {
        Vector exp;

        // ------------------- sub-case 1 -------------------
        exp = new Vector(-8234.5, 15.22, -84.12);
        assertEquals(exp, A.getRow(1));

        // ------------------- sub-case 2 -------------------
        exp = new Vector(1.123, 5525, 66.74);
        assertEquals(exp, A.getRow(0));

        // ------------------- sub-case 3 -------------------
        exp = new Vector(-9.451, -45.6, 111.345);
        assertEquals(exp, A.getRow(3));

        // ------------------- sub-case 4 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(13));

        // ------------------- sub-case 5 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(-1));
    }


    @Test
    void getRowAfterTestCase() {
        Vector exp;

        // ------------------- sub-case 1 -------------------
        exp = new Vector(15.22, -84.12);
        assertEquals(exp, A.getRow(1, 1, A.numCols));

        // ------------------- sub-case 2 -------------------
        exp = new Vector(66.74);
        assertEquals(exp, A.getRow(0, 2, A.numCols));

        // ------------------- sub-case 3 -------------------
        exp = new Vector(-9.451, -45.6, 111.345);
        assertEquals(exp, A.getRow(3, 0, A.numCols));

        // ------------------- sub-case 4 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(13));

        // ------------------- sub-case 5 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(-1));

        // ------------------- sub-case 6 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(1, -1, 2));

        // ------------------- sub-case 7 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(1, 4, 5));
    }


    @Test
    void getSliceTestCase() {
        // ------------------- sub-case 1 -------------------
        expEntries = new double[][]{{-8234.5, 15.22}, {234, 8}, {-9.451, -45.6}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.getSlice(1, 4, 0, 2));


        // ------------------- sub-case 2 -------------------
        expEntries = new double[][]{{15.22}, {8}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.getSlice(1, 3, 1, 2));
    }


    @Test
    void getColTestCase() {
        Vector exp;

        // ------------------- sub-case 1 -------------------
        exp = new Vector(5525, 15.22, 8, -45.6);
        assertEquals(exp, A.getCol(1));

        // ------------------- sub-case 2 -------------------
        exp = new Vector(1.123, -8234.5, 234, -9.451);
        assertEquals(exp, A.getCol(0));

        // ------------------- sub-case 3 -------------------
        exp = new Vector(66.74, -84.12, 1, 111.345);
        assertEquals(exp, A.getCol(2));

        // ------------------- sub-case 4 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(13));

        // ------------------- sub-case 5 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(-1));
    }
}
