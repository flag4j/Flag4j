package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixDetTests {

    double[][] aEntries;
    Matrix A;
    double exp;


    @Test
    void determinantTestCase() {
        // -------------------------- sub-case 1 --------------------------
        aEntries = new double[][]{{-234.56}};
        A = new Matrix(aEntries);
        exp = -234.56;

        assertEquals(exp, A.det().doubleValue());

        // -------------------------- sub-case 2 --------------------------
        aEntries = new double[][]{
                {32.45, 6.222},
                {-9.3455, 124.6}};
        A = new Matrix(aEntries);
        exp = 4101.417701;

        assertEquals(exp, A.det().doubleValue());


        // -------------------------- sub-case 3 --------------------------
        aEntries = new double[][]{{32.45, 6.222, 4.55},
                {-9.3455, 124.6, 4.566},
                {4.56, -845.46, 0.0003985}};
        A = new Matrix(aEntries);
        exp = 158765.67106657385;

        assertEquals(exp, A.det().doubleValue());


        // -------------------------- sub-case 4 --------------------------
        aEntries = new double[][]{{32.45, 6.222, 4.55, -3.5},
                {-9.3455, 124.6, 4.566, -0.34534},
                {4.56, -8.46, 0.0003985, 45.1},
                {3.46, 0.4356, 11.5, 45.78}};
        A = new Matrix(aEntries);
        exp = -2095944.4444574537;

        assertEquals(exp, A.det().doubleValue());


        // -------------------------- sub-case 5 --------------------------
        aEntries = new double[][]{{32.45, 6.222, 4.55, -3.5, 1.6},
                {-9.3455, 124.6, 4.566, -0.34534, -0.2345},
                {4.56, -8.46, 0.0003985, 45.1, -5.15},
                {3.46, 0.4356, 11.5, 45.78, 1.56},
                {0.00345, 0.349875, 0.345, 2.34e-06, -0.00024}};
        A = new Matrix(aEntries);
        exp = 419248.3838653321;

        assertEquals(exp, A.det().doubleValue());


        // -------------------------- sub-case 6 --------------------------
        aEntries = new double[][]{{32.45, 6.222, 4.55, -3.5, 1.6},
                {-9.3455, 124.6, 4.566, -0.34534, -0.2345},
                {4.56, -8.46, 0.0003985, 45.1, -5.15}};
        A = new Matrix(aEntries);

        assertThrows(LinearAlgebraException.class, ()->A.det());
    }
}
