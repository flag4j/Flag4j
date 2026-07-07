package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.Shape;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense.real.RealDenseTranspose.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RealDenseTransposeTests {

    Shape shape;
    double[] A;
    double[] expTranspose;
    int numRows, numCols;
    int axis1, axis2;


    @Test
    void standardTestCase() {
        // ------------- sub-case 1 ---------------
        axis1 = 0;
        axis2 = 1;
        shape = new Shape(3, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, standard(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standard(A, shape, axis2, axis1));

        // ------------- sub-case 2 ---------------
        axis1 = 0;
        axis2 = 1;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24};

        assertArrayEquals(expTranspose, standard(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standard(A, shape, axis2, axis1));

        // ------------- sub-case 3 ---------------
        axis1 = 0;
        axis2 = 2;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3, 11, 19, 7, 15, 23, 4, 12, 20, 8, 16, 24};
        assertArrayEquals(expTranspose, standard(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standard(A, shape, axis2, axis1));

        // ------------- sub-case 4 ---------------
        axis1 = 1;
        axis2 = 2;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16, 17, 21, 18, 22, 19, 23, 20, 24};
        assertArrayEquals(expTranspose, standard(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standard(A, shape, axis2, axis1));

        // ------------- sub-case 5 ---------------
        axis1 = 1;
        axis2 = 1;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        assertArrayEquals(expTranspose, standard(A, shape, axis1, axis2));

        // ------------- sub-case 6 ---------------
        axis1 = 0;
        axis2 = 1;
        shape = new Shape(3);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        assertThrows(IllegalArgumentException.class, ()->standard(A, shape, axis1, axis2));
    }


    @Test
    void standardConcurrentTestCase() {
        // ------------- sub-case 1 ---------------
        axis1 = 0;
        axis2 = 1;
        shape = new Shape(3, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis2, axis1));

        // ------------- sub-case 2 ---------------
        axis1 = 0;
        axis2 = 1;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24};

        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis2, axis1));

        // ------------- sub-case 3 ---------------
        axis1 = 0;
        axis2 = 2;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3, 11, 19, 7, 15, 23, 4, 12, 20, 8, 16, 24};
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis2, axis1));

        // ------------- sub-case 4 ---------------
        axis1 = 1;
        axis2 = 2;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16, 17, 21, 18, 22, 19, 23, 20, 24};
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis1, axis2));
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis2, axis1));

        // ------------- sub-case 5 ---------------
        axis1 = 1;
        axis2 = 1;
        shape = new Shape(3, 2, 4);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        assertArrayEquals(expTranspose, standardConcurrent(A, shape, axis1, axis2));

        // ------------- sub-case 6 ---------------
        axis1 = 0;
        axis2 = 1;
        shape = new Shape(3);
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        assertThrows(IllegalArgumentException.class, ()->standardConcurrent(A, shape, axis1, axis2));
    }


    @Test
    void standardMatrixTestCase() {
        // ------------- sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));

        // ------------- sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, standardMatrix(A, numRows, numCols));
    }


    @Test
    void blockedMatrixTestCase() {
        // ------------- sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));

        // ------------- sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, blockedMatrix(A, numRows, numCols));
    }


    @Test
    void standardMatrixConcurrentTestCase() {
        // ------------- sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));

        // ------------- sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, standardMatrixConcurrent(A, numRows, numCols));
    }


    @Test
    void blockedMatrixConcurrentTestCase() {
        // ------------- sub-case 1 ---------------
        numRows = 3;
        numCols = 4;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- sub-case 2 ---------------
        numRows = 6;
        numCols = 2;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 12;
        numCols = 1;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        expTranspose = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));

        // ------------- sub-case 3 ---------------
        numRows = 3;
        numCols = 3;
        A = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        expTranspose = new double[]{1, 4, 7, 2, 5, 8, 3, 6, 9};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));


        // ------------- sub-case 3 ---------------
        numRows = 1;
        numCols = 1;
        A = new double[]{1.13};
        expTranspose = new double[]{1.13};

        assertArrayEquals(expTranspose, blockedMatrixConcurrent(A, numRows, numCols));
    }
}
