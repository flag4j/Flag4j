package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MatrixConstructorTests {
    double[] expEntriesA;
    double[][] expEntriesA2d;
    int[][] expEntriesAint2d;
    int expNumColsA, expNumRowsA;
    Matrix A, B;
    Shape shape;

    /**
     * Tests size constructor for the matrix class.
     */
    @Test
    void sizeConstructorTestCase() {
        // ------------ sub-case 1 ------------
        int size = 0;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 2 ------------
        size = 1;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 3 ------------
        size = 145;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 4 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-1));

        // ------------ sub-case 5 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-5));
    }


    /**
     * Tests size constructor for the matrix class.
     */
    @Test
    void sizeFillConstructorTestCase() {
        // ------------ sub-case 1 ------------
        int size = 0;
        double fillValue = -1231.012;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 2 ------------
        size = 15;
        fillValue = 13231.12321;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 3 ------------
        size = 145;
        fillValue = 0;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 4 ------------
        size = 13;
        fillValue = -1231;
        expEntriesA = new double[size*size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 5 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-1));

        // ------------ sub-case 6 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-5));
    }


    /**
     * Tests constructor which takes the number of rows and columns.
     */
    @Test
    void rowColConstructorTestCase() {
        // ------------- sub-case 1 ------------------
        expNumRowsA = 0;
        expNumColsA = 0;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumRowsA);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------- sub-case 2 ------------------
        expNumRowsA = 1;
        expNumColsA = 0;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------- sub-case 3 ------------------
        expNumRowsA = 0;
        expNumColsA = 1;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------- sub-case 4 ------------------
        expNumRowsA = 12;
        expNumColsA = 51;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------- sub-case 5 ------------------
        expNumRowsA = 15;
        expNumColsA = 1;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());


        // ------------ sub-case 6 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, 2));

        // ------------ sub-case 7 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(2, -2));

        // ------------ sub-case 8 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, -2));
    }


    @Test
    void rowColFillConstructorTestCase() {
        // ------------ sub-case 1 ------------
        expNumRowsA = 0;
        expNumColsA = 1;
        double fillValue = -1231.012;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 2 ------------
        expNumRowsA = 1;
        expNumColsA = 0;
        fillValue = 13231.12321;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 3 ------------
        expNumRowsA = 12;
        expNumColsA = 14;
        fillValue = 0;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 4 ------------
        expNumRowsA = 1;
        expNumColsA = 19;
        fillValue = -144;
        expEntriesA = new double[expNumRowsA*expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(expEntriesA, fillValue);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());

        // ------------ sub-case 5 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, 2, 45));

        // ------------ sub-case 6 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(2, -2, 45));

        // ------------ sub-case 7 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, -2, 45));
    }


    @Test
    void doubleEntriesConstructorTestCase() {
        // ------------ sub-case 1 ------------
        expEntriesA2d = new double[][]
                {{1, 5.2234, 1231.2344, -112, 3.3, 0, 2.3e-4},
                {0.4, 12, -44.3, Double.NEGATIVE_INFINITY, Math.PI, Double.NaN, Double.POSITIVE_INFINITY}};
        expEntriesA = Arrays.stream(expEntriesA2d)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        expNumRowsA = expEntriesA2d.length;
        expNumColsA = expEntriesA2d[0].length;
        double[] row = new double[expNumColsA];

        A = new Matrix(expEntriesA2d);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());
    }


    @Test
    void intEntriesConstructorTestCase() {
        // ------------ sub-case 1 ------------
        expEntriesA2d = new double[][]
                {{1, 2, 3},
                {0, -1, -3},
                {Integer.MAX_VALUE, Integer.MIN_VALUE, 00341}};
        expEntriesA = Arrays.stream(expEntriesA2d)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        expEntriesAint2d = new int[][]
                {{1, 2, 3},
                {0, -1, -3},
                {Integer.MAX_VALUE, Integer.MIN_VALUE, 00341}};
        expNumRowsA = expEntriesAint2d.length;
        expNumColsA = expEntriesAint2d[0].length;
        double[] row = new double[expNumColsA];

        A = new Matrix(expEntriesAint2d);

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());
    }


    @Test void matConstructorTestCase() {
        // ------------ sub-case 1 ------------
        expEntriesA2d = new double[][]
                {{1, 5.2234, 1231.2344, -112, 3.3, 0, 2.3e-4},
                        {0.4, 12, -44.3, Double.NEGATIVE_INFINITY, Math.PI, Double.NaN, Double.POSITIVE_INFINITY}};
        expEntriesA = Arrays.stream(expEntriesA2d)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        expNumRowsA = expEntriesA2d.length;
        expNumColsA = expEntriesA2d[0].length;
        double[] row = new double[expNumColsA];

        A = new Matrix(expEntriesA2d);
        B = new Matrix(A);

        assertArrayEquals(A.data, B.data);
        assertEquals(A.numRows(), B.numRows());
        assertEquals(A.numCols(), B.numCols());
    }


    @Test
    void shapeConstructorsTestCase() {
        shape = new Shape(6, 3);
        Matrix A = new Matrix(shape);
        expEntriesA2d = new double[6][3];
        expEntriesA = Arrays.stream(expEntriesA2d)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        expNumRowsA = 6;
        expNumColsA = 3;

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());
    }


    @Test
    void shapeValueConstructorsTestCase() {
        shape = new Shape(5, 9);
        Matrix A = new Matrix(shape, -834.34);
        expEntriesA2d = new double[][]
                {{-834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34},
                {-834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34},
                {-834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34},
                {-834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34},
                {-834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34, -834.34}};
        expEntriesA = Arrays.stream(expEntriesA2d)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        expNumRowsA = 5;
        expNumColsA = 9;

        assertArrayEquals(expEntriesA, A.data);
        assertEquals(expNumRowsA, A.numRows());
        assertEquals(expNumColsA, A.numCols());
    }
}
