package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CooMatrixConstructorTests {
    double[] expNonZero;
    int[] expNonZeroI, expRowIndices, expColIndices;
    int size, rows, cols;
    Shape expShape;
    CooMatrix A, B;

    @Test
    void sizeTestCase() {
        // --------------- sub-case 1 ---------------
        size = 5;
        expShape = new Shape(5, 5);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(size);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size));
    }


    @Test
    void rowsColsTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(rows, cols);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(rows, cols);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols));

        // --------------- sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols));

        // --------------- sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols));
    }


    @Test
    void shapeTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(expShape);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[0];
        expRowIndices = new int[0];
        expColIndices = new int[0];
        A = new CooMatrix(expShape);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 6};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};
        A = new CooMatrix(size, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        size = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- sub-case 3 ---------------
        size = 2;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 6, 5};
        expRowIndices = new int[]{0, 0, 1, 3, 4};
        expColIndices = new int[]{0, 4, 1, 3, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1, 4};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));

        // --------------- sub-case 4 ---------------
        size = 5;
        expShape = new Shape(size, size);
        expNonZero = new double[]{1, 2, 5, 4};
        expRowIndices = new int[]{0, 0, 1, 3};
        expColIndices = new int[]{0, 4, 1};

        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(size, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void rowColEntriesIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 10};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = -1;
        cols = 12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- sub-case 3 ---------------
        rows = 1;
        cols = -12;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));

        // --------------- sub-case 4 ---------------
        rows = -1;
        cols = -2;
        assertThrows(IllegalArgumentException.class, () -> new CooMatrix(rows, cols, expNonZero, expRowIndices, expColIndices));
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZero = new double[]{-1, 0.234345, 133.1, 24.5, -933.1};
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new CooMatrix(expShape, expNonZero, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void rowColEntriesIntIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }

    @Test
    void shapeEntriesIntIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        A = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        A = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }


    @Test
    void copyTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 10;
        cols = 12;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 1, 2, 3, 4};
        B = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);
        A = new CooMatrix(B);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);

        // --------------- sub-case 2 ---------------
        rows = 100123;
        cols = 99123341;
        expShape = new Shape(rows, cols);
        expNonZeroI = new int[]{-1, 234345, 133, 24, -933};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expRowIndices = new int[]{0, 4, 5, 5, 9};
        expColIndices = new int[]{0, 0, 0, 1, 0};
        B = new CooMatrix(expShape, expNonZeroI, expRowIndices, expColIndices);
        A = new CooMatrix(B);

        assertEquals(expShape, A.shape);
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expColIndices, A.colIndices);
        assertArrayEquals(expRowIndices, A.rowIndices);
    }
}
