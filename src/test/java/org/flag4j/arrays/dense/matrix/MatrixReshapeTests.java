package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixReshapeTests {

    double[] entries = {1, 2, 3.444, -4, 5, -6.44, 7, 8, 9, 10, 11, 12.2344};
    Matrix A = new Matrix(new Shape(3, 4), entries);
    Matrix B;
    Shape expShape;
    int rows, cols;

    @Test
    void reshapeTestCase() {
        // --------------- sub-case 1 ---------------
        expShape = new Shape(4, 3);
        B = A.reshape(expShape);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(1, 12);
        B = A.reshape(expShape);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 3 ---------------
        expShape = new Shape(2, 6);
        B = A.reshape(expShape);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 4 ---------------
        expShape = new Shape(6, 2);
        B = A.reshape(expShape);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 5 ---------------
        expShape = new Shape(6, 1);
        assertThrows(NDArrayShapeException.class, ()->A.reshape(expShape));

        // --------------- sub-case 6 ---------------
        expShape = new Shape(12, 2);
        assertThrows(NDArrayShapeException.class, ()->A.reshape(expShape));
    }


    @Test
    void reshapeRowsColsTestCase() {
        // --------------- sub-case 1 ---------------
        rows = 4;
        cols = 3;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 2 ---------------
        rows = 1;
        cols = 12;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 3 ---------------
        rows = 2;
        cols = 6;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 4 ---------------
        rows = 6;
        cols = 2;
        expShape = new Shape(rows, cols);
        B = A.reshape(rows, cols);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 5 ---------------
        rows = 6;
        cols = 1;
        expShape = new Shape(rows, cols);
        assertThrows(NDArrayShapeException.class, ()->A.reshape(rows, cols));

        // --------------- sub-case 6 ---------------
        rows = 12;
        cols = 2;
        expShape = new Shape(12, 2);
        assertThrows(NDArrayShapeException.class, ()->A.reshape(rows, cols));
    }


    @Test
    void flattenTestCase() {
        // --------------- sub-case 1 ---------------
        expShape = new Shape(1, entries.length);
        B = A.flatten();
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(1, entries.length);
        B = A.flatten(1);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(entries.length, 1);
        B = A.flatten(0);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-cases 2-4 ---------------
        assertThrows(LinearAlgebraException.class, ()->A.flatten(-1));
        assertThrows(LinearAlgebraException.class, ()->A.flatten(4));
        assertThrows(LinearAlgebraException.class, ()->A.flatten(2));
    }
}
