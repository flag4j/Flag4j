package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixReshapeTests {
    Complex128[] entries = {
            new Complex128(1, -9.345), new Complex128(2), new Complex128(-3.444, -4371.5),
            new Complex128(-4), new Complex128(5), new Complex128(-6.44),
            new Complex128(7.77, -72.4667), new Complex128(8.435, 43.1), new Complex128(9),
            new Complex128(10.4, 156), new Complex128(0, 11), new Complex128(12.2344)};
    CMatrix A = new CMatrix(new Shape(3, 4), entries);
    CMatrix B;
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
        assertThrows(IllegalArgumentException.class, ()->A.reshape(expShape));

        // --------------- sub-case 6 ---------------
        expShape = new Shape(12, 2);
        assertThrows(IllegalArgumentException.class, ()->A.reshape(expShape));
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
        assertThrows(IllegalArgumentException.class, ()->A.reshape(rows, cols));

        // --------------- sub-case 6 ---------------
        rows = 12;
        cols = 2;
        assertThrows(IllegalArgumentException.class, ()->A.reshape(rows, cols));
    }


    @Test
    void flattenTestCase() {
        // --------------- sub-case 1 ---------------
        expShape = new Shape(1, entries.length);
        B = A.flatten();
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(entries.length, 1);
        B = A.flatten(1);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(1, entries.length);
        B = A.flatten(0);
        assertEquals(expShape, B.shape);
        assertArrayEquals(A.data, B.data);

        // --------------- sub-cases 2-4 ---------------
        assertThrows(LinearAlgebraException.class, ()->A.flatten(-1));
        assertThrows(LinearAlgebraException.class, ()->A.flatten(4));
        assertThrows(LinearAlgebraException.class, ()->A.flatten(2));
    }
}
