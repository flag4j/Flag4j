package org.flag4j.arrays.sparse.sparse_tensor;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooTensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CooTensorConstructorTests {
    double[] expNonZero;
    int[] expNonZeroI;
    int[][] expIndices;
    Shape expShape;
    CooTensor A, B;

    @Test
    void shapeTestCase() {
        // --------------- sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expNonZero = new double[0];
        expIndices = new int[0][0];

        A = new CooTensor(expShape);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(100, 2);
        expNonZero = new double[0];
        expIndices = new int[0][0];

        A = new CooTensor(expShape);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expNonZero = new double[]{1, 223.1333, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};

        A = new CooTensor(expShape, expNonZero, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(100, 2);
        expNonZero = new double[]{1, 223.1333};
        expIndices = new int[][]{{98, 0}, {99, 1}};

        A = new CooTensor(expShape, expNonZero, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);


        // --------------- sub-case 3 ---------------
        expShape = new Shape(2, 2);
        expNonZero = new double[]{1, 223.1333, 50, 133, 1335.34};
        expIndices = new int[][]{{95, 1}, {96, 0}, {97, 1}, {98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensor(expShape, expNonZero, expIndices));

        // --------------- sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZero = new double[]{1, 223.1333, 50, 133};
        expIndices = new int[][]{{98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensor(expShape, expNonZero, expIndices));

        // --------------- sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZero = new double[]{1, 223.1333};
        expIndices = new int[][]{{98, 0, 1}, {99, 1, 8}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensor(expShape, expNonZero, expIndices));
    }


    @Test
    void shapeEntriesIntIndicesTestCase() {
        // --------------- sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expNonZeroI = new int[]{1, 223, -19, 2};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};

        A = new CooTensor(expShape, expNonZeroI, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);

        // --------------- sub-case 2 ---------------
        expShape = new Shape(100, 2);
        expNonZeroI = new int[]{1, 223};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expIndices = new int[][]{{98, 0}, {99, 1}};

        A = new CooTensor(expShape, expNonZeroI, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);

        // --------------- sub-case 3 ---------------
        expShape = new Shape(2, 2);
        expNonZeroI = new int[]{1, 223, 50, 133, 1335};
        expIndices = new int[][]{{95, 1}, {96, 0}, {97, 1}, {98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensor(expShape, expNonZeroI, expIndices));

        // --------------- sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZeroI = new int[]{1, 223, 50, 133};
        expIndices = new int[][]{{98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensor(expShape, expNonZeroI, expIndices));

        // --------------- sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZeroI = new int[]{1, 223};
        expIndices = new int[][]{{98, 0, 1}, {99, 1, 8}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensor(expShape, expNonZeroI, expIndices));
    }


    @Test
    void copyTestCase() {
        expShape = new Shape(3, 4, 5, 1);
        expNonZero = new double[]{1, 223.1333, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};

        B = new CooTensor(expShape, expNonZero, expIndices);
        A = new CooTensor(B);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.data);
        assertArrayEquals(expIndices, A.indices);
    }
}
