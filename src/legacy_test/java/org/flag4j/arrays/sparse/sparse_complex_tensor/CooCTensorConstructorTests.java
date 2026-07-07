package org.flag4j.arrays.sparse.sparse_complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCTensorConstructorTests {

    Shape expShape;
    Complex128[] expNonZero;
    double[] expNonZeroD;
    int[] expNonZeroI;
    int[][] expIndices;
    CooCTensor A, B;


    @Test
    void shapeTestCase() {
        // ------------ sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new Complex128[0];
        expIndices = new int[0][0];
        A = new CooCTensor(expShape);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.data.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.data[i]);
        }

        // ------------ sub-case 2 ------------
        expShape = new Shape(12);
        expNonZero = new Complex128[0];
        expIndices = new int[0][0];
        A = new CooCTensor(expShape);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.data.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.data[i]);
        }
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // ------------ sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new Complex128[]{new Complex128(1, -0.92342), new Complex128(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}};
        A = new CooCTensor(expShape, expNonZero, expIndices);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.data.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.data[i]);
        }

        // ------------ sub-case 2 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new Complex128[]{new Complex128(1, -0.92342), new Complex128(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}, {0, 1, 22, 3, 0, 10}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZero, expIndices));

        // ------------ sub-case 3 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new Complex128[]{new Complex128(1, -0.92342), new Complex128(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0}, {0, 1, 22, 2, 0}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZero, expIndices));

        // ------------ sub-case 4 ------------
        expShape = new Shape(2);
        expNonZero = new Complex128[]{new Complex128(1, -0.92342), new Complex128(-100, 123.44), new Complex128(0, 1)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZero, expIndices));
    }


    @Test
    void shapeEntriesIndicesDTestCase() {
        // ------------ sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZeroD = new double[]{1, 2, 3, 4.023423, -9233.2};
        expNonZero = new Complex128[expNonZeroD.length];
        ArrayConversions.toComplex128(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}, {0, 1, 24, 2, 0, 10},
                {0, 1, 26, 2, 0, 10}, {0, 1, 28, 3, 0, 10}};
        A = new CooCTensor(expShape, expNonZeroD, expIndices);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.data.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.data[i]);
        }

        // ------------ sub-case 2 ------------
        expShape = new Shape(5, 3);
        expNonZeroD = new double[]{1, 2, 3, 4.023423, -9233.2};
        expNonZero = new Complex128[expNonZeroD.length];
        ArrayConversions.toComplex128(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0, 0}, {1, 1}, {2, 2}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroD, expIndices));

        // ------------ sub-case 3 ------------
        expShape = new Shape(5, 3);
        expNonZeroD = new double[]{1, 2};
        expNonZero = new Complex128[expNonZeroD.length];
        ArrayConversions.toComplex128(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0, 0, 10, 1, 0}, {0, 1, 22, 2, 0}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroD, expIndices));

        // ------------ sub-case 4 ------------
        expShape = new Shape(2);
        expNonZeroD = new double[]{1, 2, 3, 4.023423, -9233.2};
        expNonZero = new Complex128[expNonZeroD.length];
        ArrayConversions.toComplex128(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0}, {1}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroD, expIndices));
    }


    @Test
    void copyTestCase() {
        // ------------ sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new Complex128[]{new Complex128(1, -0.92342), new Complex128(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}};
        B = new CooCTensor(expShape, expNonZero, expIndices);
        A = new CooCTensor(B);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.data.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.data[i]);
        }
    }
}
