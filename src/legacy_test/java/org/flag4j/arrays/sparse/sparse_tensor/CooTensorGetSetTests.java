package org.flag4j.arrays.sparse.sparse_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooTensorGetSetTests {
    static CooTensor A;
    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;

    static CooTensor exp;
    static Shape expShape;
    static double[] expEntries;
    static int[][] expIndices;

    static double expScalar;


    @Test
    void cooTensorSetTest() {
        aShape = new Shape(3, 4, 5, 1);
        aEntries = new double[]{1, 223.1333, -0.991233, 100.1234};
        aIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        // --------------- sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new double[]{1, -451.2, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(-451.2, 1, 2, 0, 0));

        // --------------- sub-case 2 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new double[]{1, 223.1333, 32.1, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {1, 2, 4, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(32.1, 1, 2, 4, 0));

        // --------------- sub-case 3 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new double[]{0.001, 1, 223.1333, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(0.001, 0, 0, 0, 0));

        // --------------- sub-case 4 ---------------
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5d, -1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5d, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5d, 0, 0, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5d, 1, 2, 5, 0));
    }


    @Test
    void cooTensorGetTest() {
        aShape = new Shape(3, 4, 5, 1);
        aEntries = new double[]{1, 223.1333, -0.991233, 100.1234};
        aIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        // --------------- sub-case 1 ---------------
        assertEquals(223.1333, A.get(1, 2, 0, 0));

        // --------------- sub-case 2 ---------------
        assertEquals(0, A.get(1, 2, 4, 0));

        // --------------- sub-case 3 ---------------
        assertEquals(0, A.get(0, 0, 0, 0));

        // --------------- sub-case 4 ---------------
        assertEquals(1, A.get(0, 1, 0, 0));

        // --------------- sub-case 5 ---------------
        assertEquals(-0.991233, A.get(2, 3, 2, 0));

        // --------------- sub-case 6 ---------------
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(-1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(0, 0, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(1, 2, 5, 0));
    }
}
