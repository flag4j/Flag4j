package org.flag4j.arrays.sparse.sparse_complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCTensorGetSetTests {
    static CooCTensor A;
    static Shape aShape;
    static Complex128[] aEntries;
    static int[][] aIndices;

    static CooCTensor exp;
    static Shape expShape;
    static Complex128[] expEntries;
    static int[][] expIndices;


    @Test
    void cooTensorSetTest() {
        aShape = new Shape(3, 4, 5, 1);
        aEntries = new Complex128[]{
                new Complex128(1, 2), new Complex128(0.2324, -239.1),
                new Complex128(5.1), new Complex128(0, 18.2)};
        aIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        // --------------- sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new Complex128[]{
                new Complex128(1, 2), new Complex128(2.41, -23.23),
                new Complex128(5.1), new Complex128(0, 18.2)};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(new Complex128(2.41, -23.23), 1, 2, 0, 0));

        // --------------- sub-case 2 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new Complex128[]{
                new Complex128(1, 2), new Complex128(0.2324, -239.1), new Complex128(0, -9.2),
                new Complex128(5.1), new Complex128(0, 18.2)};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {1, 2, 4, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(new Complex128(0, -9.2), 1, 2, 4, 0));

        // --------------- sub-case 3 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new Complex128[]{
                new Complex128(0.001), new Complex128(1, 2), new Complex128(0.2324, -239.1),
                new Complex128(5.1), new Complex128(0, 18.2)};
        expIndices = new int[][]{{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(0.001, 0, 0, 0, 0));

        // --------------- sub-case 4 ---------------
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, -1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, 0, 0, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, 1, 2, 5, 0));
    }


    @Test
    void cooTensorGetTest() {
        aShape = new Shape(3, 4, 5, 1);
        aEntries = new Complex128[]{
                new Complex128(1, 2), new Complex128(0.2324, -239.1),
                new Complex128(5.1), new Complex128(0, 18.2)};
        aIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 0}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        // --------------- sub-case 1 ---------------
        assertEquals(new Complex128(0.2324, -239.1), A.get(1, 2, 0, 0));

        // --------------- sub-case 2 ---------------
        assertEquals(Complex128.ZERO, A.get(1, 2, 4, 0));

        // --------------- sub-case 3 ---------------
        assertEquals(Complex128.ZERO, A.get(0, 0, 0, 0));

        // --------------- sub-case 4 ---------------
        assertEquals(new Complex128(1, 2), A.get(0, 1, 0, 0));

        // --------------- sub-case 5 ---------------
        assertEquals(new Complex128(5.1), A.get(2, 3, 2, 0));

        // --------------- sub-case 6 ---------------
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(-1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(0, 0, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(1, 2, 5, 0));
    }
}
