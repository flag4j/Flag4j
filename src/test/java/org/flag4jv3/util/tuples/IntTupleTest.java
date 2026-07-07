package org.flag4jv3.util.tuples;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class IntTupleTests {

    @Test
    void constructorDefensivelyCopiesInputArray() {
        int[] source = {1, 2, 3};

        IntTuple tuple = new IntTuple(source);
        source[0] = 99;

        assertArrayEquals(new int[] {1, 2, 3}, tuple.items());
    }


    @Test
    void itemsReturnsDefensiveCopy() {
        IntTuple tuple = new IntTuple(1, 2, 3);

        int[] copy = tuple.items();
        copy[1] = 99;

        assertArrayEquals(new int[] {1, 2, 3}, tuple.items());
    }


    @Test
    void supportsEmptyTuple() {
        IntTuple tuple = new IntTuple();

        assertEquals(0, tuple.size());
        assertArrayEquals(new int[0], tuple.items());
        assertEquals("IntTuple[items=()]", tuple.toString());
    }


    @Test
    void reportsSize() {
        assertEquals(0, new IntTuple().size());
        assertEquals(1, new IntTuple(42).size());
        assertEquals(3, new IntTuple(1, 2, 3).size());
    }


    @Test
    void equalTuplesAreEqualAndHaveSameHashCode() {
        IntTuple a = new IntTuple(1, -2, 3);
        IntTuple b = new IntTuple(1, -2, 3);

        assertEquals(a, b);
        assertEquals(b, a);
        assertEquals(a.hashCode(), b.hashCode());
    }


    @Test
    void unequalTuplesAreNotEqual() {
        IntTuple tuple = new IntTuple(1, 2, 3);

        assertNotEquals(new IntTuple(1, 2), tuple);
        assertNotEquals(new IntTuple(1, 3, 2), tuple);
        assertNotEquals(new IntTuple(1, 2, 4), tuple);
        assertNotEquals(null, tuple);
        assertNotEquals("(1, 2, 3)", tuple);
    }


    @Test
    void formatsToString() {
        IntTuple tuple = new IntTuple(1, -2, 3);

        assertEquals("IntTuple[items=(1, -2, 3)]", tuple.toString());
    }


    @Test
    void rejectsNullArray() {
        assertThrows(
                NullPointerException.class,
                () -> new IntTuple((int[]) null)
        );
    }
}