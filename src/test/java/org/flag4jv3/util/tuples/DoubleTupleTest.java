package org.flag4jv3.util.tuples;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DoubleTupleTests {

    @Test
    void constructorDefensivelyCopiesInputArray() {
        double[] source = {1.0, 2.0, 3.0};

        DoubleTuple tuple = new DoubleTuple(source);
        source[0] = 99.0;

        assertArrayEquals(new double[] {1.0, 2.0, 3.0}, tuple.items());
    }


    @Test
    void itemsReturnsDefensiveCopy() {
        DoubleTuple tuple = new DoubleTuple(1.0, 2.0, 3.0);

        double[] copy = tuple.items();
        copy[1] = 99.0;

        assertArrayEquals(new double[] {1.0, 2.0, 3.0}, tuple.items());
    }


    @Test
    void supportsEmptyTuple() {
        DoubleTuple tuple = new DoubleTuple();

        assertEquals(0, tuple.size());
        assertArrayEquals(new double[0], tuple.items());
        assertEquals("DoubleTuple[items=()]", tuple.toString());
    }


    @Test
    void reportsSize() {
        assertEquals(0, new DoubleTuple().size());
        assertEquals(1, new DoubleTuple(1.0).size());
        assertEquals(3, new DoubleTuple(1.0, 2.0, 3.0).size());
    }


    @Test
    void equalTuplesAreEqualAndHaveSameHashCode() {
        DoubleTuple a = new DoubleTuple(1.0, -2.5, 3.0);
        DoubleTuple b = new DoubleTuple(1.0, -2.5, 3.0);

        assertEquals(a, b);
        assertEquals(b, a);
        assertEquals(a.hashCode(), b.hashCode());
    }


    @Test
    void unequalTuplesAreNotEqual() {
        DoubleTuple tuple = new DoubleTuple(1.0, 2.0, 3.0);

        assertNotEquals(new DoubleTuple(1.0, 2.0), tuple);
        assertNotEquals(new DoubleTuple(1.0, 3.0, 2.0), tuple);
        assertNotEquals(new DoubleTuple(1.0, 2.0, 4.0), tuple);
        assertNotEquals(null, tuple);
        assertNotEquals("(1.0, 2.0, 3.0)", tuple);
    }


    @Test
    void equalityUsesArrayDoubleSemantics() {
        // Arrays.equals considers canonical NaN values equal.
        assertEquals(
                new DoubleTuple(Double.NaN),
                new DoubleTuple(Double.NaN)
        );

        // Arrays.equals distinguishes +0.0 and -0.0.
        assertNotEquals(
                new DoubleTuple(0.0),
                new DoubleTuple(-0.0)
        );
    }


    @Test
    void hashCodesDifferForDifferentContents() {
        DoubleTuple a = new DoubleTuple(1.0, 2.0);
        DoubleTuple b = new DoubleTuple(1.0, 3.0);

        assertNotEquals(a.hashCode(), b.hashCode());
    }


    @Test
    void formatsToString() {
        DoubleTuple tuple = new DoubleTuple(1.0, -2.5, Double.NaN);

        assertEquals(
                "DoubleTuple[items=(1.0, -2.5, NaN)]",
                tuple.toString()
        );
    }


    @Test
    void rejectsNullArray() {
        assertThrows(
                NullPointerException.class,
                () -> new DoubleTuple((double[]) null)
        );
    }
}