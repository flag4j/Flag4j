package org.flag4jv3.util.tuples;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TupleTests {

    @Test
    void constructorDefensivelyCopiesInputArray() {
        String[] source = {"one", "two", "three"};

        Tuple<String> tuple = new Tuple<>(source);
        source[0] = "changed";

        assertArrayEquals(
                new String[] {"one", "two", "three"},
                tuple.items()
        );
    }


    @Test
    void itemsReturnsDefensiveCopy() {
        Tuple<String> tuple = new Tuple<>("one", "two", "three");

        String[] copy = tuple.items();
        copy[1] = "changed";

        assertArrayEquals(
                new String[] {"one", "two", "three"},
                tuple.items()
        );
    }


    @Test
    void supportsEmptyTuple() {
        Tuple<String> tuple = new Tuple<>();

        assertEquals(0, tuple.size());
        assertArrayEquals(new String[0], tuple.items());
        assertEquals("Tuple[items=()]", tuple.toString());
    }


    @Test
    void reportsSize() {
        assertEquals(0, new Tuple<>().size());
        assertEquals(1, new Tuple<>("value").size());
        assertEquals(3, new Tuple<>("a", "b", "c").size());
    }


    @Test
    void equalTuplesAreEqualAndHaveSameHashCode() {
        Tuple<String> a = new Tuple<>("one", "two", "three");
        Tuple<String> b = new Tuple<>("one", "two", "three");

        assertEquals(a, b);
        assertEquals(b, a);
        assertEquals(a.hashCode(), b.hashCode());
    }


    @Test
    void unequalTuplesAreNotEqual() {
        Tuple<String> tuple = new Tuple<>("one", "two", "three");

        assertNotEquals(new Tuple<>("one", "two"), tuple);
        assertNotEquals(new Tuple<>("one", "three", "two"), tuple);
        assertNotEquals(new Tuple<>("one", "two", "four"), tuple);
        assertNotEquals(null, tuple);
        assertNotEquals(new IntTuple(1, 2, 3), tuple);
    }


    @Test
    void supportsNullElementsIfAllowedByImplementation() {
        Tuple<String> tuple = new Tuple<>("one", null, "three");

        assertEquals(3, tuple.size());
        assertArrayEquals(
                new String[] {"one", null, "three"},
                tuple.items()
        );
    }


    @Test
    void formatsToString() {
        Tuple<String> tuple = new Tuple<>("one", null, "three");

        assertEquals(
                "Tuple[items=(one, null, three)]",
                tuple.toString()
        );
    }


    @Test
    void rejectsNullArray() {
        assertThrows(
                NullPointerException.class,
                () -> new Tuple<String>((String[]) null)
        );
    }
}