package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorReshapeTests {


    static Complex128[] aEntries;
    static Shape aShape, expShape;
    static CTensor A, exp;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(0.000123, 1.3445), new Complex128(5.23523, -0.351), new Complex128(0, -834513.36), new Complex128(235.6),
                new Complex128(934, 294.5), new Complex128(13.5, 0.00134), new Complex128(-0.0, 3.1), new Complex128(0.1, Double.NEGATIVE_INFINITY),
                new Complex128(-345, 1.4), new Complex128(-0.00134, -8345.6), new Complex128(1.00015, -0.425), new Complex128(Double.POSITIVE_INFINITY)};
        aShape = new Shape(3, 2, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void reshapeTestCase() {
        // -------------------------- sub-case 1 --------------------------
        expShape = new Shape(1, 1, 12, 1);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));

        // -------------------------- sub-case 2 --------------------------
        expShape = new Shape(4, 3);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));

        // -------------------------- sub-case 3 --------------------------
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));

        // -------------------------- sub-case 4 --------------------------
        expShape = new Shape(2, 2, 3, 1);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));
    }


    @Test
    void flattenTestCase() {
        // -------------------------- sub-case 1 --------------------------
        expShape = new Shape(12);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten());

        // -------------------------- sub-case 2 --------------------------
        expShape = new Shape(1, 1, 12);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten(2));

        // -------------------------- sub-case 3 --------------------------
        expShape = new Shape(1, 12, 1);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten(1));

        // -------------------------- sub-case 4 --------------------------
        expShape = new Shape(12, 1, 1);
        exp = new CTensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten(0));

        // -------------------------- sub-case 5 --------------------------
        assertThrows(LinearAlgebraException.class, ()->A.flatten(-1));

        // -------------------------- sub-case 6 --------------------------
        assertThrows(LinearAlgebraException.class, ()->A.flatten(5));
    }
}
