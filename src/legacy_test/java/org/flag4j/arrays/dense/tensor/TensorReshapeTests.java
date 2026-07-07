package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorReshapeTests {

    static double[] aEntries;
    static Shape aShape, expShape;
    static Tensor A, exp;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{
                0.000123, 5.23523, -834513.36, 235.6,
                934, 13.5, -0.0, 0.1,
                345, 8345.6, 1.00015, Double.POSITIVE_INFINITY};
        aShape = new Shape(3, 2, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void reshapeTestCase() {
        // -------------------------- sub-case 1 --------------------------
        expShape = new Shape(1, 1, 12, 1);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));

        // -------------------------- sub-case 2 --------------------------
        expShape = new Shape(4, 3);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));

        // -------------------------- sub-case 3 --------------------------
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));

        // -------------------------- sub-case 4 --------------------------
        expShape = new Shape(2, 2, 3, 1);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.reshape(expShape));
    }


    @Test
    void flattenTestCase() {
        // -------------------------- sub-case 1 --------------------------
        expShape = new Shape(12);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten());

        // -------------------------- sub-case 2 --------------------------
        expShape = new Shape(1, 1, 12);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten(2));

        // -------------------------- sub-case 3 --------------------------
        expShape = new Shape(1, 12, 1);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten(1));

        // -------------------------- sub-case 4 --------------------------
        expShape = new Shape(12, 1, 1);
        exp = new Tensor(expShape, Arrays.copyOf(aEntries, aEntries.length));

        assertEquals(exp, A.flatten(0));

        // -------------------------- sub-case 5 --------------------------
        assertThrows(LinearAlgebraException.class, ()->A.flatten(-1));

        // -------------------------- sub-case 6 --------------------------
        assertThrows(LinearAlgebraException.class, ()->A.flatten(5));
    }
}
