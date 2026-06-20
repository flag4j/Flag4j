package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorReshapeTests {

    static CooVector a;
    CooVector exp;

    @BeforeAll
    static void setup() {
        double[] values = {1.34, 51.6, -0.00245};
        int[] indices = {0, 5, 103};
        int size = 304;
        a = new CooVector(size, values, indices);
    }

    @Test
    void reshapeTestCase() {
        double[] values = {1.34, 51.6, -0.00245};
        int[] indices = {0, 5, 103};
        int size = 304;

        // -------------------- sub-case 1 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.reshape(new Shape(size)));

        // -------------------- sub-case 2 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.reshape(size));

        // -------------------- sub-case 3 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.flatten());

        // -------------------- sub-case 4 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.flatten(0));

        // -------------------- sub-case 5 --------------------
        assertThrows(NDArrayShapeException.class, ()->a.reshape(new Shape(size-3)));

        // -------------------- sub-case 6 --------------------
        assertThrows(NDArrayShapeException.class, ()->a.reshape(size-3));

        // -------------------- sub-case 7 --------------------
        assertThrows(LinearAlgebraException.class, ()->a.flatten(1));
    }
}
