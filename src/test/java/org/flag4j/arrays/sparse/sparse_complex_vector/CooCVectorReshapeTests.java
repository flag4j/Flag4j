package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorReshapeTests {
    static int[] aIndices, expIndices;
    static Complex128[] aEntries, expEntries;
    static int sparseSize;
    static CooCVector a, exp;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24_023;
        a = new CooCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void reshapeTestCase() {
        // ------------------ sub-case 1 ------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new CooCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.reshape(new Shape(sparseSize)));
        assertEquals(exp, a.reshape(sparseSize));
        assertEquals(exp, a.flatten());
        assertEquals(exp, a.flatten(0));

        // ------------------ sub-case 2 ------------------
        assertThrows(ArrayShapeException.class, ()->a.reshape(new Shape(sparseSize-2)));
        assertThrows(ArrayShapeException.class, ()->a.reshape(sparseSize+32));
        assertThrows(LinearAlgebraException.class, ()->a.flatten(-35));
        assertThrows(LinearAlgebraException.class, ()->a.flatten(1));
    }
}
