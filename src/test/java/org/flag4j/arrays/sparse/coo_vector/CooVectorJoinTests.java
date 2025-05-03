package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorJoinTests {

    static double[] aEntries;
    static int[] aIndices, bIndices, expIndices;
    static int sparseSize, bSize, expSize;
    static CooVector a;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.34, -8781.5, 145.4};
        aIndices = new int[]{0, 1, 6};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseRealJoinTestCase() {
        double[] bEntries, expEntries;
        CooVector b, exp;

        // ------------------- sub-case 1 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5};
        bIndices = new int[]{0, 3, 4};
        bSize = 5;
        b = new CooVector(bSize, bEntries, bIndices);
        expEntries = new double[]{1.34, -8781.5, 145.4, 24.53, 66.1, -234.5};
        expIndices = new int[]{0, 1, 6, 8, 11, 12};
        expSize = 13;
        exp = new CooVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.join(b));
    }


    @Test
    void sparseRealStackTestCase() {
        double[] bEntries, expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CooVector b;
        CooMatrix exp;

        // ------------------- sub-case 1 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 1.3};
        bIndices = new int[]{0, 5, 6, 7};
        bSize = 8;
        b = new CooVector(bSize, bEntries, bIndices);
        expEntries = new double[]{1.34, -8781.5, 145.4, 24.53, 66.1, -234.5, 1.3};
        shape = new Shape(2, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 1};
        colIndices = new int[]{0, 1, 6, 0, 5, 6, 7};
        exp = new CooMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));
        assertEquals(exp.T(), a.stack(b, 1));

        // ------------------- sub-case 2 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 1.3};
        bIndices = new int[]{0, 5, 6, 7};
        bSize = 25;
        b = new CooVector(bSize, bEntries, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 3));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, -2));
    }


    @Test
    void extendTestCase() {
        double[] expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CooMatrix exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new double[]{1.34, -8781.5, 145.4,
                1.34, -8781.5, 145.4,
                1.34, -8781.5, 145.4,
                1.34, -8781.5, 145.4};
        shape = new Shape(4, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
        colIndices = new int[]{0, 1, 6, 0, 1, 6, 0, 1, 6, 0, 1, 6};
        exp = new CooMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.repeat(4, 0));
        assertEquals(exp.T(), a.repeat(4, 1));

        // ------------------- sub-case 2 -------------------
        assertThrows(IllegalArgumentException.class, ()->a.repeat(4, -1));
        assertThrows(IllegalArgumentException.class, ()->a.repeat(4, 235));
        assertThrows(IllegalArgumentException.class, ()->a.repeat(-1, 0));
    }
}
