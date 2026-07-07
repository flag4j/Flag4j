package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorJoinTests {

    static Complex128[] aEntries;
    static int[] aIndices, bIndices, expIndices;
    static int sparseSize, bSize, expSize;
    static CooCVector a;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{new Complex128(224.5, -93.2), new Complex128(322.5), new Complex128(46.72)};
        aIndices = new int[]{0, 1, 6};
        sparseSize = 8;
        a = new CooCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseComplexJoinTestCase() {
        Complex128[] bEntries, expEntries;
        CooCVector b, exp;

        // ------------------- sub-case 1 -------------------
        bEntries = new Complex128[]{new Complex128(24.53), new Complex128(66.1), new Complex128(-234.5)};
        bIndices = new int[]{0, 3, 4};
        bSize = 5;
        b = new CooCVector(bSize, bEntries, bIndices);
        expEntries = new Complex128[]{new Complex128(224.5, -93.2), new Complex128(322.5), new Complex128(46.72)
                , new Complex128(24.53), new Complex128(66.1), new Complex128(-234.5)};
        expIndices = new int[]{0, 1, 6, 8, 11, 12};
        expSize = 13;
        exp = new CooCVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.join(b));
    }


    @Test
    void sparseComplexStackTestCase() {
        Complex128[] bEntries, expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CooCVector b;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        bEntries = new Complex128[]{new Complex128(24.5, -0.12), new Complex128(24.5, 3.4),
                new Complex128(-0.20015), new Complex128(9825.4, -85.126)};
        bIndices = new int[]{0, 5, 6, 7};
        bSize = 8;
        b = new CooCVector(bSize, bEntries, bIndices);
        expEntries = new Complex128[]{
                new Complex128(224.5, -93.2), new Complex128(322.5), new Complex128(46.72),
                new Complex128(24.5, -0.12), new Complex128(24.5, 3.4),
                new Complex128(-0.20015), new Complex128(9825.4, -85.126)};
        shape = new Shape(2, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 1};
        colIndices = new int[]{0, 1, 6, 0, 5, 6, 7};
        exp = new CooCMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));
        assertEquals(exp.T(), a.stack(b, 1));

        // ------------------- sub-case 2 -------------------
        bEntries = new Complex128[]{new Complex128(24.5, -0.12), new Complex128(24.5, 3.4),
                new Complex128(-0.20015), new Complex128(9825.4, -85.126)};
        bIndices = new int[]{0, 5, 68, 995};
        bSize = 2325;
        b = new CooCVector(bSize, bEntries, bIndices);

        CooCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 3));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, -2));
    }
}
