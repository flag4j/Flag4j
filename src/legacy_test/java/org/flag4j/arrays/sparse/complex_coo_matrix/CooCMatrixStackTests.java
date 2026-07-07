package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixStackTests {

    @Test
    void complexSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        Complex128[] bEntries;
        CooCMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new Complex128[]{new Complex128("0.25394+0.43087i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new Complex128[]{new Complex128("0.64208+0.02231i"), new Complex128("0.88585+0.80165i")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 3);
        expEntries = new Complex128[]{new Complex128("0.25394+0.43087i"), new Complex128("0.64208+0.02231i"), new Complex128("0.88585+0.80165i")};
        expRowIndices = new int[]{0, 2, 4};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new Complex128[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new Complex128[]{new Complex128("0.49415+0.73554i"), new Complex128("0.50848+0.69047i")};
        bRowIndices = new int[]{3, 3};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 2);
        expEntries = new Complex128[]{new Complex128("0.49415+0.73554i"), new Complex128("0.50848+0.69047i")};
        expRowIndices = new int[]{4, 4};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new Complex128[]{new Complex128("0.28989+0.09294i"), new Complex128("0.14286+0.13982i"), new Complex128("0.08978+0.69954i"), new Complex128("0.25905+0.81205i"), new Complex128("0.74641+0.0583i"), new Complex128("0.59938+0.46717i"), new Complex128("0.26599+0.76388i"), new Complex128("0.27019+0.47096i"), new Complex128("0.91115+0.20502i"), new Complex128("0.01355+0.83376i"), new Complex128("0.26981+0.89026i"), new Complex128("0.99982+0.06107i"), new Complex128("0.89229+0.62325i"), new Complex128("0.44728+0.68547i")};
        aRowIndices = new int[]{0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 7, 10, 10, 13};
        aColIndices = new int[]{2, 0, 1, 0, 2, 0, 1, 0, 0, 4, 4, 2, 3, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new Complex128[]{new Complex128("0.6588+0.75876i"), new Complex128("0.60697+0.64487i"), new Complex128("0.76965+0.38201i"), new Complex128("0.98363+0.84591i"), new Complex128("0.77316+0.25268i"), new Complex128("0.83365+0.37816i"), new Complex128("0.07067+0.88382i"), new Complex128("0.6195+0.37165i"), new Complex128("0.86337+0.12018i"), new Complex128("0.77066+0.0812i"), new Complex128("0.28691+0.60752i"), new Complex128("0.99723+0.46211i"), new Complex128("0.44258+0.43083i"), new Complex128("0.93875+0.27681i"), new Complex128("0.64751+0.18722i"), new Complex128("0.04095+0.24351i"), new Complex128("0.23496+0.90917i")};
        bRowIndices = new int[]{0, 0, 1, 4, 5, 5, 6, 7, 7, 8, 8, 8, 10, 10, 11, 12, 13};
        bColIndices = new int[]{0, 5, 0, 2, 0, 3, 4, 1, 3, 1, 4, 5, 0, 5, 1, 0, 2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix finalA = a;
        CooCMatrix finalB = b;
        assertThrows(Exception.class, ()->finalA.stack(finalB));
    }
}
