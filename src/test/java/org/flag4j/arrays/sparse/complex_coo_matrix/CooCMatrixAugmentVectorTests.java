package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixAugmentVectorTests {

    @Test
    void realSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bindices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.09868+0.62735i"), new Complex128("0.67488+0.96181i"), new Complex128("0.36458+0.34928i"), new Complex128("0.00493+0.29436i"), new Complex128("0.02579+0.55809i")};
        aRowIndices = new int[]{0, 0, 1, 1, 2};
        aColIndices = new int[]{1, 2, 1, 3, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.54073};
        bindices = new int[]{1};
        b = new CooVector(bShape.getSize(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new Complex128[]{new Complex128("0.09868+0.62735i"), new Complex128("0.67488+0.96181i"), new Complex128("0.36458+0.34928i"), new Complex128("0.00493+0.29436i"), new Complex128("0.54073"), new Complex128("0.02579+0.55809i")};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 2};
        expColIndices = new int[]{1, 2, 1, 3, 5, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new Complex128[]{new Complex128("0.65883+0.21666i")};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.95187};
        bindices = new int[]{1};
        b = new CooVector(bShape.getSize(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new Complex128[]{new Complex128("0.65883+0.21666i"), new Complex128("0.95187")};
        expRowIndices = new int[]{1, 1};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new Complex128[]{new Complex128("0.10575+0.33618i"), new Complex128("0.10586+0.13023i"), new Complex128("0.75827+0.28989i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.75597};
        bindices = new int[]{2};
        b = new CooVector(bShape.getSize(0), bEntries, bindices);

        CooCMatrix finala = a;
        CooVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void complexSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bindices;
        Complex128[] bEntries;
        CooCVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.25755+0.03656i"), new Complex128("0.0597+0.3623i"), new Complex128("0.55296+0.62671i"), new Complex128("0.86631+0.03059i"), new Complex128("0.68804+0.52549i")};
        aRowIndices = new int[]{0, 0, 1, 1, 1};
        aColIndices = new int[]{1, 3, 0, 1, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new Complex128[]{new Complex128("0.57124+0.51268i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.getSize(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new Complex128[]{new Complex128("0.25755+0.03656i"), new Complex128("0.0597+0.3623i"), new Complex128("0.57124+0.51268i"), new Complex128("0.55296+0.62671i"), new Complex128("0.86631+0.03059i"), new Complex128("0.68804+0.52549i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1};
        expColIndices = new int[]{1, 3, 5, 0, 1, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new Complex128[]{new Complex128("0.26302+0.62028i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new Complex128[]{new Complex128("0.33923+0.32409i")};
        bindices = new int[]{1};
        b = new CooCVector(bShape.getSize(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new Complex128[]{new Complex128("0.26302+0.62028i"), new Complex128("0.33923+0.32409i")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new Complex128[]{new Complex128("0.5785+0.45686i"), new Complex128("0.17044+0.73657i"), new Complex128("0.48501+0.29552i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{2, 0, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new Complex128[]{new Complex128("0.35569+0.58202i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.getSize(0), bEntries, bindices);

        CooCMatrix finala = a;
        CooCVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }
}
