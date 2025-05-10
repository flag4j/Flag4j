package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixAugmentVectorTests {

    @Test
    void realSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bindices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.33367, 0.26667, 0.24386, 0.05929, 0.90991};
        aRowIndices = new int[]{0, 0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 4, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.74357};
        bindices = new int[]{2};
        b = new CooVector(bShape.getSize(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new double[]{0.33367, 0.26667, 0.24386, 0.05929, 0.90991, 0.74357};
        expRowIndices = new int[]{0, 0, 0, 1, 2, 2};
        expColIndices = new int[]{1, 2, 4, 0, 0, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{0.48537};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.18184};
        bindices = new int[]{0};
        b = new CooVector(bShape.getSize(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new double[]{0.48537, 0.18184};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{0, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.39748, 0.85537, 0.1253};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.10397};
        bindices = new int[]{2};
        b = new CooVector(bShape.getSize(0), bEntries, bindices);

        CooMatrix finala = a;
        CooVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }
}
