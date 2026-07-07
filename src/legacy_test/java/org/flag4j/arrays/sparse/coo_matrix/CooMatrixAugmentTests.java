package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixAugmentTests {

    @Test
    void realSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.93474, 0.28545, 0.33318};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{2, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new double[]{0.2452};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 7);
        expEntries = new double[]{0.2452, 0.93474, 0.28545, 0.33318};
        expRowIndices = new int[]{0, 1, 2, 2};
        expColIndices = new int[]{5, 2, 2, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.05933};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(2, 4);
        expEntries = new double[]{0.05933};
        expRowIndices = new int[]{0};
        expColIndices = new int[]{3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new double[]{0.74244, 0.32181};
        aRowIndices = new int[]{0, 0};
        aColIndices = new int[]{2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 4);
        bEntries = new double[]{0.38007, 0.3302};
        bRowIndices = new int[]{1, 2};
        bColIndices = new int[]{0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finalA = a;
        CooMatrix finalB = b;
        assertThrows(Exception.class, ()->finalA.augment(finalB));
    }
}
