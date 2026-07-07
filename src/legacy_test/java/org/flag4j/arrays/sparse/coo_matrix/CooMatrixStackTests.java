package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixStackTests {

    @Test
    void realSparseStackTest() {
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
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.6994};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.53334, 0.36866};
        bRowIndices = new int[]{1, 3};
        bColIndices = new int[]{1, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 3);
        expEntries = new double[]{0.6994, 0.53334, 0.36866};
        expRowIndices = new int[]{0, 3, 5};
        expColIndices = new int[]{2, 1, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.54824, 0.3091};
        bRowIndices = new int[]{2, 4};
        bColIndices = new int[]{0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 2);
        expEntries = new double[]{0.54824, 0.3091};
        expRowIndices = new int[]{3, 5};
        expColIndices = new int[]{0, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.95223, 0.17095, 0.32156, 0.9396, 0.3762, 0.10813, 0.08055, 0.31545, 0.15498, 0.91676, 0.83886, 0.94144, 0.30784, 0.49851};
        aRowIndices = new int[]{0, 0, 1, 4, 4, 4, 5, 5, 6, 6, 6, 9, 10, 11};
        aColIndices = new int[]{0, 2, 0, 2, 3, 4, 1, 3, 0, 2, 4, 3, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.05335, 0.11684, 0.88899, 0.80113, 0.27065, 0.60548, 0.61726, 0.29856, 0.96785, 0.66154, 0.07842, 0.44824, 0.06887, 0.53813, 0.72747, 0.93312, 0.04432};
        bRowIndices = new int[]{0, 0, 0, 0, 2, 2, 2, 4, 4, 6, 7, 7, 8, 9, 9, 10, 13};
        bColIndices = new int[]{1, 2, 4, 5, 1, 3, 5, 4, 5, 1, 1, 5, 2, 4, 5, 5, 2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finalA = a;
        CooMatrix finalB = b;
        assertThrows(Exception.class, ()->finalA.stack(finalB));
    }
}
