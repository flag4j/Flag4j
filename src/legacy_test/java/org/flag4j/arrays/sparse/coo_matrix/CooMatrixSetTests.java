package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixSetTests {

    @Test
    void setTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.21324, 0.76399, 0.66487, 0.29669};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{3, 4, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.21324, 0.76399, 0.66487, 445.0, 0.29669};
        expRowIndices = new int[]{0, 0, 1, 1, 2};
        expColIndices = new int[]{3, 4, 0, 2, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(445d, 1, 2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.37729, 0.0248, 0.10208, 0.5793, 0.60641, 0.37054, 0.14913, 0.84641, 0.71452, 0.16433, 0.58993, 0.93262, 0.92069, 0.24642, 0.87162, 0.71633, 0.3899, 0.93362, 0.47546, 0.11489, 0.30476, 0.87321, 0.78269, 0.35315, 0.21626};
        aRowIndices = new int[]{0, 0, 3, 4, 4, 5, 6, 8, 9, 10, 12, 13, 13, 14, 14, 17, 17, 18, 18, 18, 20, 20, 20, 22, 22};
        aColIndices = new int[]{3, 9, 3, 4, 7, 6, 5, 7, 2, 2, 0, 4, 5, 1, 7, 6, 8, 2, 3, 5, 0, 2, 3, 0, 5};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.37729, 0.0248, 0.10208, 0.5793, 0.60641, 0.37054, 0.14913, 0.84641, 0.71452, 0.16433, 0.58993, 0.93262, 0.92069, 0.24642, 0.87162, -5.2, 0.71633, 0.3899, 0.93362, 0.47546, 0.11489, 0.30476, 0.87321, 0.78269, 0.35315, 0.21626};
        expRowIndices = new int[]{0, 0, 3, 4, 4, 5, 6, 8, 9, 10, 12, 13, 13, 14, 14, 15, 17, 17, 18, 18, 18, 20, 20, 20, 22, 22};
        expColIndices = new int[]{3, 9, 3, 4, 7, 6, 5, 7, 2, 2, 0, 4, 5, 1, 7, 9, 6, 8, 2, 3, 5, 0, 2, 3, 0, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(-5.2, 15, 9));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 32156);
        aEntries = new double[]{0.46501, 0.10584, 0.99668, 0.16603, 0.44773, 0.97608, 0.97818, 0.32553, 0.29308};
        aRowIndices = new int[]{36, 39, 478, 542, 609, 646, 842, 925, 984};
        aColIndices = new int[]{5484, 9534, 31464, 9826, 29262, 3474, 18561, 12481, 8882};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1000, 32156);
        expEntries = new double[]{0.46501, 0.10584, 7.2, 0.99668, 0.16603, 0.44773, 0.97608, 0.97818, 0.32553, 0.29308};
        expRowIndices = new int[]{36, 39, 234, 478, 542, 609, 646, 842, 925, 984};
        expColIndices = new int[]{5484, 9534, 11002, 31464, 9826, 29262, 3474, 18561, 12481, 8882};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(7.2, 234, 11002));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new double[]{0.50223, 0.06803};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.set(1d, 6, 1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new double[]{0.16745, 0.27226};
        aRowIndices = new int[]{3, 3};
        aColIndices = new int[]{0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.set(1d, 1, 9));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new double[]{0.69313, 0.83671};
        aRowIndices = new int[]{3, 4};
        aColIndices = new int[]{1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.set(1d, -1, 1));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new double[]{0.45937, 0.37515};
        aRowIndices = new int[]{2, 3};
        aColIndices = new int[]{0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.set(1d, 1, -1));

        // ---------------------  sub-case 8 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.21324, 0.76399, 0.66487, 0.29669};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{3, 4, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.21324, 445, 0.66487, 0.29669};
        expRowIndices = new int[]{0, 0, 1, 2};
        expColIndices = new int[]{3, 4, 0, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(445d, 0, 4));
    }
}
