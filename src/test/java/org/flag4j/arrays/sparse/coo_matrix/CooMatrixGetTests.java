package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixGetTests {


    @Test
    void getTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.86543, 0.89474, 0.79431};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{4, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = 0.89474;

        assertEquals(exp, a.get(1, 2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.39077, 0.19647, 0.80091, 0.42942, 0.28875};
        aRowIndices = new int[]{2, 8, 9, 9, 9};
        aColIndices = new int[]{0, 3, 3, 9, 12};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = 0.0;

        assertEquals(exp, a.get(0, 0));

        // ---------------------  sub-case 2.1 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.39077, 0.19647, 0.80091, 0.42942, 0.28875};
        aRowIndices = new int[]{2, 8, 9, 9, 9};
        aColIndices = new int[]{0, 3, 3, 9, 12};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = 0.19647;

        assertEquals(exp, a.get(8, 3));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.75552, 0.97989, 0.41491, 0.29057};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.get(3, 3));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.83077, 0.21434, 0.72733, 0.09799};
        aRowIndices = new int[]{0, 1, 4, 4};
        aColIndices = new int[]{0, 2, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.get(5, 1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.11064, 0.49239, 0.89016, 0.63649};
        aRowIndices = new int[]{0, 2, 4, 4};
        aColIndices = new int[]{2, 2, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.get(1, 5));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.6489, 0.49003, 0.11063, 0.70878};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{2, 1, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.get(-1, 2));
    }


    @Test
    void getDiagTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape expShape;
        int[] expIndices;
        double[] expEntries;
        CooVector exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.11978, 0.28555, 0.42594};
        aRowIndices = new int[]{2, 2, 2};
        aColIndices = new int[]{1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{0.28555};
        expIndices = new int[]{2};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getDiag());

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.0805, 0.73443, 0.44368, 0.26576, 0.05534, 0.99412, 0.29771, 0.99234, 0.05817, 0.17558, 0.75404, 0.41199, 0.62219, 0.70961, 0.19033};
        aRowIndices = new int[]{0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 10};
        aColIndices = new int[]{5, 1, 5, 6, 8, 12, 22, 5, 0, 2, 10, 8, 19, 15, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11);
        expEntries = new double[]{0.73443};
        expIndices = new int[]{1};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getDiag());

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.73896, 0.94136, 0.30706, 0.264};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{0, 1, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{0.73896, 0.94136};
        expIndices = new int[]{0, 1};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getDiag());

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.88859, 0.80585, 0.65323, 0.30953};
        aRowIndices = new int[]{1, 2, 2, 4};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getDiag());

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.19569, 0.46268, 0.90696, 0.04954};
        aRowIndices = new int[]{0, 0, 2, 3};
        aColIndices = new int[]{0, 1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{0.19569};
        expIndices = new int[]{0};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getDiag());

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.71884, 0.86003, 0.59532, 0.06677};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{2, 1, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{0.86003, 0.06677};
        expIndices = new int[]{1, 2};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getDiag());
    }
}
