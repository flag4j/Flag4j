package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.sparse.coo.real.RealCooMatrixGetSet;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixSetRowColTests {

    @Test
    void setColTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[] bEntries;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.42216, 0.86886, 0.51801};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.30728, 0.13698, 0.23211, 0.05517, 0.12575};

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.30728, 0.42216, 0.86886, 0.13698, 0.51801, 0.23211, 0.05517, 0.12575};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 2, 3, 4};
        expColIndices = new int[]{0, 1, 2, 0, 2, 0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setCol(a, 0, bEntries));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.86291, 0.59273, 0.14697, 0.79343, 0.0691};
        aRowIndices = new int[]{4, 5, 6, 8, 10};
        aColIndices = new int[]{14, 3, 9, 15, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.09599, 0.03342, 0.08342, 0.86195, 0.18126, 0.71121, 0.03191, 0.3479, 0.5699, 0.35584, 0.51796};

        expShape = new Shape(11, 23);
        expEntries = new double[]{0.09599, 0.03342, 0.08342, 0.86195, 0.86291, 0.18126, 0.59273, 0.71121, 0.14697, 0.03191, 0.3479, 0.79343, 0.5699, 0.35584, 0.0691, 0.51796};
        expRowIndices = new int[]{0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10};
        expColIndices = new int[]{16, 16, 16, 16, 14, 16, 3, 16, 9, 16, 16, 15, 16, 16, 4, 16};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setCol(a, 16, bEntries));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.91557, 0.99112, 0.97331, 0.46736, 0.39273, 0.9236, 0.55027, 0.96506, 0.46553};
        aRowIndices = new int[]{0, 1, 2, 3, 3, 4, 4, 4, 4};
        aColIndices = new int[]{118, 335, 419, 424, 880, 134, 358, 492, 949};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.86214, 0.01468, 0.80744, 0.38058, 0.27367};

        expShape = new Shape(5, 1000);
        expEntries = new double[]{0.91557, 0.86214, 0.99112, 0.01468, 0.97331, 0.80744, 0.46736, 0.39273, 0.38058, 0.9236, 0.55027, 0.96506, 0.46553, 0.27367};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4};
        expColIndices = new int[]{118, 999, 335, 999, 419, 999, 424, 880, 999, 134, 358, 492, 949, 999};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setCol(a, 999, bEntries));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.20695, 0.08553, 0.58839, 0.42649};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 4, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.70299, 0.12535, 0.51468};

        CooMatrix final0a = a;
        double[] final0b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setCol(final0a, 6, final0b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.8715, 0.48536, 0.74835, 0.61107};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{2, 4, 2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.18264, 0.50269, 0.62068, 0.68308, 0.25792};

        CooMatrix final1a = a;
        double[] final1b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setCol(final1a, 3, final1b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.97644, 0.04564, 0.1204, 0.19723};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{4, 1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.69692, 0.15703};

        CooMatrix final2a = a;
        double[] final2b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setCol(final2a, 3, final2b));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.9503, 0.0484, 0.44488, 0.29844};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{0, 1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.36708, 0.70117, 0.73955};

        CooMatrix final3a = a;
        double[] final3b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setCol(final3a, 19, final3b));
    }


    @Test
    void setColSparseVectorTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bIndices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.69683, 0.7974, 0.01005};
        aRowIndices = new int[]{0, 3, 4};
        aColIndices = new int[]{1, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.42925, 0.95116};
        bIndices = new int[]{2, 3};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.69683, 0.42925, 0.95116, 0.01005};
        expRowIndices = new int[]{0, 2, 3, 4};
        expColIndices = new int[]{1, 0, 0, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.09879, 0.44944, 0.39054, 0.51234, 0.10826};
        aRowIndices = new int[]{1, 3, 7, 8, 10};
        aColIndices = new int[]{14, 15, 16, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11);
        bEntries = new double[]{0.42701, 0.22431, 0.48719, 0.79679};
        bIndices = new int[]{5, 6, 7, 10};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(11, 23);
        expEntries = new double[]{0.09879, 0.44944, 0.42701, 0.22431, 0.48719, 0.51234, 0.10826, 0.79679};
        expRowIndices = new int[]{1, 3, 5, 6, 7, 8, 10, 10};
        expColIndices = new int[]{14, 15, 16, 16, 16, 3, 4, 16};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 16));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.548, 0.12782, 0.71044, 0.03123, 0.73197, 0.23329, 0.76449, 0.62306, 0.77283};
        aRowIndices = new int[]{0, 1, 1, 1, 2, 2, 2, 3, 4};
        aColIndices = new int[]{663, 597, 620, 926, 73, 153, 627, 66, 743};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.92473, 0.36888};
        bIndices = new int[]{1, 4};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(5, 1000);
        expEntries = new double[]{0.548, 0.12782, 0.71044, 0.03123, 0.92473, 0.73197, 0.23329, 0.76449, 0.62306, 0.77283, 0.36888};
        expRowIndices = new int[]{0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4};
        expColIndices = new int[]{663, 597, 620, 926, 999, 73, 153, 627, 66, 743, 999};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.38374, 0.24165, 0.20689, 0.73343};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{4, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.93917};
        bIndices = new int[]{2};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final0a = a;
        CooVector final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, 6));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.52077, 0.42897, 0.35701, 0.94909};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{2, 2, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.41526, 0.41046};
        bIndices = new int[]{0, 2};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final1a = a;
        CooVector final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 3));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.89024, 0.42578, 0.66571, 0.53301};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{2, 0, 1, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.55374};
        bIndices = new int[]{1};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final2a = a;
        CooVector final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 3));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.74812, 0.07704, 0.80715, 0.45783};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.838};
        bIndices = new int[]{0};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final3a = a;
        CooVector final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 19));
    }


    @Test
    void setRowCooVectorTests() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bIndices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.69683, 0.7974, 0.01005};
        aRowIndices = new int[]{1, 0, 2};
        aColIndices = new int[]{0, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(5);
        bEntries = new double[]{0.42925, 0.95116};
        bIndices = new int[]{2, 3};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.69683, 0.42925, 0.95116, 0.01005};
        expRowIndices = new int[]{1, 0, 0, 2};
        expColIndices = new int[]{0, 2, 3, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).sortIndices();

        assertEquals(exp, a.setRow(b, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.09879, 0.44944, 0.39054, 0.51234, 0.10826};
        aColIndices = new int[]{1, 3, 7, 8, 10};
        aRowIndices = new int[]{14, 15, 16, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(11);
        bEntries = new double[]{0.42701, 0.22431, 0.48719, 0.79679};
        bIndices = new int[]{5, 6, 7, 10};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.09879, 0.44944, 0.42701, 0.22431, 0.48719, 0.51234, 0.10826, 0.79679};
        expColIndices = new int[]{1, 3, 5, 6, 7, 8, 10, 10};
        expRowIndices = new int[]{14, 15, 16, 16, 16, 3, 4, 16};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).sortIndices();

        assertEquals(exp, a.setRow(b, 16));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.548, 0.12782, 0.71044, 0.03123, 0.73197, 0.23329, 0.76449, 0.62306, 0.77283};
        aColIndices = new int[]{0, 1, 1, 1, 2, 2, 2, 3, 4};
        aRowIndices = new int[]{663, 597, 620, 926, 73, 153, 627, 66, 743};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(5);
        bEntries = new double[]{0.92473, 0.36888};
        bIndices = new int[]{1, 4};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.548, 0.12782, 0.71044, 0.03123, 0.92473, 0.73197, 0.23329, 0.76449, 0.62306, 0.77283, 0.36888};
        expColIndices = new int[]{0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4};
        expRowIndices = new int[]{663, 597, 620, 926, 999, 73, 153, 627, 66, 743, 999};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).sortIndices();

        assertEquals(exp, a.setRow(b, 999));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.38374, 0.24165, 0.20689, 0.73343};
        aColIndices = new int[]{0, 1, 2, 2};
        aRowIndices = new int[]{4, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(3);
        bEntries = new double[]{0.93917};
        bIndices = new int[]{2};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final0a = a;
        CooVector final0b = b;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 6));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.52077, 0.42897, 0.35701, 0.94909};
        aColIndices = new int[]{0, 1, 2, 2};
        aRowIndices = new int[]{2, 2, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(5);
        bEntries = new double[]{0.41526, 0.41046};
        bIndices = new int[]{0, 2};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final1a = a;
        CooVector final1b = b;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 3));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.89024, 0.42578, 0.66571, 0.53301};
        aColIndices = new int[]{0, 1, 1, 2};
        aRowIndices = new int[]{2, 0, 1, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(2);
        bEntries = new double[]{0.55374};
        bIndices = new int[]{1};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final2a = a;
        CooVector final2b = b;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, 3));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.74812, 0.07704, 0.80715, 0.45783};
        aColIndices = new int[]{0, 1, 2, 2};
        aRowIndices = new int[]{0, 1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bShape = new Shape(3);
        bEntries = new double[]{0.838};
        bIndices = new int[]{0};
        b = new CooVector(bShape.getSize(0), bEntries, bIndices);

        CooMatrix final3a = a;
        CooVector final3b = b;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 19));
    }

    @Test
    void setRowDenseTests() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[] bEntries;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.42216, 0.86886, 0.51801};
        aColIndices = new int[]{0, 0, 1};
        aRowIndices = new int[]{1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.30728, 0.13698, 0.23211, 0.05517, 0.12575};

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.30728, 0.42216, 0.86886, 0.13698, 0.51801, 0.23211, 0.05517, 0.12575};
        expColIndices = new int[]{0, 0, 0, 1, 1, 2, 3, 4};
        expRowIndices = new int[]{0, 1, 2, 0, 2, 0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).sortIndices();

        assertEquals(exp, RealCooMatrixGetSet.setRow(a, 0, bEntries));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.86291, 0.59273, 0.14697, 0.79343, 0.0691};
        aColIndices = new int[]{4, 5, 6, 8, 10};
        aRowIndices = new int[]{14, 3, 9, 15, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.09599, 0.03342, 0.08342, 0.86195, 0.18126, 0.71121, 0.03191, 0.3479, 0.5699, 0.35584, 0.51796};

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.09599, 0.03342, 0.08342, 0.86195, 0.86291, 0.18126, 0.59273, 0.71121, 0.14697, 0.03191, 0.3479, 0.79343, 0.5699, 0.35584, 0.0691, 0.51796};
        expColIndices = new int[]{0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10};
        expRowIndices = new int[]{16, 16, 16, 16, 14, 16, 3, 16, 9, 16, 16, 15, 16, 16, 4, 16};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).sortIndices();

        assertEquals(exp, RealCooMatrixGetSet.setRow(a, 16, bEntries));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.91557, 0.99112, 0.97331, 0.46736, 0.39273, 0.9236, 0.55027, 0.96506, 0.46553};
        aColIndices = new int[]{0, 1, 2, 3, 3, 4, 4, 4, 4};
        aRowIndices = new int[]{118, 335, 419, 424, 880, 134, 358, 492, 949};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.86214, 0.01468, 0.80744, 0.38058, 0.27367};

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.91557, 0.86214, 0.99112, 0.01468, 0.97331, 0.80744, 0.46736, 0.39273, 0.38058, 0.9236, 0.55027, 0.96506, 0.46553, 0.27367};
        expColIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4};
        expRowIndices = new int[]{118, 999, 335, 999, 419, 999, 424, 880, 999, 134, 358, 492, 949, 999};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices).sortIndices();

        assertEquals(exp, RealCooMatrixGetSet.setRow(a, 999, bEntries));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.20695, 0.08553, 0.58839, 0.42649};
        aColIndices = new int[]{0, 0, 1, 2};
        aRowIndices = new int[]{1, 2, 4, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.70299, 0.12535, 0.51468};

        CooMatrix final0a = a;
        double[] final0b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setRow(final0a, 6, final0b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.8715, 0.48536, 0.74835, 0.61107};
        aColIndices = new int[]{1, 1, 2, 2};
        aRowIndices = new int[]{2, 4, 2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.18264, 0.50269, 0.62068, 0.68308, 0.25792};

        CooMatrix final1a = a;
        double[] final1b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setRow(final1a, 3, final1b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.97644, 0.04564, 0.1204, 0.19723};
        aColIndices = new int[]{0, 2, 2, 2};
        aRowIndices = new int[]{4, 1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.69692, 0.15703};

        CooMatrix final2a = a;
        double[] final2b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setRow(final2a, 3, final2b));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.9503, 0.0484, 0.44488, 0.29844};
        aColIndices = new int[]{0, 2, 2, 2};
        aRowIndices = new int[]{0, 1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{0.36708, 0.70117, 0.73955};

        CooMatrix final3a = a;
        double[] final3b = bEntries;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setRow(final3a, 19, final3b));

        // ---------------------  sub-case 8 ---------------------
        aShape = new Shape(21, 32);
        aEntries = new double[]{0.9503, 0.0484, 0.44488, 0.29844, -1.515, 20234.123};
        aRowIndices = new int[]{3, 15, 15, 15, 17, 20};
        aColIndices = new int[]{1, 5,  8,  13, 3,  0 };
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices).sortIndices();

        bEntries = new double[]{-9.865780643419962, 4.755111047514857, -6.443769475611005, -6.389082329867075, -7.77229497365825,
                2.9256765346746505, -0.4403770113908312, -9.930275071364584, -1.301116729717764, 4.112954038109342,
                5.840693384369942, -0.9163855794658371, 7.877142592316218, -4.962843547135085, -9.342168182539481,
                -7.875571161798163, -2.5778080197216173, -5.910510433509996, -5.982349806812259, 7.9221497570046004,
                -6.637400014097297, -7.2100059336746, 6.580249567644238, -1.2577616266417806, 2.9633941118990776,
                1.68922841488396, 1.3879844182529482, 1.304876540209957, -1.7553804221712515, 1.075723178061402,
                -7.01621785650135, 0.8428004605597845};

        expShape = new Shape(21, 32);
        expEntries = new double[]{0.9503, -9.865780643419962, 4.755111047514857, -6.443769475611005, -6.389082329867075,
                -7.77229497365825, 2.9256765346746505, -0.4403770113908312, -9.930275071364584, -1.301116729717764, 4.112954038109342,
                5.840693384369942, -0.9163855794658371, 7.877142592316218, -4.962843547135085, -9.342168182539481,
                -7.875571161798163, -2.5778080197216173, -5.910510433509996, -5.982349806812259, 7.9221497570046004,
                -6.637400014097297, -7.2100059336746, 6.580249567644238, -1.2577616266417806, 2.9633941118990776,
                1.68922841488396, 1.3879844182529482, 1.304876540209957, -1.7553804221712515, 1.075723178061402,
                -7.01621785650135, 0.8428004605597845, -1.515, 20234.123};
        expRowIndices = new int[]{3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 17, 20};
        expColIndices = new int[]{1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 3, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 15));
    }
}
