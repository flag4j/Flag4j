package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.sparse.coo.real.RealCooMatrixGetSet;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixSetSliceTests {

    @Test
    void setSliceTest() {
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
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.26601, 0.19614, 0.44447};
        aRowIndices = new int[]{0, 0, 3};
        aColIndices = new int[]{0, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.04135, 0.54584};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{2, 0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.26601, 0.19614, 0.04135, 0.54584};
        expRowIndices = new int[]{0, 0, 2, 3};
        expColIndices = new int[]{0, 2, 2, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 2, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.8099, 0.98458, 0.86365, 0.22484, 0.12245};
        aRowIndices = new int[]{4, 7, 13, 21, 22};
        aColIndices = new int[]{8, 2, 10, 8, 6};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 9);
        bEntries = new double[]{0.32305, 0.32692, 0.79176, 0.19365, 0.52579, 0.74421, 0.28361, 0.49306, 0.97286, 0.93381, 0.32774, 0.92891, 0.72314, 0.30798, 0.60839, 0.60584};
        bRowIndices = new int[]{0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4};
        bColIndices = new int[]{3, 7, 0, 1, 5, 6, 3, 6, 7, 8, 1, 3, 6, 7, 0, 4};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.8099, 0.98458, 0.86365, 0.32305, 0.32692, 0.79176, 0.19365, 0.52579, 0.74421, 0.28361, 0.49306, 0.97286, 0.93381, 0.32774, 0.92891, 0.72314, 0.30798, 0.60839, 0.60584};
        expRowIndices = new int[]{4, 7, 13, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22};
        expColIndices = new int[]{8, 2, 10, 4, 8, 1, 2, 6, 7, 4, 7, 8, 9, 2, 4, 7, 8, 1, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 18, 1));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.66147, 0.39264, 0.46927, 0.54339, 0.8033, 0.4519, 0.39398, 0.46898, 0.78049};
        aRowIndices = new int[]{41, 45, 141, 341, 376, 480, 522, 644, 828};
        aColIndices = new int[]{2, 1, 4, 2, 2, 2, 4, 1, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new double[]{0.11246, 0.6844};
        bRowIndices = new int[]{1, 2};
        bColIndices = new int[]{1, 0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.11246, 0.6844, 0.66147, 0.39264, 0.46927, 0.54339, 0.8033, 0.4519, 0.39398, 0.46898, 0.78049};
        expRowIndices = new int[]{1, 2, 41, 45, 141, 341, 376, 480, 522, 644, 828};
        expColIndices = new int[]{1, 0, 2, 1, 4, 2, 2, 2, 4, 1, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 0, 0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.00945, 0.0801, 0.88807, 0.02644};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 0, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.05356};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final0a = a;
        CooMatrix final0b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final0a, final0b, -1, 2));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.59692, 0.47247, 0.26357, 0.74655};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{0, 4, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 1);
        bEntries = new double[]{0.20907};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final1a = a;
        CooMatrix final1b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final1a, final1b, 0, 16));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.15919, 0.18619, 0.9727, 0.67348};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 3, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 2);
        bEntries = new double[]{0.81163};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final2a = a;
        CooMatrix final2b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final2a, final2b, 2, 0));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.11608, 0.45899, 0.18176, 0.52675};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.88384};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final3a = a;
        CooMatrix final3b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final3a, final3b, 0, 4));
    }


    @Test
    void setSliceDenseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[][] bEntries;
        Matrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.27021, 0.29417, 0.06904};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.88098, 0.32602, 0.83928},
                {0.08236, 0.83795, 0.84279}};
        b = new Matrix(bEntries);

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.27021, 0.88098, 0.32602, 0.83928, 0.08236, 0.83795, 0.84279};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{0, 0, 1, 2, 0, 1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 2, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.64914, 0.66932, 0.40628, 0.37954, 0.08519};
        aRowIndices = new int[]{0, 5, 6, 16, 22};
        aColIndices = new int[]{2, 3, 3, 9, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.58805, 0.25559, 0.41371, 0.98028, 0.63021, 0.31739, 0.53601, 0.31622, 0.79944},
                {0.47782, 0.07173, 0.48916, 0.19796, 0.33267, 0.11585, 0.55903, 0.85354, 0.76878},
                {0.37356, 0.92958, 0.878, 0.50643, 0.05278, 0.85421, 0.29942, 0.52806, 0.28666},
                {0.63041, 0.87807, 0.18841, 0.78023, 0.9306, 0.81551, 0.04105, 0.0534, 0.23816},
                {0.76035, 0.43175, 0.25131, 0.8096, 0.84916, 0.16624, 0.28679, 0.13698, 0.12409}};
        b = new Matrix(bEntries);

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.64914, 0.66932, 0.40628, 0.37954, 0.58805, 0.25559, 0.41371, 0.98028, 0.63021, 0.31739, 0.53601, 0.31622, 0.79944, 0.47782, 0.07173, 0.48916, 0.19796, 0.33267, 0.11585, 0.55903, 0.85354, 0.76878, 0.37356, 0.92958, 0.878, 0.50643, 0.05278, 0.85421, 0.29942, 0.52806, 0.28666, 0.63041, 0.87807, 0.18841, 0.78023, 0.9306, 0.81551, 0.04105, 0.0534, 0.23816, 0.76035, 0.43175, 0.25131, 0.8096, 0.84916, 0.16624, 0.28679, 0.13698, 0.12409};
        expRowIndices = new int[]{0, 5, 6, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{2, 3, 3, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 18, 1));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.09033, 0.75299, 0.78946, 0.91141, 0.66149, 0.23721, 0.78215, 0.14658, 0.48493};
        aRowIndices = new int[]{31, 88, 174, 224, 258, 291, 562, 595, 854};
        aColIndices = new int[]{4, 1, 3, 3, 0, 4, 4, 3, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.92281, 0.27413},
                {0.0728, 0.35943},
                {0.65062, 0.823}};
        b = new Matrix(bEntries);

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.92281, 0.27413, 0.0728, 0.35943, 0.65062, 0.823, 0.09033, 0.75299, 0.78946, 0.91141, 0.66149, 0.23721, 0.78215, 0.14658, 0.48493};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 31, 88, 174, 224, 258, 291, 562, 595, 854};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 4, 1, 3, 3, 0, 4, 4, 3, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 0, 0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.62672, 0.72454, 0.03301, 0.05962};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{4, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.88379, 0.23006, 0.79116}};
        b = new Matrix(bEntries);

        CooMatrix final0a = a;
        Matrix final0b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final0a, final0b, -1, 2));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.36772, 0.68086, 0.78025, 0.7059};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.37548},
                {0.90032},
                {0.54146}};
        b = new Matrix(bEntries);

        CooMatrix final1a = a;
        Matrix final1b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final1a, final1b, 0, 16));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.08019, 0.07101, 0.62705, 0.60587};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.77061, 0.2345},
                {0.34159, 0.25256}};
        b = new Matrix(bEntries);

        CooMatrix final2a = a;
        Matrix final2b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final2a, final2b, 2, 0));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.63961, 0.47008, 0.91095, 0.36202};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{3, 2, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.17845, 0.95046, 0.59089}};
        b = new Matrix(bEntries);

        CooMatrix final3a = a;
        Matrix final3b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final3a, final3b, 0, 4));
    }


    @Test
    void setSliceDenseArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.27021, 0.29417, 0.06904};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.88098, 0.32602, 0.83928},
                {0.08236, 0.83795, 0.84279}};

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.27021, 0.88098, 0.32602, 0.83928, 0.08236, 0.83795, 0.84279};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{0, 0, 1, 2, 0, 1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 2, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.64914, 0.66932, 0.40628, 0.37954, 0.08519};
        aRowIndices = new int[]{0, 5, 6, 16, 22};
        aColIndices = new int[]{2, 3, 3, 9, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.58805, 0.25559, 0.41371, 0.98028, 0.63021, 0.31739, 0.53601, 0.31622, 0.79944},
                {0.47782, 0.07173, 0.48916, 0.19796, 0.33267, 0.11585, 0.55903, 0.85354, 0.76878},
                {0.37356, 0.92958, 0.878, 0.50643, 0.05278, 0.85421, 0.29942, 0.52806, 0.28666},
                {0.63041, 0.87807, 0.18841, 0.78023, 0.9306, 0.81551, 0.04105, 0.0534, 0.23816},
                {0.76035, 0.43175, 0.25131, 0.8096, 0.84916, 0.16624, 0.28679, 0.13698, 0.12409}};

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.64914, 0.66932, 0.40628, 0.37954, 0.58805, 0.25559, 0.41371, 0.98028, 0.63021, 0.31739, 0.53601, 0.31622, 0.79944, 0.47782, 0.07173, 0.48916, 0.19796, 0.33267, 0.11585, 0.55903, 0.85354, 0.76878, 0.37356, 0.92958, 0.878, 0.50643, 0.05278, 0.85421, 0.29942, 0.52806, 0.28666, 0.63041, 0.87807, 0.18841, 0.78023, 0.9306, 0.81551, 0.04105, 0.0534, 0.23816, 0.76035, 0.43175, 0.25131, 0.8096, 0.84916, 0.16624, 0.28679, 0.13698, 0.12409};
        expRowIndices = new int[]{0, 5, 6, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{2, 3, 3, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 18, 1));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.09033, 0.75299, 0.78946, 0.91141, 0.66149, 0.23721, 0.78215, 0.14658, 0.48493};
        aRowIndices = new int[]{31, 88, 174, 224, 258, 291, 562, 595, 854};
        aColIndices = new int[]{4, 1, 3, 3, 0, 4, 4, 3, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.92281, 0.27413},
                {0.0728, 0.35943},
                {0.65062, 0.823}};

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.92281, 0.27413, 0.0728, 0.35943, 0.65062, 0.823, 0.09033, 0.75299, 0.78946, 0.91141, 0.66149, 0.23721, 0.78215, 0.14658, 0.48493};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 31, 88, 174, 224, 258, 291, 562, 595, 854};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 4, 1, 3, 3, 0, 4, 4, 3, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 0, 0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.62672, 0.72454, 0.03301, 0.05962};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{4, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.88379, 0.23006, 0.79116}};

        CooMatrix final0a = a;
        double[][] final0b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final0a, final0b, -1, 2));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.36772, 0.68086, 0.78025, 0.7059};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.37548},
                {0.90032},
                {0.54146}};

        CooMatrix final1a = a;
        double[][] final1b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final1a, final1b, 0, 16));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.08019, 0.07101, 0.62705, 0.60587};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.77061, 0.2345},
                {0.34159, 0.25256}};

        CooMatrix final2a = a;
        double[][] final2b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final2a, final2b, 2, 0));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.63961, 0.47008, 0.91095, 0.36202};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{3, 2, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.17845, 0.95046, 0.59089}};

        CooMatrix final3a = a;
        double[][] final3b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final3a, final3b, 0, 4));
    }


    @Test
    void setSliceDenseIntArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        int[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.27021, 0.29417, 0.06904};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {88098, 32602, 83928},
                {8236, 83795, 84279}};

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.27021, 88098, 32602, 83928, 8236, 83795, 84279};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{0, 0, 1, 2, 0, 1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 2, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.64914, 0.66932, 0.40628, 0.37954, 0.08519};
        aRowIndices = new int[]{0, 5, 6, 16, 22};
        aColIndices = new int[]{2, 3, 3, 9, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {58805, 25559, 41371, 98028, 63021, 31739, 53601, 31622, 79944},
                {47782, 7173, 48916, 19796, 33267, 11585, 55903, 85354, 76878},
                {37356, 92958, 878, 50643, 5278, 85421, 29942, 52806, 28666},
                {63041, 87807, 18841, 78023, 9306, 81551, 4105, 534, 23816},
                {76035, 43175, 25131, 8096, 84916, 16624, 28679, 13698, 12409}};

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.64914, 0.66932, 0.40628, 0.37954, 58805, 25559, 41371, 98028, 63021, 31739, 53601, 31622, 79944, 47782, 7173, 48916, 19796, 33267, 11585, 55903, 85354, 76878, 37356, 92958, 878, 50643, 5278, 85421, 29942, 52806, 28666, 63041, 87807, 18841, 78023, 9306, 81551, 4105, 534, 23816, 76035, 43175, 25131, 8096, 84916, 16624, 28679, 13698, 12409};
        expRowIndices = new int[]{0, 5, 6, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{2, 3, 3, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 18, 1));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.09033, 0.75299, 0.78946, 0.91141, 0.66149, 0.23721, 0.78215, 0.14658, 0.48493};
        aRowIndices = new int[]{31, 88, 174, 224, 258, 291, 562, 595, 854};
        aColIndices = new int[]{4, 1, 3, 3, 0, 4, 4, 3, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {92281, 27413},
                {728, 35943},
                {65062, 823}};

        expShape = new Shape(1000, 5);
        expEntries = new double[]{92281, 27413, 728, 35943, 65062, 823, 0.09033, 0.75299, 0.78946, 0.91141, 0.66149, 0.23721, 0.78215, 0.14658, 0.48493};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 31, 88, 174, 224, 258, 291, 562, 595, 854};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 4, 1, 3, 3, 0, 4, 4, 3, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealCooMatrixGetSet.setSlice(a, b, 0, 0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.62672, 0.72454, 0.03301, 0.05962};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{4, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {88379, 23006, 79116}};

        CooMatrix final0a = a;
        int[][] final0b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final0a, final0b, -1, 2));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.36772, 0.68086, 0.78025, 0.7059};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {37548},
                {90032},
                {54146}};

        CooMatrix final1a = a;
        int[][] final1b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final1a, final1b, 0, 16));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.08019, 0.07101, 0.62705, 0.60587};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {77061, 2345},
                {34159, 25256}};

        CooMatrix final2a = a;
        int[][] final2b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final2a, final2b, 2, 0));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.63961, 0.47008, 0.91095, 0.36202};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{3, 2, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {17845, 95046, 59089}};

        CooMatrix final3a = a;
        int[][] final3b = b;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.setSlice(final3a, final3b, 0, 4));
    }
}
