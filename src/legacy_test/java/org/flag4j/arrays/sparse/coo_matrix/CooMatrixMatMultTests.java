package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixMatMultTests {

    @Test
    void multSparseTest() {
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

        double[][] expEntries;
        Matrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.25182, 0.60999, 0.9924};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{1, 4, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 6);
        bEntries = new double[]{0.19019, 0.87031, 0.25254, 0.56097, 0.28014, 0.57818, 0.96506, 0.82435};
        bRowIndices = new int[]{1, 1, 2, 2, 3, 4, 4, 4};
        bColIndices = new int[]{0, 4, 4, 5, 2, 0, 2, 4};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.047893645799999995, 0.0, 0.0, 0.0, 0.2191614642, 0.0},
                {0.35268401820000006, 0.0, 0.5886769494, 0.0, 0.5028452565, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.77547, 0.71339, 0.98072, 0.24965, 0.7238, 0.16365, 0.3075, 0.74171, 0.12391, 0.15652, 0.76702, 0.85707, 0.87356, 0.21696, 0.88303};
        aRowIndices = new int[]{1, 3, 3, 3, 5, 5, 5, 5, 5, 6, 6, 7, 9, 10, 10};
        aColIndices = new int[]{9, 10, 11, 12, 5, 7, 8, 11, 22, 2, 17, 2, 3, 0, 22};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(23, 11);
        bEntries = new double[]{0.47513, 0.63059, 0.4386, 0.72764, 0.14003, 0.34452, 0.89196, 0.7611, 0.27144, 0.22006, 0.72077, 0.85079, 0.87737, 0.43646, 0.53535, 0.24215, 0.91807, 0.49793, 0.94274, 0.30852};
        bRowIndices = new int[]{1, 1, 1, 2, 2, 4, 5, 6, 7, 9, 12, 12, 12, 13, 16, 18, 20, 20, 20, 22};
        bColIndices = new int[]{6, 7, 9, 8, 9, 0, 6, 10, 3, 10, 2, 4, 9, 6, 9, 2, 4, 5, 7, 9};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1706499282},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.17994023050000002, 0.0, 0.21239972350000003, 0.0, 0.0, 0.0, 0.0, 0.2190354205, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.044421155999999996, 0.0, 0.0, 0.6456006479999999, 0.0, 0.0, 0.03822871320000001, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11389021279999999, 0.021917495599999996, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6236384147999999, 0.12001551209999999, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27243241560000003, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.1748, 0.1456, 0.25279, 0.97145};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{1, 1, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 1);
        bEntries = new double[]{0.52007, 0.66995, 0.32649};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 0, 0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.11710726000000002},
                {0.09754472000000002},
                {0.0},
                {0.16935666050000003},
                {0.6508229275}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.35419, 0.91318, 0.69528, 0.14512};
        aRowIndices = new int[]{0, 2, 3, 4};
        aColIndices = new int[]{0, 0, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new double[]{0.3606, 0.40418, 0.01176};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.127720914, 0.0},
                {0.0, 0.0},
                {0.329292708, 0.0},
                {0.2810182704, 0.0},
                {0.052330272, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.607, 0.14284, 0.45743, 0.77819};
        aRowIndices = new int[]{1, 2, 3, 3};
        aColIndices = new int[]{0, 1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 5);
        bEntries = new double[]{0.96396, 0.96965};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{3, 3};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final0a = a;
        CooMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.mult(final0b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.14191, 0.02704, 0.38007, 0.80064};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{1, 1, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.61101, 0.61258};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{4, 0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final1a = a;
        CooMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.mult(final1b));
    }


    @Test
    void multSparseComplexTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        Complex128[] bEntries;
        CooCMatrix b;

        Complex128[][] expEntries;
        CMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.33137, 0.18814, 0.95844};
        aRowIndices = new int[]{0, 2, 2};
        aColIndices = new int[]{4, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 6);
        bEntries = new Complex128[]{new Complex128("0.97659+0.45171i"), new Complex128("0.46223+0.97387i"), new Complex128("0.41417+0.44065i"), new Complex128("0.24366+0.1466i"), new Complex128("0.85504+0.88395i"), new Complex128("0.57655+0.76298i"), new Complex128("0.9899+0.63576i"), new Complex128("0.80167+0.75526i")};
        bRowIndices = new int[]{0, 1, 2, 2, 2, 4, 4, 4};
        bColIndices = new int[]{4, 0, 2, 4, 5, 0, 1, 5};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.1910513735+0.2528286826i"), new Complex128("0.328023163+0.21067179119999999i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.2656493879+0.2502705062i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.39695709479999997+0.42233658599999996i"), new Complex128("0.0"), new Complex128("0.417269133+0.22549202340000002i"), new Complex128("0.8195045376+0.847213038i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.68187, 0.32014, 0.79403, 0.88781, 0.93595, 0.13241, 0.68188, 0.79228, 0.51634, 0.2253, 0.72079, 0.91121, 0.15124, 0.39048, 0.25872};
        aRowIndices = new int[]{1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 6, 6, 9, 10, 10};
        aColIndices = new int[]{0, 3, 6, 1, 2, 18, 20, 6, 20, 21, 13, 19, 20, 3, 13};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(23, 11);
        bEntries = new Complex128[]{new Complex128("0.38502+0.04552i"), new Complex128("0.88032+0.92607i"), new Complex128("0.69907+0.14743i"), new Complex128("0.07625+0.69487i"), new Complex128("0.94955+0.72765i"), new Complex128("0.46059+0.2665i"), new Complex128("0.29256+0.59195i"), new Complex128("0.35903+0.22833i"), new Complex128("0.68831+0.51803i"), new Complex128("0.79154+0.54282i"), new Complex128("0.05943+0.73128i"), new Complex128("0.12081+0.91091i"), new Complex128("0.26697+0.15171i"), new Complex128("0.61822+0.37349i"), new Complex128("0.98401+0.95508i"), new Complex128("0.02764+0.09211i"), new Complex128("0.30162+0.36956i"), new Complex128("0.16954+0.90695i"), new Complex128("0.71821+0.88465i"), new Complex128("0.97409+0.48991i")};
        bRowIndices = new int[]{0, 0, 0, 1, 1, 7, 9, 11, 12, 14, 14, 14, 14, 16, 17, 18, 18, 18, 19, 22};
        bColIndices = new int[]{1, 7, 8, 4, 10, 0, 10, 4, 10, 4, 5, 6, 10, 9, 5, 5, 6, 8, 8, 6};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.2625335874+0.031038722399999996i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.6002637984+0.6314593508999999i"), new Complex128("0.47667486089999994+0.1005280941i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0676955125+0.6169125347i"), new Complex128("0.0036598124000000003+0.012196285099999999i"), new Complex128("0.0399375042+0.0489334396i"), new Complex128("0.0"), new Complex128("0.0224487914+0.12008924950000001i"), new Complex128("0.0"), new Complex128("0.8430199855+0.6460149465i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.6544401341+0.8061019265i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.07754, 0.56599, 0.12724, 0.88643};
        aRowIndices = new int[]{0, 1, 1, 1};
        aColIndices = new int[]{1, 0, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 1);
        bEntries = new Complex128[]{new Complex128("0.7751+0.28177i"), new Complex128("0.19258+0.76919i"), new Complex128("0.41388+0.04817i")};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 0, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.0149326532+0.0596429926i")},
                {new Complex128("0.8300783766000001+0.30005007100000003i")},
                {new Complex128("0.0")},
                {new Complex128("0.0")},
                {new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.68362, 0.30186, 0.79635, 0.33173};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 1, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new Complex128[]{new Complex128("0.8595+0.96772i"), new Complex128("0.2566+0.82298i"), new Complex128("0.31544+0.16909i")};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 1, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.175416892+0.5626055876i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.251200644+0.1346548215i"), new Complex128("0.077457276+0.24842474280000001i")},
                {new Complex128("0.28512193500000005+0.3210217556i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.mult(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.7739, 0.85143, 0.0054, 0.88411};
        aRowIndices = new int[]{0, 0, 1, 4};
        aColIndices = new int[]{1, 2, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 5);
        bEntries = new Complex128[]{new Complex128("0.22478+0.28297i"), new Complex128("0.76721+0.81123i")};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{1, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final0a = a;
        CooCMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.mult(final0b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.11658, 0.17026, 0.57168, 0.11777};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{1, 0, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new Complex128[]{new Complex128("0.12542+0.21611i"), new Complex128("0.14945+0.22213i")};
        bRowIndices = new int[]{3, 4};
        bColIndices = new int[]{0, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final1a = a;
        CooCMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.mult(final1b));
    }
}
