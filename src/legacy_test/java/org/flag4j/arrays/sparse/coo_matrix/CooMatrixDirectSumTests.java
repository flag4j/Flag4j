package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.DirectSum;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooMatrixDirectSumTests {

    @Test
    void realSparseDirectSumTest() {
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
        aEntries = new double[]{0.36314};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.26466, 0.5763};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new double[]{0.36314, 0.26466, 0.5763};
        expRowIndices = new int[]{1, 2, 4};
        expColIndices = new int[]{1, 3, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.07606, 0.91556};
        bRowIndices = new int[]{1, 4};
        bColIndices = new int[]{1, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new double[]{0.07606, 0.91556};
        expRowIndices = new int[]{2, 5};
        expColIndices = new int[]{3, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.57548, 0.04127, 0.84891, 0.95484, 0.91981, 0.76013, 0.98434, 0.95096, 0.83673, 0.84207, 0.36798, 0.25599, 0.43903, 0.5061};
        aRowIndices = new int[]{0, 1, 1, 2, 3, 3, 3, 4, 6, 7, 9, 10, 11, 13};
        aColIndices = new int[]{1, 2, 4, 2, 0, 1, 2, 0, 3, 0, 3, 0, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.47467, 0.23944, 0.80356, 0.30997, 0.87977, 0.62583, 0.65635, 0.88966, 0.27114, 0.93367, 0.59854, 0.67495, 0.58148, 0.22202, 0.90246, 0.04564, 0.21657};
        bRowIndices = new int[]{0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 7, 7, 8, 9, 10, 10, 12};
        bColIndices = new int[]{2, 2, 5, 4, 5, 3, 5, 4, 2, 3, 0, 5, 5, 3, 2, 3, 3};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new double[]{0.57548, 0.04127, 0.84891, 0.95484, 0.91981, 0.76013, 0.98434, 0.95096, 0.83673, 0.84207, 0.36798, 0.25599, 0.43903, 0.5061, 0.47467, 0.23944, 0.80356, 0.30997, 0.87977, 0.62583, 0.65635, 0.88966, 0.27114, 0.93367, 0.59854, 0.67495, 0.58148, 0.22202, 0.90246, 0.04564, 0.21657};
        expRowIndices = new int[]{0, 1, 1, 2, 3, 3, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 15, 16, 16, 17, 17, 19, 20, 20, 21, 21, 22, 23, 24, 24, 26};
        expColIndices = new int[]{1, 2, 4, 2, 0, 1, 2, 0, 3, 0, 3, 0, 2, 2, 7, 7, 10, 9, 10, 8, 10, 9, 7, 8, 5, 10, 10, 8, 7, 8, 8};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void complexSparseDirectSumTest() {
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

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.12576};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new Complex128[]{new Complex128("0.29068+0.52611i"), new Complex128("0.05686+0.1533i")};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{0, 2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new Complex128[]{new Complex128("0.12576"), new Complex128("0.29068+0.52611i"), new Complex128("0.05686+0.1533i")};
        expRowIndices = new int[]{1, 2, 3};
        expColIndices = new int[]{0, 3, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new Complex128[]{new Complex128("0.76653+0.511i"), new Complex128("0.98272+0.34485i")};
        bRowIndices = new int[]{1, 4};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new Complex128[]{new Complex128("0.76653+0.511i"), new Complex128("0.98272+0.34485i")};
        expRowIndices = new int[]{2, 5};
        expColIndices = new int[]{2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.4423, 0.27239, 0.56716, 0.40177, 0.09355, 0.6171, 0.75503, 0.84551, 0.56827, 0.84476, 0.52399, 0.04809, 0.53526, 0.29315};
        aRowIndices = new int[]{0, 0, 1, 1, 3, 4, 6, 6, 8, 10, 10, 12, 12, 13};
        aColIndices = new int[]{2, 3, 0, 4, 0, 4, 1, 3, 0, 0, 2, 0, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new Complex128[]{new Complex128("0.19656+0.15389i"), new Complex128("0.99278+0.64363i"), new Complex128("0.51115+0.66117i"), new Complex128("0.98703+0.44285i"), new Complex128("0.55436+0.00072i"), new Complex128("0.33388+0.30031i"), new Complex128("0.25571+0.90675i"), new Complex128("0.13093+0.00301i"), new Complex128("0.56693+0.46193i"), new Complex128("0.375+0.12438i"), new Complex128("0.26962+0.61082i"), new Complex128("0.97709+0.03678i"), new Complex128("0.40657+0.68236i"), new Complex128("0.40298+0.3305i"), new Complex128("0.50161+0.23079i"), new Complex128("0.80417+0.43609i"), new Complex128("0.33521+0.83137i")};
        bRowIndices = new int[]{0, 1, 3, 5, 6, 7, 7, 8, 8, 8, 9, 9, 10, 11, 11, 11, 11};
        bColIndices = new int[]{4, 3, 3, 4, 4, 3, 5, 0, 1, 2, 0, 5, 5, 1, 2, 3, 5};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new Complex128[]{new Complex128("0.4423"), new Complex128("0.27239"), new Complex128("0.56716"), new Complex128("0.40177"), new Complex128("0.09355"), new Complex128("0.6171"), new Complex128("0.75503"), new Complex128("0.84551"), new Complex128("0.56827"), new Complex128("0.84476"), new Complex128("0.52399"), new Complex128("0.04809"), new Complex128("0.53526"), new Complex128("0.29315"), new Complex128("0.19656+0.15389i"), new Complex128("0.99278+0.64363i"), new Complex128("0.51115+0.66117i"), new Complex128("0.98703+0.44285i"), new Complex128("0.55436+0.00072i"), new Complex128("0.33388+0.30031i"), new Complex128("0.25571+0.90675i"), new Complex128("0.13093+0.00301i"), new Complex128("0.56693+0.46193i"), new Complex128("0.375+0.12438i"), new Complex128("0.26962+0.61082i"), new Complex128("0.97709+0.03678i"), new Complex128("0.40657+0.68236i"), new Complex128("0.40298+0.3305i"), new Complex128("0.50161+0.23079i"), new Complex128("0.80417+0.43609i"), new Complex128("0.33521+0.83137i")};
        expRowIndices = new int[]{0, 0, 1, 1, 3, 4, 6, 6, 8, 10, 10, 12, 12, 13, 14, 15, 17, 19, 20, 21, 21, 22, 22, 22, 23, 23, 24, 25, 25, 25, 25};
        expColIndices = new int[]{2, 3, 0, 4, 0, 4, 1, 3, 0, 0, 2, 0, 2, 4, 9, 8, 8, 9, 9, 8, 10, 5, 6, 7, 5, 10, 10, 6, 7, 8, 10};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void realDenseDirectSumTest() {
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
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.9471};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.78063, 0.14848, 0.84453},
                {0.29046, 0.90379, 0.19881},
                {0.60646, 0.17191, 0.80605},
                {0.35467, 0.82455, 0.66883}};
        b = new Matrix(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new double[]{0.9471, 0.78063, 0.14848, 0.84453, 0.29046, 0.90379, 0.19881, 0.60646, 0.17191, 0.80605, 0.35467, 0.82455, 0.66883};
        expRowIndices = new int[]{0, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5};
        expColIndices = new int[]{2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.95946, 0.46849},
                {0.1267, 0.1966},
                {0.46751, 0.28133},
                {0.40378, 0.34267},
                {0.68666, 0.04621}};
        b = new Matrix(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new double[]{0.95946, 0.46849, 0.1267, 0.1966, 0.46751, 0.28133, 0.40378, 0.34267, 0.68666, 0.04621};
        expRowIndices = new int[]{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.53133, 0.69632, 0.10432, 0.45471, 0.8554, 0.70034, 0.00942, 0.82027, 0.03932, 0.47388, 0.97456, 0.33064, 0.24773, 0.19777};
        aRowIndices = new int[]{1, 1, 1, 2, 4, 5, 6, 7, 7, 7, 8, 10, 11, 12};
        aColIndices = new int[]{0, 1, 3, 0, 2, 2, 3, 0, 1, 2, 2, 2, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.65902, 0.35946, 0.78255, 0.78345, 0.0983, 0.38273},
                {0.28829, 0.48223, 0.38455, 0.50828, 0.86524, 0.7055},
                {0.44414, 0.73977, 0.90158, 0.99077, 0.24698, 0.44316},
                {0.48544, 0.22846, 0.72684, 0.38091, 0.12083, 0.8205},
                {0.0789, 0.35711, 0.19032, 0.47534, 0.73126, 0.82564},
                {0.30244, 0.43364, 0.74498, 0.48812, 0.64507, 0.91902},
                {0.86838, 0.02126, 0.49515, 0.61257, 0.92955, 0.92565},
                {0.20163, 0.40987, 0.18285, 0.49095, 0.16805, 0.62921},
                {0.15065, 0.07543, 0.32682, 0.73049, 0.58299, 0.16706},
                {0.22822, 0.4694, 0.19263, 0.99446, 0.27772, 0.86876},
                {0.58823, 0.84645, 0.60044, 0.41445, 0.08495, 0.3587},
                {0.83611, 0.39451, 0.84648, 0.37964, 0.31351, 0.85914},
                {0.46213, 0.56689, 0.77718, 0.38792, 0.72643, 0.70424},
                {0.53557, 0.3165, 0.47377, 0.93039, 0.35041, 0.69137}};
        b = new Matrix(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new double[]{0.53133, 0.69632, 0.10432, 0.45471, 0.8554, 0.70034, 0.00942, 0.82027, 0.03932, 0.47388, 0.97456, 0.33064, 0.24773, 0.19777, 0.65902, 0.35946, 0.78255, 0.78345, 0.0983, 0.38273, 0.28829, 0.48223, 0.38455, 0.50828, 0.86524, 0.7055, 0.44414, 0.73977, 0.90158, 0.99077, 0.24698, 0.44316, 0.48544, 0.22846, 0.72684, 0.38091, 0.12083, 0.8205, 0.0789, 0.35711, 0.19032, 0.47534, 0.73126, 0.82564, 0.30244, 0.43364, 0.74498, 0.48812, 0.64507, 0.91902, 0.86838, 0.02126, 0.49515, 0.61257, 0.92955, 0.92565, 0.20163, 0.40987, 0.18285, 0.49095, 0.16805, 0.62921, 0.15065, 0.07543, 0.32682, 0.73049, 0.58299, 0.16706, 0.22822, 0.4694, 0.19263, 0.99446, 0.27772, 0.86876, 0.58823, 0.84645, 0.60044, 0.41445, 0.08495, 0.3587, 0.83611, 0.39451, 0.84648, 0.37964, 0.31351, 0.85914, 0.46213, 0.56689, 0.77718, 0.38792, 0.72643, 0.70424, 0.53557, 0.3165, 0.47377, 0.93039, 0.35041, 0.69137};
        expRowIndices = new int[]{1, 1, 1, 2, 4, 5, 6, 7, 7, 7, 8, 10, 11, 12, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27};
        expColIndices = new int[]{0, 1, 3, 0, 2, 2, 3, 0, 1, 2, 2, 2, 0, 2, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void complexDenseDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Complex128[][] bEntries;
        CMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.4884};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.57432+0.16659i"), new Complex128("0.47122+0.62557i"), new Complex128("0.61823+0.61842i")},
                {new Complex128("0.62178+0.95351i"), new Complex128("0.25424+0.19833i"), new Complex128("0.05743+0.75706i")},
                {new Complex128("0.28481+0.34564i"), new Complex128("0.16388+0.69526i"), new Complex128("0.22402+0.84895i")},
                {new Complex128("0.53869+0.9115i"), new Complex128("0.75743+0.82263i"), new Complex128("0.84954+0.75556i")}};
        b = new CMatrix(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new Complex128[]{new Complex128("0.4884"), new Complex128("0.57432+0.16659i"), new Complex128("0.47122+0.62557i"), new Complex128("0.61823+0.61842i"), new Complex128("0.62178+0.95351i"), new Complex128("0.25424+0.19833i"), new Complex128("0.05743+0.75706i"), new Complex128("0.28481+0.34564i"), new Complex128("0.16388+0.69526i"), new Complex128("0.22402+0.84895i"), new Complex128("0.53869+0.9115i"), new Complex128("0.75743+0.82263i"), new Complex128("0.84954+0.75556i")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5};
        expColIndices = new int[]{0, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.20067+0.76104i"), new Complex128("0.769+0.51839i")},
                {new Complex128("0.37303+0.69709i"), new Complex128("0.76031+0.35138i")},
                {new Complex128("0.24139+0.51966i"), new Complex128("0.67086+0.96679i")},
                {new Complex128("0.26419+0.67671i"), new Complex128("0.41824+0.79949i")},
                {new Complex128("0.42013+0.98147i"), new Complex128("0.82586+0.99174i")}};
        b = new CMatrix(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new Complex128[]{new Complex128("0.20067+0.76104i"), new Complex128("0.769+0.51839i"), new Complex128("0.37303+0.69709i"), new Complex128("0.76031+0.35138i"), new Complex128("0.24139+0.51966i"), new Complex128("0.67086+0.96679i"), new Complex128("0.26419+0.67671i"), new Complex128("0.41824+0.79949i"), new Complex128("0.42013+0.98147i"), new Complex128("0.82586+0.99174i")};
        expRowIndices = new int[]{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.18246, 0.62977, 0.54024, 0.1938, 0.72261, 0.03498, 0.10563, 0.66627, 0.25401, 0.49059, 0.54795, 0.89787, 0.79596, 0.56759};
        aRowIndices = new int[]{0, 0, 1, 2, 3, 6, 7, 8, 9, 9, 10, 11, 12, 12};
        aColIndices = new int[]{1, 4, 4, 0, 1, 1, 4, 0, 1, 4, 1, 2, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.46846+0.10833i"), new Complex128("0.14377+0.75617i"), new Complex128("0.07594+0.29978i"), new Complex128("0.56306+0.82182i"), new Complex128("0.19916+0.08544i"), new Complex128("0.39548+0.62883i")},
                {new Complex128("0.67248+0.81796i"), new Complex128("0.35025+0.69244i"), new Complex128("0.61628+0.73238i"), new Complex128("0.5961+0.88939i"), new Complex128("0.21971+0.65859i"), new Complex128("0.8511+0.31135i")},
                {new Complex128("0.0449+0.0442i"), new Complex128("0.11554+0.87542i"), new Complex128("0.56225+0.39056i"), new Complex128("0.85716+0.46189i"), new Complex128("0.60369+0.98709i"), new Complex128("0.56481+0.74681i")},
                {new Complex128("0.17488+0.91857i"), new Complex128("0.63517+0.21242i"), new Complex128("0.7328+0.80571i"), new Complex128("0.32908+0.40579i"), new Complex128("0.55495+0.42956i"), new Complex128("0.06775+0.65319i")},
                {new Complex128("0.24194+0.34182i"), new Complex128("0.75516+0.23242i"), new Complex128("0.30002+0.52049i"), new Complex128("0.0195+0.66491i"), new Complex128("0.11195+0.99746i"), new Complex128("0.06188+0.76093i")},
                {new Complex128("0.02372+0.88658i"), new Complex128("0.80361+0.77282i"), new Complex128("0.62455+0.86084i"), new Complex128("0.48187+0.41808i"), new Complex128("0.53663+0.0254i"), new Complex128("0.45026+0.77011i")},
                {new Complex128("0.16472+0.57632i"), new Complex128("0.36278+0.92842i"), new Complex128("0.2307+0.63646i"), new Complex128("0.49377+0.31166i"), new Complex128("0.63059+0.51516i"), new Complex128("0.00098+0.93881i")},
                {new Complex128("0.96944+0.83608i"), new Complex128("0.07921+0.71263i"), new Complex128("0.46583+0.25113i"), new Complex128("0.99034+0.57762i"), new Complex128("0.90913+0.56968i"), new Complex128("0.51301+0.02804i")},
                {new Complex128("0.94222+0.00757i"), new Complex128("0.27242+0.95991i"), new Complex128("0.74864+0.97997i"), new Complex128("0.87243+0.02436i"), new Complex128("0.68545+0.13708i"), new Complex128("0.73992+0.67858i")},
                {new Complex128("0.46409+0.66538i"), new Complex128("0.29854+0.91704i"), new Complex128("0.75629+0.41053i"), new Complex128("0.51912+0.93605i"), new Complex128("0.66452+0.97836i"), new Complex128("0.78868+0.60342i")},
                {new Complex128("0.80154+0.02122i"), new Complex128("0.05816+0.57239i"), new Complex128("0.23776+0.29626i"), new Complex128("0.55069+0.60787i"), new Complex128("0.6668+0.873i"), new Complex128("0.70669+0.45406i")},
                {new Complex128("0.0234+0.02001i"), new Complex128("0.52699+0.32663i"), new Complex128("0.89666+0.28074i"), new Complex128("0.48736+0.80453i"), new Complex128("0.46938+0.79886i"), new Complex128("0.29248+0.41249i")},
                {new Complex128("0.2419+0.84832i"), new Complex128("0.81226+0.54305i"), new Complex128("0.96564+0.82395i"), new Complex128("0.18676+0.93304i"), new Complex128("0.66785+0.29056i"), new Complex128("0.42223+0.83921i")},
                {new Complex128("0.11808+0.94414i"), new Complex128("0.46313+0.88265i"), new Complex128("0.77841+0.69067i"), new Complex128("0.30383+0.41319i"), new Complex128("0.15568+0.47926i"), new Complex128("0.94116+0.97658i")}};
        b = new CMatrix(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new Complex128[]{new Complex128("0.18246"), new Complex128("0.62977"), new Complex128("0.54024"), new Complex128("0.1938"), new Complex128("0.72261"), new Complex128("0.03498"), new Complex128("0.10563"), new Complex128("0.66627"), new Complex128("0.25401"), new Complex128("0.49059"), new Complex128("0.54795"), new Complex128("0.89787"), new Complex128("0.79596"), new Complex128("0.56759"), new Complex128("0.46846+0.10833i"), new Complex128("0.14377+0.75617i"), new Complex128("0.07594+0.29978i"), new Complex128("0.56306+0.82182i"), new Complex128("0.19916+0.08544i"), new Complex128("0.39548+0.62883i"), new Complex128("0.67248+0.81796i"), new Complex128("0.35025+0.69244i"), new Complex128("0.61628+0.73238i"), new Complex128("0.5961+0.88939i"), new Complex128("0.21971+0.65859i"), new Complex128("0.8511+0.31135i"), new Complex128("0.0449+0.0442i"), new Complex128("0.11554+0.87542i"), new Complex128("0.56225+0.39056i"), new Complex128("0.85716+0.46189i"), new Complex128("0.60369+0.98709i"), new Complex128("0.56481+0.74681i"), new Complex128("0.17488+0.91857i"), new Complex128("0.63517+0.21242i"), new Complex128("0.7328+0.80571i"), new Complex128("0.32908+0.40579i"), new Complex128("0.55495+0.42956i"), new Complex128("0.06775+0.65319i"), new Complex128("0.24194+0.34182i"), new Complex128("0.75516+0.23242i"), new Complex128("0.30002+0.52049i"), new Complex128("0.0195+0.66491i"), new Complex128("0.11195+0.99746i"), new Complex128("0.06188+0.76093i"), new Complex128("0.02372+0.88658i"), new Complex128("0.80361+0.77282i"), new Complex128("0.62455+0.86084i"), new Complex128("0.48187+0.41808i"), new Complex128("0.53663+0.0254i"), new Complex128("0.45026+0.77011i"), new Complex128("0.16472+0.57632i"), new Complex128("0.36278+0.92842i"), new Complex128("0.2307+0.63646i"), new Complex128("0.49377+0.31166i"), new Complex128("0.63059+0.51516i"), new Complex128("0.00098+0.93881i"), new Complex128("0.96944+0.83608i"), new Complex128("0.07921+0.71263i"), new Complex128("0.46583+0.25113i"), new Complex128("0.99034+0.57762i"), new Complex128("0.90913+0.56968i"), new Complex128("0.51301+0.02804i"), new Complex128("0.94222+0.00757i"), new Complex128("0.27242+0.95991i"), new Complex128("0.74864+0.97997i"), new Complex128("0.87243+0.02436i"), new Complex128("0.68545+0.13708i"), new Complex128("0.73992+0.67858i"), new Complex128("0.46409+0.66538i"), new Complex128("0.29854+0.91704i"), new Complex128("0.75629+0.41053i"), new Complex128("0.51912+0.93605i"), new Complex128("0.66452+0.97836i"), new Complex128("0.78868+0.60342i"), new Complex128("0.80154+0.02122i"), new Complex128("0.05816+0.57239i"), new Complex128("0.23776+0.29626i"), new Complex128("0.55069+0.60787i"), new Complex128("0.6668+0.873i"), new Complex128("0.70669+0.45406i"), new Complex128("0.0234+0.02001i"), new Complex128("0.52699+0.32663i"), new Complex128("0.89666+0.28074i"), new Complex128("0.48736+0.80453i"), new Complex128("0.46938+0.79886i"), new Complex128("0.29248+0.41249i"), new Complex128("0.2419+0.84832i"), new Complex128("0.81226+0.54305i"), new Complex128("0.96564+0.82395i"), new Complex128("0.18676+0.93304i"), new Complex128("0.66785+0.29056i"), new Complex128("0.42223+0.83921i"), new Complex128("0.11808+0.94414i"), new Complex128("0.46313+0.88265i"), new Complex128("0.77841+0.69067i"), new Complex128("0.30383+0.41319i"), new Complex128("0.15568+0.47926i"), new Complex128("0.94116+0.97658i")};
        expRowIndices = new int[]{0, 0, 1, 2, 3, 6, 7, 8, 9, 9, 10, 11, 12, 12, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27};
        expColIndices = new int[]{1, 4, 4, 0, 1, 1, 4, 0, 1, 4, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void realSparseInvDirectSumTest() {
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
        aEntries = new double[]{0.06509};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.50055, 0.7369};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{2, 2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new double[]{0.50055, 0.7369, 0.06509};
        expRowIndices = new int[]{0, 1, 5};
        expColIndices = new int[]{5, 5, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.99751, 0.34404};
        bRowIndices = new int[]{3, 4};
        bColIndices = new int[]{1, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new double[]{0.99751, 0.34404};
        expRowIndices = new int[]{3, 4};
        expColIndices = new int[]{3, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.40097, 0.89822, 0.14835, 0.98776, 0.36868, 0.71529, 0.42539, 0.72885, 0.57388, 0.51657, 0.51903, 0.41499, 0.29246, 0.07744};
        aRowIndices = new int[]{2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 12, 13};
        aColIndices = new int[]{2, 4, 1, 4, 3, 0, 4, 1, 4, 2, 1, 2, 3, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.40774, 0.73689, 0.77809, 0.63376, 0.96878, 0.16107, 0.10433, 0.23923, 0.13031, 0.58009, 0.2367, 0.46062, 0.41659, 0.49808, 0.40022, 0.00312, 0.95164};
        bRowIndices = new int[]{0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 9, 10, 11, 11};
        bColIndices = new int[]{2, 5, 2, 5, 1, 4, 1, 4, 2, 2, 4, 4, 5, 4, 1, 1, 3};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new double[]{0.40774, 0.73689, 0.77809, 0.63376, 0.96878, 0.16107, 0.10433, 0.23923, 0.13031, 0.58009, 0.2367, 0.46062, 0.41659, 0.49808, 0.40022, 0.00312, 0.95164, 0.40097, 0.89822, 0.14835, 0.98776, 0.36868, 0.71529, 0.42539, 0.72885, 0.57388, 0.51657, 0.51903, 0.41499, 0.29246, 0.07744};
        expRowIndices = new int[]{0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 9, 10, 11, 11, 16, 16, 17, 17, 18, 19, 20, 21, 22, 23, 25, 26, 26, 27};
        expColIndices = new int[]{7, 10, 7, 10, 6, 9, 6, 9, 7, 7, 9, 9, 10, 9, 6, 6, 8, 2, 4, 1, 4, 3, 0, 4, 1, 4, 2, 1, 2, 3, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }


    @Test
    void complexSparseInvDirectSumTest() {
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

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.95463};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new Complex128[]{new Complex128("0.53322+0.92647i"), new Complex128("0.61992+0.2719i")};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{2, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new Complex128[]{new Complex128("0.53322+0.92647i"), new Complex128("0.61992+0.2719i"), new Complex128("0.95463")};
        expRowIndices = new int[]{0, 1, 5};
        expColIndices = new int[]{5, 4, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new Complex128[]{new Complex128("0.88968+0.52092i"), new Complex128("0.60458+0.94442i")};
        bRowIndices = new int[]{1, 4};
        bColIndices = new int[]{1, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new Complex128[]{new Complex128("0.88968+0.52092i"), new Complex128("0.60458+0.94442i")};
        expRowIndices = new int[]{1, 4};
        expColIndices = new int[]{3, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.45761, 0.59949, 0.93871, 0.69319, 0.86086, 0.07661, 0.05989, 0.42798, 0.2897, 0.71213, 0.26142, 0.5184, 0.37619, 0.56648};
        aRowIndices = new int[]{1, 1, 2, 3, 3, 4, 4, 6, 7, 8, 11, 13, 13, 13};
        aColIndices = new int[]{1, 4, 0, 0, 1, 0, 4, 3, 1, 3, 2, 0, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new Complex128[]{new Complex128("0.52156+0.16834i"), new Complex128("0.75412+0.03679i"), new Complex128("0.48347+0.18359i"), new Complex128("0.67699+0.24396i"), new Complex128("0.48046+0.83665i"), new Complex128("0.18781+0.33929i"), new Complex128("0.672+0.25296i"), new Complex128("0.62251+0.99364i"), new Complex128("0.35781+0.5819i"), new Complex128("0.25598+0.25756i"), new Complex128("0.40395+0.2463i"), new Complex128("0.64474+0.52902i"), new Complex128("0.59587+0.81059i"), new Complex128("0.74482+0.85024i"), new Complex128("0.63444+0.45007i"), new Complex128("0.08587+0.97854i"), new Complex128("0.31249+0.90372i")};
        bRowIndices = new int[]{1, 2, 2, 3, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 13, 13};
        bColIndices = new int[]{3, 2, 4, 1, 5, 3, 0, 2, 4, 5, 1, 1, 0, 1, 1, 3, 5};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new Complex128[]{new Complex128("0.52156+0.16834i"), new Complex128("0.75412+0.03679i"), new Complex128("0.48347+0.18359i"), new Complex128("0.67699+0.24396i"), new Complex128("0.48046+0.83665i"), new Complex128("0.18781+0.33929i"), new Complex128("0.672+0.25296i"), new Complex128("0.62251+0.99364i"), new Complex128("0.35781+0.5819i"), new Complex128("0.25598+0.25756i"), new Complex128("0.40395+0.2463i"), new Complex128("0.64474+0.52902i"), new Complex128("0.59587+0.81059i"), new Complex128("0.74482+0.85024i"), new Complex128("0.63444+0.45007i"), new Complex128("0.08587+0.97854i"), new Complex128("0.31249+0.90372i"), new Complex128("0.45761"), new Complex128("0.59949"), new Complex128("0.93871"), new Complex128("0.69319"), new Complex128("0.86086"), new Complex128("0.07661"), new Complex128("0.05989"), new Complex128("0.42798"), new Complex128("0.2897"), new Complex128("0.71213"), new Complex128("0.26142"), new Complex128("0.5184"), new Complex128("0.37619"), new Complex128("0.56648")};
        expRowIndices = new int[]{1, 2, 2, 3, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 13, 13, 15, 15, 16, 17, 17, 18, 18, 20, 21, 22, 25, 27, 27, 27};
        expColIndices = new int[]{8, 7, 9, 6, 10, 8, 5, 7, 9, 10, 6, 6, 5, 6, 6, 8, 10, 1, 4, 0, 0, 1, 0, 4, 3, 1, 3, 2, 0, 3, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }


    @Test
    void realDenseInvDirectSumTest() {
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
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.0778};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.76532, 0.51659, 0.54962},
                {0.39302, 0.53839, 0.61549},
                {0.50429, 0.50156, 0.30362},
                {0.93078, 0.34518, 0.6854}};
        b = new Matrix(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new double[]{0.76532, 0.51659, 0.54962, 0.39302, 0.53839, 0.61549, 0.50429, 0.50156, 0.30362, 0.93078, 0.34518, 0.6854, 0.0778};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
        expColIndices = new int[]{3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.9895, 0.66022},
                {0.31201, 0.15653},
                {0.0436, 0.8439},
                {0.06687, 0.30178},
                {0.02106, 0.84792}};
        b = new Matrix(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new double[]{0.9895, 0.66022, 0.31201, 0.15653, 0.0436, 0.8439, 0.06687, 0.30178, 0.02106, 0.84792};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.75921, 0.23954, 0.45291, 0.30381, 0.14104, 0.99878, 0.34272, 0.06406, 0.14733, 0.75918, 0.79406, 0.68084, 0.10748, 0.19839};
        aRowIndices = new int[]{0, 2, 2, 3, 5, 6, 6, 8, 8, 9, 10, 10, 11, 12};
        aColIndices = new int[]{3, 3, 4, 3, 3, 1, 2, 1, 2, 2, 1, 4, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.26822, 0.37849, 0.35694, 0.10746, 0.00328, 0.3996},
                {0.24014, 0.12577, 0.31152, 0.40922, 0.02657, 0.57162},
                {0.47627, 0.18149, 0.01357, 0.11488, 0.88359, 0.27094},
                {0.52003, 0.95039, 0.24626, 0.55939, 0.68338, 0.90858},
                {0.56346, 0.18074, 0.99682, 0.44922, 0.91819, 0.41014},
                {0.87124, 0.55163, 0.954, 0.48999, 0.95899, 0.66659},
                {0.47058, 0.00265, 0.28464, 0.6164, 0.27142, 0.33421},
                {0.94399, 0.43872, 0.82363, 0.07901, 0.79046, 0.64227},
                {0.92647, 0.67403, 0.70681, 0.98536, 0.96611, 0.00183},
                {0.42248, 0.00074, 0.8854, 0.06665, 0.69193, 0.71229},
                {0.33101, 0.71886, 0.99038, 0.7506, 0.59483, 0.07408},
                {0.73345, 0.21992, 0.18397, 0.13365, 0.32326, 0.38326},
                {0.34359, 0.40562, 0.8703, 0.65249, 0.37304, 0.94381},
                {0.847, 0.60078, 0.131, 0.04455, 0.28184, 0.17891}};
        b = new Matrix(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new double[]{0.26822, 0.37849, 0.35694, 0.10746, 0.00328, 0.3996, 0.24014, 0.12577, 0.31152, 0.40922, 0.02657, 0.57162, 0.47627, 0.18149, 0.01357, 0.11488, 0.88359, 0.27094, 0.52003, 0.95039, 0.24626, 0.55939, 0.68338, 0.90858, 0.56346, 0.18074, 0.99682, 0.44922, 0.91819, 0.41014, 0.87124, 0.55163, 0.954, 0.48999, 0.95899, 0.66659, 0.47058, 0.00265, 0.28464, 0.6164, 0.27142, 0.33421, 0.94399, 0.43872, 0.82363, 0.07901, 0.79046, 0.64227, 0.92647, 0.67403, 0.70681, 0.98536, 0.96611, 0.00183, 0.42248, 0.00074, 0.8854, 0.06665, 0.69193, 0.71229, 0.33101, 0.71886, 0.99038, 0.7506, 0.59483, 0.07408, 0.73345, 0.21992, 0.18397, 0.13365, 0.32326, 0.38326, 0.34359, 0.40562, 0.8703, 0.65249, 0.37304, 0.94381, 0.847, 0.60078, 0.131, 0.04455, 0.28184, 0.17891, 0.75921, 0.23954, 0.45291, 0.30381, 0.14104, 0.99878, 0.34272, 0.06406, 0.14733, 0.75918, 0.79406, 0.68084, 0.10748, 0.19839};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 16, 16, 17, 19, 20, 20, 22, 22, 23, 24, 24, 25, 26};
        expColIndices = new int[]{5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 3, 3, 4, 3, 3, 1, 2, 1, 2, 2, 1, 4, 1, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }


    @Test
    void complexDenseInvDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Complex128[][] bEntries;
        CMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.24679};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.24736+0.75522i"), new Complex128("0.67449+0.37946i"), new Complex128("0.80865+0.74668i")},
                {new Complex128("0.40987+0.0367i"), new Complex128("0.67109+0.40036i"), new Complex128("0.07186+0.12732i")},
                {new Complex128("0.39774+0.05174i"), new Complex128("0.93539+0.47693i"), new Complex128("0.86447+0.59669i")},
                {new Complex128("0.2025+0.06263i"), new Complex128("0.01157+0.53881i"), new Complex128("0.49949+0.33721i")}};
        b = new CMatrix(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new Complex128[]{new Complex128("0.24736+0.75522i"), new Complex128("0.67449+0.37946i"), new Complex128("0.80865+0.74668i"), new Complex128("0.40987+0.0367i"), new Complex128("0.67109+0.40036i"), new Complex128("0.07186+0.12732i"), new Complex128("0.39774+0.05174i"), new Complex128("0.93539+0.47693i"), new Complex128("0.86447+0.59669i"), new Complex128("0.2025+0.06263i"), new Complex128("0.01157+0.53881i"), new Complex128("0.49949+0.33721i"), new Complex128("0.24679")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
        expColIndices = new int[]{3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.95961+0.55012i"), new Complex128("0.07709+0.21971i")},
                {new Complex128("0.79181+0.14924i"), new Complex128("0.28403+0.49931i")},
                {new Complex128("0.31983+0.35642i"), new Complex128("0.17321+0.7033i")},
                {new Complex128("0.40622+0.18483i"), new Complex128("0.22014+0.08962i")},
                {new Complex128("0.63706+0.09784i"), new Complex128("0.95532+0.74443i")}};
        b = new CMatrix(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new Complex128[]{new Complex128("0.95961+0.55012i"), new Complex128("0.07709+0.21971i"), new Complex128("0.79181+0.14924i"), new Complex128("0.28403+0.49931i"), new Complex128("0.31983+0.35642i"), new Complex128("0.17321+0.7033i"), new Complex128("0.40622+0.18483i"), new Complex128("0.22014+0.08962i"), new Complex128("0.63706+0.09784i"), new Complex128("0.95532+0.74443i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.79976, 0.76613, 0.17925, 0.18628, 0.19477, 0.71918, 0.43503, 0.89331, 0.27575, 0.81762, 0.00097, 0.56518, 0.01486, 0.3944};
        aRowIndices = new int[]{0, 0, 1, 2, 2, 3, 5, 7, 7, 8, 10, 11, 11, 13};
        aColIndices = new int[]{0, 2, 3, 1, 3, 3, 4, 1, 4, 1, 1, 1, 2, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.09495+0.60496i"), new Complex128("0.91311+0.14609i"), new Complex128("0.22331+0.86692i"), new Complex128("0.34818+0.98298i"), new Complex128("0.38552+0.13002i"), new Complex128("0.24886+0.0809i")},
                {new Complex128("0.46818+0.86879i"), new Complex128("0.76817+0.0027i"), new Complex128("0.89061+0.38355i"), new Complex128("0.49315+0.0735i"), new Complex128("0.01459+0.12956i"), new Complex128("0.93247+0.28149i")},
                {new Complex128("0.90017+0.86808i"), new Complex128("0.71967+0.46991i"), new Complex128("0.68536+0.8864i"), new Complex128("0.25795+0.55715i"), new Complex128("0.89709+0.14283i"), new Complex128("0.70287+0.90793i")},
                {new Complex128("0.26908+0.84306i"), new Complex128("0.78192+0.79156i"), new Complex128("0.27281+0.70007i"), new Complex128("0.72163+0.35773i"), new Complex128("0.82102+0.34788i"), new Complex128("0.60869+0.83386i")},
                {new Complex128("0.03427+0.99641i"), new Complex128("0.21224+0.90445i"), new Complex128("0.3773+0.93432i"), new Complex128("0.27705+0.44558i"), new Complex128("0.65139+0.96358i"), new Complex128("0.56276+0.95405i")},
                {new Complex128("0.79041+0.71642i"), new Complex128("0.93389+0.25596i"), new Complex128("0.47872+0.6751i"), new Complex128("0.94296+0.40684i"), new Complex128("0.4038+0.87913i"), new Complex128("0.22385+0.98267i")},
                {new Complex128("0.70965+0.10326i"), new Complex128("0.48874+0.81958i"), new Complex128("0.14389+0.46799i"), new Complex128("0.3366+0.19123i"), new Complex128("0.65009+0.78853i"), new Complex128("0.28628+0.3122i")},
                {new Complex128("0.25582+0.6892i"), new Complex128("0.81468+0.85371i"), new Complex128("0.49466+0.71942i"), new Complex128("0.92267+0.3747i"), new Complex128("0.09491+0.09278i"), new Complex128("0.10452+0.40149i")},
                {new Complex128("0.73491+0.22062i"), new Complex128("0.46567+0.51446i"), new Complex128("0.80724+0.37621i"), new Complex128("0.09681+0.81967i"), new Complex128("0.73089+0.4304i"), new Complex128("0.75874+0.73927i")},
                {new Complex128("0.40421+0.91567i"), new Complex128("0.79238+0.33504i"), new Complex128("0.49745+0.55204i"), new Complex128("0.96481+0.59047i"), new Complex128("0.2134+0.3848i"), new Complex128("0.87603+0.62491i")},
                {new Complex128("0.82563+0.65723i"), new Complex128("0.64511+0.20912i"), new Complex128("0.79224+0.336i"), new Complex128("0.74347+0.63977i"), new Complex128("0.59219+0.61107i"), new Complex128("0.09252+0.40304i")},
                {new Complex128("0.79418+0.12344i"), new Complex128("0.38493+0.26857i"), new Complex128("0.8192+0.60589i"), new Complex128("0.89572+0.1365i"), new Complex128("0.40387+0.25125i"), new Complex128("0.89349+0.45239i")},
                {new Complex128("0.67492+0.40309i"), new Complex128("0.18231+0.28143i"), new Complex128("0.19703+0.35099i"), new Complex128("0.54147+0.4359i"), new Complex128("0.50925+0.76791i"), new Complex128("0.40236+0.67426i")},
                {new Complex128("0.88436+0.42101i"), new Complex128("0.75941+0.17821i"), new Complex128("0.78757+0.0649i"), new Complex128("0.83376+0.69951i"), new Complex128("0.57271+0.0057i"), new Complex128("0.8819+0.38706i")}};
        b = new CMatrix(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new Complex128[]{new Complex128("0.09495+0.60496i"), new Complex128("0.91311+0.14609i"), new Complex128("0.22331+0.86692i"), new Complex128("0.34818+0.98298i"), new Complex128("0.38552+0.13002i"), new Complex128("0.24886+0.0809i"), new Complex128("0.46818+0.86879i"), new Complex128("0.76817+0.0027i"), new Complex128("0.89061+0.38355i"), new Complex128("0.49315+0.0735i"), new Complex128("0.01459+0.12956i"), new Complex128("0.93247+0.28149i"), new Complex128("0.90017+0.86808i"), new Complex128("0.71967+0.46991i"), new Complex128("0.68536+0.8864i"), new Complex128("0.25795+0.55715i"), new Complex128("0.89709+0.14283i"), new Complex128("0.70287+0.90793i"), new Complex128("0.26908+0.84306i"), new Complex128("0.78192+0.79156i"), new Complex128("0.27281+0.70007i"), new Complex128("0.72163+0.35773i"), new Complex128("0.82102+0.34788i"), new Complex128("0.60869+0.83386i"), new Complex128("0.03427+0.99641i"), new Complex128("0.21224+0.90445i"), new Complex128("0.3773+0.93432i"), new Complex128("0.27705+0.44558i"), new Complex128("0.65139+0.96358i"), new Complex128("0.56276+0.95405i"), new Complex128("0.79041+0.71642i"), new Complex128("0.93389+0.25596i"), new Complex128("0.47872+0.6751i"), new Complex128("0.94296+0.40684i"), new Complex128("0.4038+0.87913i"), new Complex128("0.22385+0.98267i"), new Complex128("0.70965+0.10326i"), new Complex128("0.48874+0.81958i"), new Complex128("0.14389+0.46799i"), new Complex128("0.3366+0.19123i"), new Complex128("0.65009+0.78853i"), new Complex128("0.28628+0.3122i"), new Complex128("0.25582+0.6892i"), new Complex128("0.81468+0.85371i"), new Complex128("0.49466+0.71942i"), new Complex128("0.92267+0.3747i"), new Complex128("0.09491+0.09278i"), new Complex128("0.10452+0.40149i"), new Complex128("0.73491+0.22062i"), new Complex128("0.46567+0.51446i"), new Complex128("0.80724+0.37621i"), new Complex128("0.09681+0.81967i"), new Complex128("0.73089+0.4304i"), new Complex128("0.75874+0.73927i"), new Complex128("0.40421+0.91567i"), new Complex128("0.79238+0.33504i"), new Complex128("0.49745+0.55204i"), new Complex128("0.96481+0.59047i"), new Complex128("0.2134+0.3848i"), new Complex128("0.87603+0.62491i"), new Complex128("0.82563+0.65723i"), new Complex128("0.64511+0.20912i"), new Complex128("0.79224+0.336i"), new Complex128("0.74347+0.63977i"), new Complex128("0.59219+0.61107i"), new Complex128("0.09252+0.40304i"), new Complex128("0.79418+0.12344i"), new Complex128("0.38493+0.26857i"), new Complex128("0.8192+0.60589i"), new Complex128("0.89572+0.1365i"), new Complex128("0.40387+0.25125i"), new Complex128("0.89349+0.45239i"), new Complex128("0.67492+0.40309i"), new Complex128("0.18231+0.28143i"), new Complex128("0.19703+0.35099i"), new Complex128("0.54147+0.4359i"), new Complex128("0.50925+0.76791i"), new Complex128("0.40236+0.67426i"), new Complex128("0.88436+0.42101i"), new Complex128("0.75941+0.17821i"), new Complex128("0.78757+0.0649i"), new Complex128("0.83376+0.69951i"), new Complex128("0.57271+0.0057i"), new Complex128("0.8819+0.38706i"), new Complex128("0.79976"), new Complex128("0.76613"), new Complex128("0.17925"), new Complex128("0.18628"), new Complex128("0.19477"), new Complex128("0.71918"), new Complex128("0.43503"), new Complex128("0.89331"), new Complex128("0.27575"), new Complex128("0.81762"), new Complex128("0.00097"), new Complex128("0.56518"), new Complex128("0.01486"), new Complex128("0.3944")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 15, 16, 16, 17, 19, 21, 21, 22, 24, 25, 25, 27};
        expColIndices = new int[]{5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 0, 2, 3, 1, 3, 3, 4, 1, 4, 1, 1, 1, 2, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }
}
