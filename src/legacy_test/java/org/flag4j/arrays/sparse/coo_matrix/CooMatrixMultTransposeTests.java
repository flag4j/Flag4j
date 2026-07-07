package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CooMatrixMultTransposeTests {

    @Test
    void realSparseMultTransposeTest() {
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
        aEntries = new double[]{0.78156, 0.09594, 0.7923};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6, 5);
        bEntries = new double[]{0.46839, 0.47218, 0.85592, 0.41846, 0.03665, 0.40249, 0.39273, 0.71011, 0.50029, 0.19742};
        bRowIndices = new int[]{0, 1, 2, 2, 3, 3, 3, 3, 4, 4};
        bColIndices = new int[]{3, 4, 1, 2, 0, 2, 3, 4, 3, 4};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0, 0.0, 0.6689528352, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.003516201, 0.0, 0.0},
                {0.0, 0.0, 0.678145416, 0.0, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.99702, 0.04209, 0.43944, 0.33732, 0.37757, 0.05866, 0.89726, 0.68715, 0.32244, 0.352, 0.47304, 0.41871, 0.49412, 0.88239, 0.77977};
        aRowIndices = new int[]{0, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 10};
        aColIndices = new int[]{21, 4, 18, 10, 12, 15, 12, 18, 19, 20, 9, 15, 3, 10, 9};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11, 23);
        bEntries = new double[]{0.43453, 0.96681, 0.53232, 0.42268, 0.55315, 0.32311, 0.36187, 0.41291, 0.05022, 0.31466, 0.16399, 0.38665, 0.04099, 0.55214, 0.09856, 0.16123, 0.07109, 0.18844, 0.68079, 0.44251, 0.48795, 0.90615, 0.27059, 0.91353, 0.16297, 0.83766, 0.98706, 0.71687, 0.78636, 0.15918, 0.69246, 0.3795, 0.39076, 0.00326, 0.21866, 0.08403, 0.53308, 0.79918, 0.44156, 0.58684, 0.62729, 0.00474, 0.94979, 0.65794, 0.63977, 0.95383, 0.87742, 0.40367, 0.61562, 0.16512, 0.81519, 0.5108, 0.45016, 0.28453, 0.65645, 0.45985, 0.95643, 0.23393, 0.38601, 0.58675, 0.32708, 0.04307, 0.35009, 0.34327, 0.11326, 0.35421, 0.86255, 0.37221, 0.07691, 0.03179, 0.66814, 0.38309, 0.9688, 0.3194, 0.99001, 0.57288, 0.08684, 0.84867, 0.19332, 0.61244, 0.16635, 0.38489, 0.61532, 0.96326, 0.48203, 0.18518, 0.44577, 0.78922, 0.85329};
        bRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10};
        bColIndices = new int[]{0, 5, 6, 8, 14, 19, 21, 5, 7, 8, 17, 18, 19, 20, 0, 1, 4, 5, 10, 14, 15, 20, 21, 9, 10, 16, 21, 0, 9, 10, 19, 20, 22, 0, 3, 4, 6, 7, 13, 15, 20, 21, 1, 4, 5, 6, 9, 11, 13, 14, 21, 22, 1, 3, 5, 7, 14, 15, 18, 21, 0, 1, 6, 9, 10, 11, 15, 16, 21, 0, 2, 3, 7, 12, 13, 14, 15, 18, 19, 20, 21, 22, 1, 8, 13, 14, 15, 20, 21};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.36079162740000004, 0.0, 0.26978364180000003, 0.9841185612000001, 0.0, 0.0047258748, 0.8127607338, 0.585001485, 0.0766808082, 0.165854277, 0.8507471958},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.169909476, 0.0029921781000000003, 0.0, 0.0, 0.0035368227, 0.0276926946, 0.1696282344, 0.0, 0.3729395448, 0.0},
                {0.0, 0.0, 0.2296440828, 0.054973040400000006, 0.0536945976, 0.0, 0.0, 0.0, 0.0382048632, 0.0, 0.0},
                {0.0, 0.0, 0.028623146999999998, 0.0, 0.0, 0.0344240344, 0.0, 0.0137223338, 0.050597183, 0.1256898924, 0.026148868199999998},
                {0.1041835884, 0.47325664310000004, 0.3189648, 0.0, 0.35686080239999995, 0.22080608, 0.0, 0.26524677150000003, 0.0, 1.1476614153000002, 0.27780544},
                {0.0, 0.0, 0.20430954450000002, 0.4321362312, 0.3719797344, 0.24571577640000003, 0.4150547568, 0.09794883030000001, 0.5235387513, 0.0363607764, 0.18664835670000002},
                {0.0, 0.0, 0.6007222881000001, 0.14380309830000002, 0.1404588402, 0.1080442792, 0.0, 0.1405919636, 0.0999394914, 0.1892924308, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.7123432880999999, 0.6131799372, 0.0, 0.6841857934, 0.0, 0.2676716479, 0.0, 0.0}
        };
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.6657, 0.789, 0.34576, 0.67106};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{0, 0, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.40693};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0},
                {0.0},
                {0.0},
                {0.0},
                {0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.63359, 0.98973, 0.65753, 0.27274};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{2, 1, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.25708, 0.02006};
        bRowIndices = new int[]{0, 0};
        bColIndices = new int[]{0, 2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0, 0.0},
                {0.012709815400000001, 0.0},
                {0.0, 0.0},
                {0.16903781239999996, 0.0},
                {0.07011599919999999, 0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.43181, 0.80241, 0.7987, 0.961};
        aRowIndices = new int[]{1, 2, 4, 4};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.79177, 0.7031, 0.58915, 0.31236};
        bRowIndices = new int[]{0, 1, 3, 4};
        bColIndices = new int[]{1, 0, 1, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final0a = a;
        CooMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.28394, 0.65788, 0.63941, 0.47642};
        aRowIndices = new int[]{0, 2, 4, 4};
        aColIndices = new int[]{1, 2, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.23905, 0.29491, 0.73915, 0.58077, 0.92534, 0.0505, 0.64781, 0.49145, 0.58197};
        bRowIndices = new int[]{0, 0, 0, 1, 2, 2, 2, 3, 4};
        bColIndices = new int[]{0, 2, 4, 2, 1, 2, 3, 2, 4};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final1a = a;
        CooMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }
}
