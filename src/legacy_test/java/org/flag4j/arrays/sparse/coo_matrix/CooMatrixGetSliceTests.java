package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixGetSliceTests {

    @Test
    void getSliceTest() {
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
        aEntries = new double[]{0.57864, 0.29143, 0.6719};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{0, 3, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(2, 5);
        expEntries = new double[]{0.57864, 0.29143, 0.6719};
        expRowIndices = new int[]{0, 0, 1};
        expColIndices = new int[]{0, 3, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.getSlice(1, 3, 0, 5));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.40224, 0.71072, 0.02414, 0.08009, 0.77722};
        aRowIndices = new int[]{2, 3, 4, 6, 6};
        aColIndices = new int[]{1, 14, 18, 5, 13};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(6, 2);
        expEntries = new double[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.getSlice(2, 8, 9, 11));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.11039, 0.08607, 0.1597, 0.96325, 0.78209, 0.55771, 0.60035, 0.81156, 0.53868, 0.72552, 0.11207, 0.77063, 0.47979, 0.5244, 0.64588, 0.83039, 0.37721, 0.59008, 0.30295, 0.79798, 0.66421, 0.59238, 0.22446, 0.17068, 0.65914, 0.10437, 0.30251, 0.71144, 0.56038, 0.39275, 0.59353, 0.05126, 0.95321, 0.10937, 0.83926, 0.96745, 0.52445, 0.04341, 0.09122, 0.54406, 0.78048, 0.29304, 0.29885, 0.58542, 0.4731, 0.39942, 0.71464, 0.93901, 0.68053, 0.25877, 0.34548, 0.38683, 0.49397, 0.55366, 0.82593, 0.36953, 0.51132, 0.36827, 0.86507, 0.40292, 0.64449, 0.83377, 0.4911, 0.56302, 0.21556, 0.09014, 0.21245, 0.48857, 0.06146, 0.67002, 0.29456, 0.76681, 0.54477, 0.29906, 0.97646, 0.40446, 0.40303, 0.59803, 0.58887, 0.99502, 0.43677, 0.58847, 0.5435, 0.19821, 0.25935, 0.6456, 0.81696, 0.77924, 0.33159, 0.06404, 0.21883, 0.69595, 0.04291, 0.61922, 0.10557, 0.94247, 0.10053, 0.6303};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{188, 234, 305, 443, 483, 527, 549, 600, 630, 651, 796, 12, 28, 36, 82, 99, 122, 246, 249, 285, 404, 419, 457, 461, 498, 533, 550, 609, 614, 700, 783, 893, 945, 2, 53, 66, 186, 270, 284, 377, 419, 422, 459, 475, 546, 670, 695, 718, 745, 810, 816, 822, 829, 860, 867, 874, 907, 987, 71, 88, 99, 108, 343, 412, 428, 447, 458, 479, 537, 659, 679, 784, 902, 930, 32, 82, 94, 104, 199, 202, 299, 338, 348, 352, 380, 431, 441, 553, 570, 583, 600, 715, 737, 749, 805, 948, 967, 995};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 37);
        expEntries = new double[]{0.64588, 0.83039};
        expRowIndices = new int[]{1, 1};
        expColIndices = new int[]{10, 27};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.getSlice(0, 3, 72, 109));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.18862, 0.44481, 0.74362, 0.27797};
        aRowIndices = new int[]{3, 3, 4, 4};
        aColIndices = new int[]{0, 1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getSlice(-1, 2, 1, 2));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.41805, 0.46165, 0.8042, 0.77989};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{1, 2, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getSlice(0, 1, -1, 2));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.79502, 0.83795, 0.93046, 0.74467};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{0, 1, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.getSlice(0, 6, 0, 1));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.24533, 0.32357, 0.63084, 0.62244};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{0, 2, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.getSlice(0, 2, 0, 40));
    }

}
