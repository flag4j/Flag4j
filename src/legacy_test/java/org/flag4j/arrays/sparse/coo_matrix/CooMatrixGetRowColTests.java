package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.sparse.coo.real.RealCooMatrixGetSet;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixGetRowColTests {


    @Test
    void getRowTest() {
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
        aShape = new Shape(5, 3);
        aEntries = new double[]{1.0, 0.49936, 0.68478};
        aRowIndices = new int[]{2, 4, 4};
        aColIndices = new int[]{2, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{1.0};
        expIndices = new int[]{2};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.36666, 0.93312, 0.07692, 0.98456, 0.22229};
        aRowIndices = new int[]{0, 13, 19, 21, 22};
        aColIndices = new int[]{8, 7, 6, 6, 6};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(18));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.11286, 0.53266, 0.53358, 0.41511, 0.70554, 0.21056, 0.34396, 0.99792, 0.30181, 0.92438, 0.1297, 0.58991, 0.11402, 0.14567, 0.77321, 0.81184, 0.56986, 0.29824, 0.67377, 0.67568, 0.4824, 0.1568, 0.37857, 0.95562, 0.71837, 0.84369, 0.97882, 0.96502, 0.26741, 0.22641, 0.20198, 0.70493, 0.07416, 0.36901, 0.77372, 0.55024, 0.16722, 0.84518, 0.52659, 0.85514, 0.62962, 0.67955, 0.54798, 0.73438, 0.97605, 0.47371, 0.91858, 0.06186, 0.12977, 0.47371, 0.44096, 0.78256, 0.94133, 0.4721, 0.82157, 0.65828, 0.58433, 0.60539, 0.14044, 0.40646, 0.85086, 0.38847, 0.89009, 0.64733, 0.59814, 0.58499, 0.24108, 0.39847, 0.51196, 0.72189, 0.85715, 0.60302, 0.17197, 0.46277, 0.95027, 0.99916, 0.0679, 0.26619, 0.56531, 0.56737, 0.23521, 0.87058, 0.35723, 0.48772, 0.63005, 0.45435, 0.46076, 0.0068, 0.78308, 0.89293, 0.45885, 0.46711, 0.34827, 0.33707, 0.725, 0.42075, 0.66232, 0.22166};
        aRowIndices = new int[]{11, 40, 55, 58, 60, 72, 72, 92, 96, 97, 108, 116, 121, 159, 163, 177, 183, 198, 210, 219, 221, 222, 232, 236, 250, 272, 272, 276, 280, 282, 301, 316, 318, 323, 356, 358, 366, 379, 397, 403, 419, 447, 466, 486, 499, 514, 516, 527, 548, 562, 562, 563, 577, 581, 582, 590, 614, 625, 637, 665, 671, 679, 683, 684, 690, 692, 693, 696, 700, 715, 727, 728, 739, 754, 759, 759, 764, 765, 777, 778, 784, 790, 791, 820, 824, 831, 849, 862, 869, 887, 896, 918, 923, 935, 940, 943, 973, 983};
        aColIndices = new int[]{4, 1, 1, 3, 0, 0, 2, 1, 2, 4, 3, 1, 2, 4, 0, 2, 3, 1, 1, 2, 0, 1, 0, 1, 0, 2, 3, 0, 4, 4, 4, 1, 1, 4, 3, 3, 2, 4, 4, 2, 1, 3, 0, 3, 3, 1, 1, 1, 1, 0, 1, 3, 4, 0, 4, 1, 2, 0, 2, 1, 4, 1, 4, 1, 0, 1, 3, 3, 1, 1, 2, 4, 0, 4, 0, 3, 2, 3, 4, 2, 4, 3, 3, 4, 0, 4, 4, 0, 4, 0, 2, 2, 4, 3, 0, 1, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.80208, 0.06677, 0.98614, 0.23957};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{2, 2, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getRow(-1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.10266, 0.45464, 0.96301, 0.53585};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{2, 0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getRow(3));
    }


    @Test
    void getRowSliceTest() {
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
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.54329, 0.11495, 0.58252};
        aRowIndices = new int[]{1, 1, 4};
        aColIndices = new int[]{1, 2, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(2);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, RealCooMatrixGetSet.getRow(a, 2, 1, 3));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.49229, 0.87251, 0.78602, 0.37743, 0.53322};
        aRowIndices = new int[]{4, 4, 5, 13, 14};
        aColIndices = new int[]{7, 8, 8, 8, 8};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(7);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, RealCooMatrixGetSet.getRow(a, 18, 0, 7));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.08045, 0.15028, 0.87921, 0.39041, 0.74299, 0.34942, 0.84353, 0.66429, 0.58565, 0.81405, 0.41159, 0.00734, 0.21917, 0.34396, 0.81508, 0.63815, 0.29402, 0.24424, 0.10666, 0.46142, 0.3713, 0.91604, 0.65281, 0.36722, 0.59828, 0.63998, 0.24231, 0.78107, 0.90565, 0.98953, 0.15044, 0.92787, 0.24013, 0.87921, 0.82568, 0.73436, 0.29501, 0.70664, 0.16402, 0.70169, 0.24982, 0.52572, 0.69861, 0.60027, 0.09263, 0.95337, 0.59764, 0.49835, 0.84475, 0.13502, 0.73124, 0.21129, 0.24911, 0.3989, 0.81239, 0.72483, 0.72416, 0.20215, 0.27749, 0.32916, 0.56992, 0.43389, 0.09049, 0.45894, 0.63919, 0.18407, 0.09646, 0.11269, 0.58566, 0.12624, 0.48882, 0.73737, 0.15487, 0.18531, 0.75527, 0.25211, 0.6769, 0.21024, 0.1682, 0.24659, 0.76189, 0.83512, 0.46363, 0.34829, 0.84391, 0.49122, 0.22278, 0.35609, 0.40248, 0.09336, 0.70193, 0.82112, 0.43694, 0.46546, 0.36865, 0.11906, 0.02577, 0.32453};
        aRowIndices = new int[]{0, 14, 22, 23, 36, 40, 41, 45, 50, 52, 54, 55, 64, 75, 96, 108, 132, 141, 158, 170, 177, 179, 188, 193, 194, 199, 202, 203, 208, 209, 213, 213, 221, 231, 245, 248, 254, 272, 280, 282, 308, 320, 330, 335, 343, 347, 384, 393, 410, 413, 418, 440, 442, 443, 446, 460, 465, 468, 483, 524, 528, 539, 540, 581, 615, 656, 660, 669, 679, 696, 704, 705, 712, 716, 719, 731, 745, 753, 754, 776, 777, 795, 797, 820, 835, 848, 851, 853, 871, 885, 918, 949, 953, 967, 968, 986, 994, 995};
        aColIndices = new int[]{4, 1, 2, 1, 0, 0, 2, 0, 4, 3, 0, 3, 2, 4, 1, 4, 1, 1, 1, 3, 0, 4, 4, 4, 0, 3, 1, 3, 0, 4, 0, 3, 1, 2, 2, 2, 3, 0, 3, 1, 1, 4, 1, 4, 1, 1, 3, 1, 0, 2, 0, 0, 3, 3, 3, 2, 4, 1, 3, 2, 3, 2, 3, 1, 4, 1, 0, 0, 4, 4, 2, 0, 4, 2, 2, 4, 4, 2, 3, 3, 3, 4, 3, 3, 4, 0, 0, 1, 0, 1, 3, 3, 3, 3, 4, 2, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, RealCooMatrixGetSet.getRow(a, 0, 1, 4));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.13855, 0.89783, 0.1084, 0.44505};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{4, 0, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final0a = a;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.getRow(final0a, -1, 1, 3));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.78898, 0.58061, 0.32456, 0.88779};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{2, 4, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final1a = a;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.getRow(final1a, 3, 1, 3));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.73121, 0.81367, 0.83744, 0.04126};
        aRowIndices = new int[]{1, 1, 1, 2};
        aColIndices = new int[]{2, 3, 4, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final2a = a;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.getRow(final2a, 2, -1, 3));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.11283, 0.96564, 0.22042, 0.27551};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{1, 2, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final3a = a;
        assertThrows(Exception.class, ()-> RealCooMatrixGetSet.getRow(final3a, 2, 1, 6));
    }


    @Test
    void getColTest() {
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
        aEntries = new double[]{0.82217, 0.54326, 0.7119};
        aRowIndices = new int[]{0, 2, 2};
        aColIndices = new int[]{0, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.49685, 0.29949, 0.45767, 0.65021, 0.86053};
        aRowIndices = new int[]{2, 3, 6, 10, 10};
        aColIndices = new int[]{13, 17, 13, 8, 13};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(18));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.61353, 0.45236, 0.98732, 0.66187, 0.18643, 0.4535, 0.61496, 0.84772, 0.92487, 0.65438, 0.06026, 0.91128, 0.51471, 0.09689, 0.71218, 0.24603, 0.35096, 0.61809, 0.70675, 0.76454, 0.07659, 0.56006, 0.61405, 0.83066, 0.76712, 0.98462, 0.58847, 0.35396, 0.305, 0.30769, 0.0031, 0.51223, 0.87145, 0.03676, 0.77447, 0.22208, 0.55149, 0.38503, 0.8805, 0.24426, 0.58108, 0.22275, 0.43373, 0.08354, 0.64883, 0.82157, 0.20174, 0.53521, 0.56324, 0.82049, 0.17812, 0.49506, 0.9431, 0.43849, 0.24048, 0.8503, 0.47013, 0.98455, 0.66708, 0.43895, 0.80218, 0.15376, 0.46876, 0.61946, 0.7082, 0.42247, 0.8287, 0.06949, 0.95149, 0.6212, 0.41053, 0.1709, 0.30086, 0.35668, 0.56014, 0.52698, 0.07533, 0.61173, 0.55307, 0.91841, 0.02965, 0.51771, 0.52122, 0.64732, 0.04325, 0.86323, 0.44958, 0.32653, 0.91803, 0.55382, 0.61067, 0.27746, 0.32455, 0.97932, 0.4557, 0.18606, 0.71778, 0.3769};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{82, 104, 109, 164, 204, 247, 254, 272, 315, 322, 380, 387, 420, 460, 645, 705, 744, 783, 795, 803, 863, 874, 33, 93, 123, 152, 189, 424, 436, 438, 452, 571, 600, 645, 681, 698, 722, 734, 841, 858, 905, 936, 7, 101, 142, 199, 209, 263, 267, 270, 313, 354, 397, 443, 446, 503, 549, 552, 577, 582, 648, 718, 737, 872, 893, 956, 78, 141, 212, 276, 290, 307, 423, 487, 554, 592, 615, 707, 744, 845, 851, 916, 925, 962, 982, 6, 52, 111, 124, 275, 362, 427, 561, 644, 750, 779, 884, 978};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new double[]{};
        expIndices = new int[]{};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.26113, 0.02393, 0.97081, 0.81325};
        aRowIndices = new int[]{0, 2, 2, 4};
        aColIndices = new int[]{2, 1, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getCol(-1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.54996, 0.55301, 0.20074, 0.14401};
        aRowIndices = new int[]{1, 3, 3, 4};
        aColIndices = new int[]{0, 1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getCol(3));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.18116, 0.68854, 0.5203, 0.62009};
        aRowIndices = new int[]{1, 1, 4, 4};
        aColIndices = new int[]{0, 1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new double[]{0.62009};
        expIndices = new int[]{4};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(2));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.80851, 0.29534, 0.97174, 0.75706};
        aRowIndices = new int[]{3, 3, 4, 4};
        aColIndices = new int[]{1, 2, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new double[]{0.29534, 0.75706};
        expIndices = new int[]{3, 4};
        exp = new CooVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(2));
    }
}
