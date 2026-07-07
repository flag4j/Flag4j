package org.flag4j.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixRemoveRowColTests {

    Shape expShape, actShape;
    Complex128[] expData, actData;
    int[] expRowIndices, actRowIndices;
    int[] expColIndices, actColIndices;
    CooCMatrix exp, act;

    @Test
    void removeRowTests() {
        // ------------------ sub-case 1 ------------------
        actShape = new Shape(12, 45);
        actData = new Complex128[]{new Complex128(0.77866, 0.69048), new Complex128(0.76284, 0.7625), new Complex128(0.50343, 0.5486), new Complex128(0.94497, 0.41532), new Complex128(0.53088, 0.74234)};
        actRowIndices = new int[]{1, 3, 5, 7, 11};
        actColIndices = new int[]{13, 38, 3, 33, 3};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(11, 45);
        expData = new Complex128[]{new Complex128(0.77866, 0.69048), new Complex128(0.76284, 0.7625), new Complex128(0.50343, 0.5486), new Complex128(0.94497, 0.41532), new Complex128(0.53088, 0.74234)};
        expRowIndices = new int[]{0, 2, 4, 6, 10};
        expColIndices = new int[]{13, 38, 3, 33, 3};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRow(0));

        // ------------------ sub-case 2 ------------------
        actShape = new Shape(12, 45);
        actData = new Complex128[]{new Complex128(0.71598, 0.48605), new Complex128(0.50074, 0.4636), new Complex128(0.4353, 0.48429), new Complex128(0.26224, 0.1529), new Complex128(0.65118, 0.59656)};
        actRowIndices = new int[]{1, 4, 9, 10, 11};
        actColIndices = new int[]{25, 2, 16, 41, 10};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(11, 45);
        expData = new Complex128[]{new Complex128(0.71598, 0.48605), new Complex128(0.4353, 0.48429), new Complex128(0.26224, 0.1529), new Complex128(0.65118, 0.59656)};
        expRowIndices = new int[]{1, 8, 9, 10};
        expColIndices = new int[]{25, 16, 41, 10};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRow(4));

        // ------------------ sub-case 3 ------------------
        actShape = new Shape(12, 45);
        actData = new Complex128[]{new Complex128(0.64416, 0.53894), new Complex128(0.80314, 0.25352), new Complex128(0.1104, 0.96916), new Complex128(0.34511, 0.53383), new Complex128(0.44366, 0.67431)};
        actRowIndices = new int[]{1, 3, 4, 8, 11};
        actColIndices = new int[]{3, 24, 14, 27, 34};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(11, 45);
        expData = new Complex128[]{new Complex128(0.64416, 0.53894), new Complex128(0.80314, 0.25352), new Complex128(0.1104, 0.96916), new Complex128(0.34511, 0.53383)};
        expRowIndices = new int[]{1, 3, 4, 8};
        expColIndices = new int[]{3, 24, 14, 27};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRow(11));

        // ------------------ sub-case 4 ------------------
        actShape = new Shape(12, 45);
        actData = new Complex128[]{};
        actRowIndices = new int[]{};
        actColIndices = new int[]{};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(11, 45);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRow(2));

        // ------------------ sub-case 5 ------------------
        act = new CooCMatrix(new Shape(3, 9));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeRow(4));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeRow(-1));
    }

    @Test
    void removeRowsTests() {
        // ------------------ sub-case 1 ------------------
        actShape = new Shape(12, 45);
        actData = new Complex128[]{new Complex128(0.69978, 0.68298), new Complex128(0.73937, 0.09706), new Complex128(0.89267, 0.05119), new Complex128(0.57729, 0.10399), new Complex128(0.55045, 0.73243)};
        actRowIndices = new int[]{1, 1, 8, 9, 11};
        actColIndices = new int[]{15, 23, 13, 25, 18};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(9, 45);
        expData = new Complex128[]{new Complex128(0.69978, 0.68298), new Complex128(0.73937, 0.09706), new Complex128(0.89267, 0.05119), new Complex128(0.57729, 0.10399), new Complex128(0.55045, 0.73243)};
        expRowIndices = new int[]{0, 0, 5, 6, 8};
        expColIndices = new int[]{15, 23, 13, 25, 18};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRows(0, 3, 4));

        // ------------------ sub-case 2 ------------------
        actShape = new Shape(15, 12);
        actData = new Complex128[]{new Complex128(0.47902, 0.11469), new Complex128(0.4156, 0.84995), new Complex128(0.01331, 0.61483), new Complex128(0.84897, 0.45753), new Complex128(0.89824, 0.53618), new Complex128(0.34686, 0.98037), new Complex128(0.50154, 0.17636), new Complex128(0.47826, 0.48458), new Complex128(0.63824, 0.79207), new Complex128(0.12708, 0.28845), new Complex128(0.20776, 0.21964)};
        actRowIndices = new int[]{0, 0, 1, 1, 1, 5, 6, 7, 8, 13, 13};
        actColIndices = new int[]{2, 10, 4, 5, 9, 6, 8, 10, 2, 1, 9};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(12, 12);
        expData = new Complex128[]{new Complex128(0.01331, 0.61483), new Complex128(0.84897, 0.45753), new Complex128(0.89824, 0.53618), new Complex128(0.34686, 0.98037), new Complex128(0.50154, 0.17636), new Complex128(0.47826, 0.48458), new Complex128(0.63824, 0.79207), new Complex128(0.12708, 0.28845), new Complex128(0.20776, 0.21964)};
        expRowIndices = new int[]{0, 0, 0, 2, 3, 4, 5, 10, 10};
        expColIndices = new int[]{4, 5, 9, 6, 8, 10, 2, 1, 9};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRows(4, 0, 3));

        // ------------------ sub-case 3 ------------------
        actShape = new Shape(15, 12);
        actData = new Complex128[]{};
        actRowIndices = new int[]{};
        actColIndices = new int[]{};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(12, 12);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeRows(2, 5, 9));

        // ------------------ sub-case 4 ------------------
        act = new CooCMatrix(new Shape(3, 9));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeRows(2, 5, 9));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeRows(0, -1));
    }


    @Test
    void removeColTests() {
        // ------------------ sub-case 1 ------------------
        actShape = new Shape(15, 24);
        actData = new Complex128[]{new Complex128(0.50744, 0.22408), new Complex128(0.03128, 0.09434), new Complex128(0.67216, 0.71928), new Complex128(0.26396, 0.84781), new Complex128(0.74211, 0.78747), new Complex128(0.53803, 0.80745), new Complex128(0.45852, 0.08599)};
        actRowIndices = new int[]{0, 3, 3, 5, 13, 14, 14};
        actColIndices = new int[]{7, 7, 22, 9, 10, 0, 1};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(15, 23);
        expData = new Complex128[]{new Complex128(0.50744, 0.22408), new Complex128(0.03128, 0.09434), new Complex128(0.67216, 0.71928), new Complex128(0.26396, 0.84781), new Complex128(0.74211, 0.78747), new Complex128(0.45852, 0.08599)};
        expRowIndices = new int[]{0, 3, 3, 5, 13, 14};
        expColIndices = new int[]{6, 6, 21, 8, 9, 0};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCol(0));

        // ------------------ sub-case 2 ------------------
        actShape = new Shape(24, 15);
        actData = new Complex128[]{new Complex128(0.66056, 0.75806), new Complex128(0.82459, 0.65965), new Complex128(0.63419, 0.16823), new Complex128(0.74155, 0.35718), new Complex128(0.48856, 0.71093), new Complex128(0.3763, 0.7764), new Complex128(0.66567, 0.19264)};
        actRowIndices = new int[]{0, 7, 7, 13, 15, 18, 19};
        actColIndices = new int[]{9, 2, 13, 14, 0, 0, 8};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(24, 14);
        expData = new Complex128[]{new Complex128(0.66056, 0.75806), new Complex128(0.82459, 0.65965), new Complex128(0.63419, 0.16823), new Complex128(0.74155, 0.35718), new Complex128(0.66567, 0.19264)};
        expRowIndices = new int[]{0, 7, 7, 13, 19};
        expColIndices = new int[]{8, 1, 12, 13, 7};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCol(0));

        // ------------------ sub-case 3 ------------------
        actShape = new Shape(24, 15);
        actData = new Complex128[]{new Complex128(0.85192, 0.11152), new Complex128(0.46917, 0.82861), new Complex128(0.62357, 0.56506), new Complex128(0.72122, 0.04086), new Complex128(0.17467, 0.97519), new Complex128(0.93648, 0.79936), new Complex128(0.19383, 0.26116), new Complex128(0.35744, 0.12276), new Complex128(0.714, 0.69116), new Complex128(0.26821, 0.17701), new Complex128(0.306, 0.75897), new Complex128(0.50037, 0.63035)};
        actRowIndices = new int[]{1, 1, 4, 5, 12, 15, 16, 17, 17, 18, 21, 22};
        actColIndices = new int[]{1, 8, 11, 8, 2, 2, 4, 11, 14, 1, 1, 4};

        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);
        expShape = new Shape(24, 14);
        expData = new Complex128[]{new Complex128(0.85192, 0.11152), new Complex128(0.46917, 0.82861), new Complex128(0.62357, 0.56506), new Complex128(0.72122, 0.04086), new Complex128(0.17467, 0.97519), new Complex128(0.93648, 0.79936), new Complex128(0.35744, 0.12276), new Complex128(0.714, 0.69116), new Complex128(0.26821, 0.17701), new Complex128(0.306, 0.75897)};
        expRowIndices = new int[]{1, 1, 4, 5, 12, 15, 17, 17, 18, 21};
        expColIndices = new int[]{1, 7, 10, 7, 2, 2, 10, 13, 1, 1};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCol(4));

        // ------------------ sub-case 4 ------------------
        actShape = new Shape(24, 15);
        actData = new Complex128[]{new Complex128(0.77821, 0.00128), new Complex128(0.2461, 0.13312), new Complex128(0.97996, 0.751), new Complex128(0.58622, 0.18542), new Complex128(0.32885, 0.45088), new Complex128(0.24512, 0.82565), new Complex128(0.08009, 0.86475), new Complex128(0.0346, 0.32301), new Complex128(0.68427, 0.36554), new Complex128(0.07324, 0.88487), new Complex128(0.54839, 0.4006), new Complex128(0.1038, 0.73258)};
        actRowIndices = new int[]{0, 2, 4, 5, 7, 8, 8, 20, 20, 21, 22, 22};
        actColIndices = new int[]{12, 0, 5, 12, 14, 9, 12, 8, 12, 11, 3, 9};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(24, 14);
        expData = new Complex128[]{new Complex128(0.77821, 0.00128), new Complex128(0.2461, 0.13312), new Complex128(0.97996, 0.751), new Complex128(0.58622, 0.18542), new Complex128(0.24512, 0.82565), new Complex128(0.08009, 0.86475), new Complex128(0.0346, 0.32301), new Complex128(0.68427, 0.36554), new Complex128(0.07324, 0.88487), new Complex128(0.54839, 0.4006), new Complex128(0.1038, 0.73258)};
        expRowIndices = new int[]{0, 2, 4, 5, 8, 8, 20, 20, 21, 22, 22};
        expColIndices = new int[]{12, 0, 5, 12, 9, 12, 8, 12, 11, 3, 9};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCol(14));

        // ------------------ sub-case 5 ------------------
        actShape = new Shape(15, 24);
        actData = new Complex128[]{new Complex128(0.50744, 0.22408),
                new Complex128(0.03128, 0.09434),
                new Complex128(0.67216, 0.71928),
                new Complex128(0.26396, 0.84781),
                new Complex128(0.74211, 0.78747),
                new Complex128(0.53803, 0.80745),
                new Complex128(0.45852, 0.08599)};
        actRowIndices = new int[]{0, 1, 2, 3, 5, 11, 13};
        actColIndices = new int[]{5, 5, 5, 5, 5, 5, 5};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(15, 23);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCol(5));

        // ------------------ sub-case 6 ------------------
        act = new CooCMatrix(new Shape(3, 9));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeCol(14));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeRow(-1));
    }


    @Test
    void removeColsTests() {
        // ------------------ sub-case 1 ------------------
        actShape = new Shape(24, 15);
        actData = new Complex128[]{new Complex128(0.34666, 0.12705), new Complex128(0.0907, 0.15388), new Complex128(0.95514, 0.67512), new Complex128(0.84671, 0.65075), new Complex128(0.57485, 0.02965), new Complex128(0.00081, 0.82227), new Complex128(0.94886, 0.3042), new Complex128(0.03307, 0.24848), new Complex128(0.82155, 0.00032), new Complex128(0.63371, 0.2528), new Complex128(0.01639, 0.46448), new Complex128(0.3547, 0.83087)};
        actRowIndices = new int[]{0, 0, 4, 6, 9, 9, 10, 14, 15, 16, 19, 19};
        actColIndices = new int[]{13, 14, 2, 12, 3, 4, 6, 2, 9, 6, 4, 5};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(24, 11);
        expData = new Complex128[]{new Complex128(0.34666, 0.12705), new Complex128(0.95514, 0.67512), new Complex128(0.84671, 0.65075), new Complex128(0.57485, 0.02965), new Complex128(0.00081, 0.82227), new Complex128(0.94886, 0.3042), new Complex128(0.03307, 0.24848), new Complex128(0.82155, 0.00032), new Complex128(0.63371, 0.2528), new Complex128(0.01639, 0.46448)};
        expRowIndices = new int[]{0, 4, 6, 9, 9, 10, 14, 15, 16, 19};
        expColIndices = new int[]{10, 0, 9, 1, 2, 3, 0, 6, 3, 2};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCols(14, 5, 0, 1));

        // ------------------ sub-case 2 ------------------
        actShape = new Shape(24, 15);
        actData = new Complex128[]{};
        actRowIndices = new int[]{};
        actColIndices = new int[]{};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(24, 11);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.removeCols(1, 5, 0, 2));

        // ------------------ sub-case 4 ------------------
        act = new CooCMatrix(new Shape(9, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeCols(2, 5, 9));
        assertThrows(IndexOutOfBoundsException.class, () -> act.removeCols(0, -1));
    }
}
