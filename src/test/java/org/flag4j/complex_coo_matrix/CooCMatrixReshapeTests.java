package org.flag4j.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixReshapeTests {

    Shape expShape, actShape;
    Complex128[] expData, actData;
    int[] expRowIndices, actRowIndices;
    int[] expColIndices, actColIndices;
    CooCMatrix exp, act;

    @Test
    void flattenTests() {
        // ---------------- sub-case 1 ----------------
        actShape = new Shape(54, 12);
        actData = new Complex128[]{new Complex128(1, 2), new Complex128(3, 4), new Complex128(5, 6),
                new Complex128(7, 8), new Complex128(9, 10)};
        actRowIndices = new int[]{0, 14, 14, 14, 45};
        actColIndices = new int[]{9, 4, 5, 11, 6};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(1, 54*12);
        expData = new Complex128[]{new Complex128(1, 2), new Complex128(3, 4), new Complex128(5, 6),
                new Complex128(7, 8), new Complex128(9, 10)};
        expRowIndices = new int[]{0, 0, 0, 0, 0};
        expColIndices = new int[]{9, 14*12 + 4, 14*12 + 5, 14*12 + 11, 45*12 + 6};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.flatten());

        // ---------------- sub-case 2 ----------------
        actShape = new Shape(54, 12);
        actData = new Complex128[]{};
        actRowIndices = new int[]{};
        actColIndices = new int[]{};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(1, 54*12);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.flatten());

        // ---------------- sub-case 3 ----------------
        actShape = new Shape(25, 15);
        actData = new Complex128[]{new Complex128(0.2777, 0.94248), new Complex128(0.38635, 0.24736), new Complex128(0.3829, 0.22189), new Complex128(0.49247, 0.31679), new Complex128(0.5719, 0.25363), new Complex128(0.24135, 0.72457), new Complex128(0.83898, 0.67385), new Complex128(0.43352, 0.61757)};
        actRowIndices = new int[]{6, 7, 8, 8, 15, 17, 19, 23};
        actColIndices = new int[]{2, 9, 2, 11, 6, 4, 4, 12};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(1, 375);
        expData = new Complex128[]{new Complex128(0.2777, 0.94248), new Complex128(0.38635, 0.24736), new Complex128(0.3829, 0.22189), new Complex128(0.49247, 0.31679), new Complex128(0.5719, 0.25363), new Complex128(0.24135, 0.72457), new Complex128(0.83898, 0.67385), new Complex128(0.43352, 0.61757)};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{92, 114, 122, 131, 231, 259, 289, 357};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.flatten(1));

        // ---------------- sub-case 4 ----------------
        actShape = new Shape(12, 54);
        actData = new Complex128[]{new Complex128(0.51154, 0.50554), new Complex128(0.79968, 0.71548), new Complex128(0.4903, 0.39299), new Complex128(0.93451, 0.13945), new Complex128(0.89019, 0.7986), new Complex128(0.59208, 0.65129)};
        actRowIndices = new int[]{2, 4, 5, 7, 7, 9};
        actColIndices = new int[]{36, 43, 12, 2, 37, 6};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(648, 1);
        expData = new Complex128[]{new Complex128(0.51154, 0.50554), new Complex128(0.79968, 0.71548), new Complex128(0.4903, 0.39299), new Complex128(0.93451, 0.13945), new Complex128(0.89019, 0.7986), new Complex128(0.59208, 0.65129)};
        expRowIndices = new int[]{144, 259, 282, 380, 415, 492};
        expColIndices = new int[]{0, 0, 0, 0, 0, 0};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.flatten(0));

        // ---------------- sub-case 5 ----------------
        actShape = new Shape(5, 5);
        actData = new Complex128[]{};
        actRowIndices = new int[]{};
        actColIndices = new int[]{};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(25, 1);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.flatten(0));
    }


    @Test
    void reshapeTests() {
        // ---------------- sub-case 1 ----------------
        actShape = new Shape(12, 4);
        actData = new Complex128[]{};
        actRowIndices = new int[]{};
        actColIndices = new int[]{};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(3, 16);
        expData = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.reshape(expShape));

        // ---------------- sub-case 2 ----------------
        actShape = new Shape(12, 4);
        actData = new Complex128[]{new Complex128(0.34722, 0.89473), new Complex128(0.3139, 0.63988), new Complex128(0.18243, 0.63827), new Complex128(0.34544, 0.55157), new Complex128(0.85195, 0.06837)};
        actRowIndices = new int[]{0, 2, 4, 5, 10};
        actColIndices = new int[]{1, 1, 3, 3, 1};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(3, 16);
        expData = new Complex128[]{new Complex128(0.34722, 0.89473), new Complex128(0.3139, 0.63988), new Complex128(0.18243, 0.63827), new Complex128(0.34544, 0.55157), new Complex128(0.85195, 0.06837)};
        expRowIndices = new int[]{0, 0, 1, 1, 2};
        expColIndices = new int[]{1, 9, 3, 7, 9};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.reshape(expShape));

        // ---------------- sub-case 3 ----------------
        actShape = new Shape(8, 8);
        actData = new Complex128[]{new Complex128(0.07652, 0.43013), new Complex128(0.58944, 0.31322), new Complex128(0.66872, 0.3021)};
        actRowIndices = new int[]{4, 5, 6};
        actColIndices = new int[]{4, 0, 3};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(32, 2);
        expData = new Complex128[]{new Complex128(0.07652, 0.43013), new Complex128(0.58944, 0.31322), new Complex128(0.66872, 0.3021)};
        expRowIndices = new int[]{18, 20, 25};
        expColIndices = new int[]{0, 0, 1};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.reshape(expShape));

        // ---------------- sub-case 4 ----------------
        actShape = new Shape(8, 8);
        actData = new Complex128[]{new Complex128(0.07652, 0.43013), new Complex128(0.58944, 0.31322), new Complex128(0.66872, 0.3021)};
        actRowIndices = new int[]{4, 5, 6};
        actColIndices = new int[]{4, 0, 3};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        assertThrows(ArrayShapeException.class, ()->act.reshape(new Shape(4, 5)));
    }
}
