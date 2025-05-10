package org.flag4j.arrays.sparse.sparse_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooTensorReshapeTests {
    static CooTensor A;
    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;

    static CooTensor exp;
    static Shape expShape;
    static double[] expEntries;
    static int[][] expIndices;

    @Test
    void reshapeTests() {
        // -------------------------- sub-case 1 --------------------------
        aShape = new Shape(5, 4, 2, 1);
        aEntries = new double[]{-0.2594625644447393, -0.11800739013805872, -1.8499182919471657};
        aIndices = new int[][]{
                {2, 2, 0, 0},
                {2, 3, 0, 0},
                {4, 1, 1, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 5, 2, 2);
        expEntries = new double[]{-0.2594625644447393, -0.11800739013805872, -1.8499182919471657};
        expIndices = new int[][]{
                {1, 0, 0, 0},
                {1, 0, 1, 0},
                {1, 3, 1, 1}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.reshape(2, 5, 2, 2));

        // ----------------------------- sub-case 2 -----------------------------
        aShape = new Shape(5, 4, 2, 3, 15);
        aEntries = new double[]{-1.1499856217218563, -0.33276615768221923, -1.7712698524382784, 0.45988194186997083, -1.0000258840502727, -1.2896045900552038, -1.0495292341142137, 0.5653540034076624, 0.09844833075965526, 1.389726418783007, -0.03253455760212258, 0.8128434562240154, -0.5805363805458708, -0.9687145707590211, 0.005130776492485523, -1.1693463926292427, -0.18736097719279932, -0.588774063376806};
        aIndices = new int[][]{
                {0, 2, 1, 2, 8},
                {1, 0, 0, 0, 2},
                {1, 1, 0, 1, 9},
                {1, 2, 0, 2, 13},
                {1, 3, 1, 1, 3},
                {1, 3, 1, 1, 7},
                {2, 0, 0, 1, 9},
                {2, 2, 1, 0, 11},
                {2, 3, 1, 2, 10},
                {3, 0, 0, 1, 13},
                {3, 0, 1, 0, 6},
                {3, 1, 0, 0, 10},
                {3, 1, 1, 1, 9},
                {4, 0, 1, 0, 9},
                {4, 1, 1, 0, 14},
                {4, 2, 0, 0, 8},
                {4, 3, 0, 2, 6},
                {4, 3, 1, 2, 12}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(15, 2, 4, 15);
        expEntries = new double[]{-1.1499856217218563, -0.33276615768221923, -1.7712698524382784, 0.45988194186997083, -1.0000258840502727, -1.2896045900552038, -1.0495292341142137, 0.5653540034076624, 0.09844833075965526, 1.389726418783007, -0.03253455760212258, 0.8128434562240154, -0.5805363805458708, -0.9687145707590211, 0.005130776492485523, -1.1693463926292427, -0.18736097719279932, -0.588774063376806};
        expIndices = new int[][]{
                {2, 0, 1, 8},
                {3, 0, 0, 2},
                {3, 1, 3, 9},
                {4, 1, 2, 13},
                {5, 1, 2, 3},
                {5, 1, 2, 7},
                {6, 0, 1, 9},
                {7, 1, 3, 11},
                {8, 1, 3, 10},
                {9, 0, 1, 13},
                {9, 0, 3, 6},
                {9, 1, 2, 10},
                {10, 0, 2, 9},
                {12, 0, 3, 9},
                {13, 0, 1, 14},
                {13, 1, 0, 8},
                {14, 1, 0, 6},
                {14, 1, 3, 12}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.reshape(15, 2, 4, 15));

        // ----------------------------- sub-case 3 -----------------------------
        aShape = new Shape(3, 16);
        aEntries = new double[]{-0.5564583772612858, 1.3880160320768695, -0.041746799108138805, 0.22670438356409295};
        aIndices = new int[][]{
                {0, 7},
                {0, 8},
                {0, 15},
                {1, 7}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 4, 3, 2);
        expEntries = new double[]{-0.5564583772612858, 1.3880160320768695, -0.041746799108138805, 0.22670438356409295};
        expIndices = new int[][]{
                {0, 1, 0, 1},
                {0, 1, 1, 0},
                {0, 2, 1, 1},
                {0, 3, 2, 1}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.reshape(2, 4, 3, 2));

        // ----------------------------- sub-case 4 -----------------------------
        aShape = new Shape(3, 16);
        aEntries = new double[]{-0.5564583772612858, 1.3880160320768695, -0.041746799108138805, 0.22670438356409295};
        aIndices = new int[][]{
                {0, 7},
                {0, 8},
                {0, 15},
                {1, 7}};
        A = new CooTensor(aShape, aEntries, aIndices);
        assertThrows(ArrayShapeException.class, ()->A.reshape(150, 12));
    }


    @Test
    void flattenTests() {
        // -------------------------- sub-case 1 --------------------------
        aShape = new Shape(3, 16);
        aEntries = new double[]{-0.4396095255063526, -0.008544443239199374, 1.6354416874939133, -0.7535470743266395};
        aIndices = new int[][]{
                {0, 4},
                {0, 7},
                {1, 9},
                {2, 10}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(48);
        expEntries = new double[]{-0.4396095255063526, -0.008544443239199374, 1.6354416874939133, -0.7535470743266395};
        expIndices = new int[][]{
                {4},
                {7},
                {25},
                {42}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten());

        // -------------------------- sub-case 2 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new double[]{0.8809801303815625, 1.2884192212811383, 0.6540684159095426};
        aIndices = new int[][]{
                {0, 1, 3},
                {1, 0, 1},
                {1, 1, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(24);
        expEntries = new double[]{0.8809801303815625, 1.2884192212811383, 0.6540684159095426};
        expIndices = new int[][]{
                {7},
                {9},
                {12}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten());

        // -------------------------- sub-case 3 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new double[]{-0.2100281314624281, 1.4401356481011265, -0.1396976427551165};
        aIndices = new int[][]{
                {1, 0, 1},
                {1, 1, 1},
                {2, 0, 1}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(1, 24, 1);
        expEntries = new double[]{-0.2100281314624281, 1.4401356481011265, -0.1396976427551165};
        expIndices = new int[][]{
                {0, 9, 0},
                {0, 13, 0},
                {0, 17, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten(1));

        // -------------------------- sub-case 4 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new double[]{-0.2100281314624281, 1.4401356481011265, -0.1396976427551165};
        aIndices = new int[][]{
                {1, 0, 1},
                {1, 1, 1},
                {2, 0, 1}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(1, 1, 24);
        expEntries = new double[]{-0.2100281314624281, 1.4401356481011265, -0.1396976427551165};
        expIndices = new int[][]{
                {0, 0, 9},
                {0, 0, 13},
                {0, 0, 17}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten(2));

        // -------------------------- sub-case 5 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new double[]{-0.2100281314624281, 1.4401356481011265, -0.1396976427551165};
        aIndices = new int[][]{
                {1, 0, 1},
                {1, 1, 1},
                {2, 0, 1}};
        A = new CooTensor(aShape, aEntries, aIndices);
        assertThrows(IndexOutOfBoundsException.class, ()->A.flatten(5));
        assertThrows(IndexOutOfBoundsException.class, ()->A.flatten(-1));
    }
}
