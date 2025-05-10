package org.flag4j.arrays.sparse.sparse_complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCTensorReshapeTests {
    static CooCTensor A;
    static Shape aShape;
    static Complex128[] aEntries;
    static int[][] aIndices;

    static CooCTensor exp;
    static Shape expShape;
    static Complex128[] expEntries;
    static int[][] expIndices;

    @Test
    void reshapeTests() {
        // -------------------------- sub-case 1 --------------------------
        aShape = new Shape(5, 4, 2, 1);
        aEntries = new Complex128[]{new Complex128(0.2856, 0.1775), new Complex128(0.2455, 0.6139), new Complex128(0.9386, 0.8602), new Complex128(0.194, 0.921), new Complex128(0.8078, 0.4986), new Complex128(0.359, 0.5673)};
        aIndices = new int[][]{
                {0, 0, 0, 0},
                {0, 0, 1, 0},
                {2, 2, 1, 0},
                {2, 3, 1, 0},
                {3, 3, 0, 0},
                {4, 3, 0, 0}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 5, 2, 2);
        expEntries = new Complex128[]{new Complex128(0.2856, 0.1775), new Complex128(0.2455, 0.6139), new Complex128(0.9386, 0.8602), new Complex128(0.194, 0.921), new Complex128(0.8078, 0.4986), new Complex128(0.359, 0.5673)};
        expIndices = new int[][]{
                {0, 0, 0, 0},
                {0, 0, 0, 1},
                {1, 0, 0, 1},
                {1, 0, 1, 1},
                {1, 2, 1, 0},
                {1, 4, 1, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.reshape(2, 5, 2, 2));

        // ----------------------------- sub-case 2 -----------------------------
        aShape = new Shape(5, 4, 2, 3, 15);
        aEntries = new Complex128[]{new Complex128(0.8103, 0.0203), new Complex128(0.5684, 0.4151), new Complex128(0.9044, 0.8734), new Complex128(0.201, 0.7032), new Complex128(0.9682, 0.2723), new Complex128(0.4699, 0.8203), new Complex128(0.3871, 0.3395), new Complex128(0.7851, 0.3768), new Complex128(0.2315, 0.7695), new Complex128(0.8333, 0.8837), new Complex128(0.0398, 0.559), new Complex128(0.0405, 0.9707), new Complex128(0.488, 0.8343), new Complex128(0.2441, 0.7806), new Complex128(0.3995, 0.6793), new Complex128(0.3689, 0.6126), new Complex128(0.0767, 0.9631), new Complex128(0.8007, 0.4023)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 1, 0, 0, 7},
                {0, 2, 1, 2, 4},
                {0, 3, 0, 0, 11},
                {0, 3, 1, 0, 10},
                {0, 3, 1, 2, 6},
                {1, 1, 1, 0, 7},
                {2, 0, 0, 0, 8},
                {2, 0, 0, 0, 10},
                {2, 2, 0, 2, 12},
                {2, 2, 1, 1, 6},
                {3, 0, 0, 0, 2},
                {3, 0, 1, 1, 1},
                {4, 0, 0, 2, 10},
                {4, 0, 1, 0, 7},
                {4, 2, 1, 1, 8},
                {4, 3, 0, 0, 6},
                {4, 3, 1, 2, 13}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(15, 2, 4, 15);
        expEntries = new Complex128[]{new Complex128(0.8103, 0.0203), new Complex128(0.5684, 0.4151), new Complex128(0.9044, 0.8734), new Complex128(0.201, 0.7032), new Complex128(0.9682, 0.2723), new Complex128(0.4699, 0.8203), new Complex128(0.3871, 0.3395), new Complex128(0.7851, 0.3768), new Complex128(0.2315, 0.7695), new Complex128(0.8333, 0.8837), new Complex128(0.0398, 0.559), new Complex128(0.0405, 0.9707), new Complex128(0.488, 0.8343), new Complex128(0.2441, 0.7806), new Complex128(0.3995, 0.6793), new Complex128(0.3689, 0.6126), new Complex128(0.0767, 0.9631), new Complex128(0.8007, 0.4023)};
        expIndices = new int[][]{
                {0, 0, 0, 0},
                {0, 1, 2, 7},
                {2, 0, 1, 4},
                {2, 0, 2, 11},
                {2, 1, 1, 10},
                {2, 1, 3, 6},
                {4, 0, 1, 7},
                {6, 0, 0, 8},
                {6, 0, 0, 10},
                {7, 1, 2, 12},
                {8, 0, 0, 6},
                {9, 0, 0, 2},
                {9, 1, 0, 1},
                {12, 0, 2, 10},
                {12, 0, 3, 7},
                {14, 0, 0, 8},
                {14, 0, 2, 6},
                {14, 1, 3, 13}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.reshape(15, 2, 4, 15));

        // ----------------------------- sub-case 3 -----------------------------
        aShape = new Shape(3, 16);
        aShape = new Shape(3, 16);
        aEntries = new Complex128[]{new Complex128(0.5938, 0.762), new Complex128(0.4295, 0.7988), new Complex128(0.0332, 0.3233), new Complex128(0.7022, 0.1686), new Complex128(0.7114, 0.6353), new Complex128(0.5935, 0.0851), new Complex128(0.7148, 0.5695)};
        aIndices = new int[][]{
                {1, 5},
                {1, 14},
                {1, 15},
                {2, 0},
                {2, 2},
                {2, 3},
                {2, 10}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 4, 3, 2);
        expEntries = new Complex128[]{new Complex128(0.5938, 0.762), new Complex128(0.4295, 0.7988), new Complex128(0.0332, 0.3233), new Complex128(0.7022, 0.1686), new Complex128(0.7114, 0.6353), new Complex128(0.5935, 0.0851), new Complex128(0.7148, 0.5695)};
        expIndices = new int[][]{
                {0, 3, 1, 1},
                {1, 1, 0, 0},
                {1, 1, 0, 1},
                {1, 1, 1, 0},
                {1, 1, 2, 0},
                {1, 1, 2, 1},
                {1, 3, 0, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.reshape(2, 4, 3, 2));

        // ----------------------------- sub-case 4 -----------------------------
        assertThrows(ArrayShapeException.class, ()->A.reshape(150, 12));
    }


    @Test
    void flattenTests() {
        // -------------------------- sub-case 1 --------------------------
        aShape = new Shape(3, 16);
        aEntries = new Complex128[]{
                new Complex128(1, -2), new Complex128(9.145, 0.00013),
                new Complex128(234),  new Complex128(0, 15)};
        aIndices = new int[][]{
                {0, 4},
                {0, 7},
                {1, 9},
                {2, 10}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(48);
        expEntries = new Complex128[]{new Complex128(1, -2), new Complex128(9.145, 0.00013),
                new Complex128(234),  new Complex128(0, 15)};
        expIndices = new int[][]{
                {4},
                {7},
                {25},
                {42}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten());

        // -------------------------- sub-case 2 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new Complex128[]{new Complex128(0, 15), new Complex128(0), new Complex128(-9, 154)};
        aIndices = new int[][]{
                {0, 1, 3},
                {1, 0, 1},
                {1, 1, 0}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(24);
        expEntries = new Complex128[]{new Complex128(0, 15), new Complex128(0), new Complex128(-9, 154)};
        expIndices = new int[][]{
                {7},
                {9},
                {12}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten());

        // -------------------------- sub-case 3 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new Complex128[]{new Complex128(0, 15), new Complex128(2), new Complex128(-9, 154)};
        aIndices = new int[][]{
                {1, 0, 1},
                {1, 1, 1},
                {2, 0, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(1, 24, 1);
        expEntries = new Complex128[]{new Complex128(0, 15), new Complex128(2), new Complex128(-9, 154)};
        expIndices = new int[][]{
                {0, 9, 0},
                {0, 13, 0},
                {0, 17, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten(1));

        // -------------------------- sub-case 4 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new Complex128[]{new Complex128(0, 15), new Complex128(234), new Complex128(-9, 154)};
        aIndices = new int[][]{
                {1, 0, 1},
                {1, 1, 1},
                {2, 0, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(1, 1, 24);
        expEntries = new Complex128[]{new Complex128(0, 15), new Complex128(234), new Complex128(-9, 154)};
        expIndices = new int[][]{
                {0, 0, 9},
                {0, 0, 13},
                {0, 0, 17}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.flatten(2));

        // -------------------------- sub-case 5 --------------------------
        aShape = new Shape(3, 2, 4);
        aEntries = new Complex128[]{new Complex128(0, 15), new Complex128(234), new Complex128(-9, 154)};
        aIndices = new int[][]{
                {1, 0, 1},
                {1, 1, 1},
                {2, 0, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);
        assertThrows(IndexOutOfBoundsException.class, ()->A.flatten(5));
        assertThrows(IndexOutOfBoundsException.class, ()->A.flatten(-1));
    }
}
