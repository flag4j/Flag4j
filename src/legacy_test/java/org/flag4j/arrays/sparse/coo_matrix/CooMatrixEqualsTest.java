package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

@SuppressWarnings("AssertBetweenInconvertibleTypes")
class CooMatrixEqualsTest {

    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;
    static CooMatrix A;


    @BeforeAll
    static void setup() {
        aShape = new Shape(401, 13_440);
        aEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5};
        aIndices = new int[][]{
                {9, 13, 141, 141, 398},
                {1_002, 5, 41, 12_234, 9_013}
        };

        A  = new CooMatrix(aShape, aEntries, aIndices[0], aIndices[1]);
    }


    @Test
    void sparseEqualsTest() {
        double[] bEntries;
        int[][] bIndices;
        CooMatrix B;

        // --------------------- sub-case 1 ---------------------
        B = A.copy();
        assertEquals(A, B);

        // --------------------- sub-case 2 ---------------------
        bEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5, 24.5};
        bIndices = new int[][]{
                {9, 13, 141, 141, 398, 400},
                {1_002, 5, 41, 12_234, 9_013, 27}
        };
        B = new CooMatrix(A.shape, bEntries, bIndices[0], bIndices[1]);
        assertNotEquals(A, B);

        // --------------------- sub-case 3 ---------------------
        bEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5};
        bIndices = new int[][]{
                {9, 13, 141, 141, 398},
                {1_002, 5, 41, 12_234, 9_013}
        };
        B = new CooMatrix(new Shape(2451, 134415), bEntries, bIndices[0], bIndices[1]);
        assertNotEquals(A, B);
    }
}
