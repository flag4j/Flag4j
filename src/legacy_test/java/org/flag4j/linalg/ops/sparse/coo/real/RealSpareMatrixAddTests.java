package org.flag4j.linalg.ops.sparse.coo.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealSpareMatrixAddTests {

    int[][] aIndices, bIndices, expIndices;
    double[] aEntries, bEntries, expEntries;
    CooMatrix A, B, exp;
    Shape aShape, bShape, expShape;


    @Test
    void addTestCase() {
        // ------------------- sub-case 1 -------------------
        aShape = new Shape(2025, 10005);
        aEntries = new double[]{1.144, -99.25, 1.566, 0.00356, 100.36, 9954.256, 345.2};
        aIndices = new int[][]{
                {0, 25, 502, 502, 789, 1003, 2008},
                {9892, 0, 9, 608, 92, 608, 26}
        };
        A = new CooMatrix(aShape, aEntries, aIndices[0], aIndices[1]);

        bShape = new Shape(2025, 10005);
        bEntries = new double[]{-9.456, 234.6, -22.0044, 88.3216};
        bIndices = new int[][]{
                {0, 25, 502, 2014},
                {9892, 902, 3, 14}
        };
        B = new CooMatrix(bShape, bEntries, bIndices[0], bIndices[1]);

        expShape = new Shape(2025, 10005);
        expEntries = new double[]{1.144-9.456, -99.25, 234.6, -22.0044,
                1.566, 0.00356, 100.36, 9954.256, 345.2, 88.3216};
        expIndices = new int[][]{
                {0, 25, 25, 502, 502, 502, 789, 1003, 2008, 2014},
                {9892, 0, 902, 3, 9, 608, 92, 608, 26, 14}
        };
        exp = new CooMatrix(expShape, expEntries, expIndices[0], expIndices[1]);


        assertEquals(exp, RealSparseMatrixOps.add(A, B));
    }
}
