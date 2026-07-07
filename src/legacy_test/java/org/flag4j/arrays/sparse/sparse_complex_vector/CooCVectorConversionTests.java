package org.flag4j.arrays.sparse.sparse_complex_vector;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CooCVectorConversionTests {

    static int[] aIndices;
    static Complex128[] aEntries;
    static int sparseSize;
    static CooCVector a;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24_023;
        a = new CooCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void toMatrixTestCase() {
        Complex128[] expEntries;
        int[][] expIndices;
        Shape expShape;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new CooCMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix());

        // ------------------- sub-case 2 -------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new CooCMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(true));

        // ------------------- sub-case 3 -------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        expIndices = new int[][]{{0, 0, 0, 0}, {4, 56, 9903, 14643}};
        expShape = new Shape(1, sparseSize);
        exp = new CooCMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(false));
    }


    @Test
    void toRealTestCase() {
        double[] expEntries;
        int[] expIndices;
        int expSize;
        CooVector exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new double[]{2.455, 0, 24.56, -9356.1};
        expIndices = new int[]{4, 56, 9903, 14643};
        expSize = sparseSize;
        exp = new CooVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.toReal());
    }


    @Test
    void toTensor() {
        Complex128[] expEntries;
        int[][] expIndices;
        Shape expShape;
        CooCTensor exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new Complex128[]{
                new Complex128(2.455, -83.6), new Complex128(0, 24.56),
                new Complex128(24.56), new Complex128(-9356.1, 35)
        };
        expIndices = new int[][]{{4}, {56}, {9903}, {14643}};
        expShape = new Shape(sparseSize);
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, a.toTensor());
    }


    @Test
    void toDenseTestCase() {
        Complex128[] expEntries;
        CVector exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new Complex128[sparseSize];
        Arrays.fill(expEntries, Complex128.ZERO);       
        expEntries[aIndices[0]] = aEntries[0];
        expEntries[aIndices[1]] = aEntries[1];
        expEntries[aIndices[2]] = aEntries[2];
        expEntries[aIndices[3]] = aEntries[3];
        exp = new CVector(expEntries);

        assertTrue(exp.equals(a.toDense()));
    }


    @Test
    void fromDenseTestCase() {
        Complex128[] denseEntries;
        CVector denseVector;

        // ------------------- sub-case 1 -------------------
        denseEntries = new Complex128[sparseSize];
        Arrays.fill(denseEntries, Complex128.ZERO);
        denseEntries[aIndices[0]] = aEntries[0];
        denseEntries[aIndices[1]] = aEntries[1];
        denseEntries[aIndices[2]] = aEntries[2];
        denseEntries[aIndices[3]] = aEntries[3];
        denseVector = new CVector(denseEntries);

        assertEquals(a, denseVector.toCoo());
    }
}
