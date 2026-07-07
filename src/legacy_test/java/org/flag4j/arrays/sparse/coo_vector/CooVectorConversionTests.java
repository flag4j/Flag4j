package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooVectorConversionTests {

    static int[] aIndices;
    static double[] aEntries;
    static int sparseSize;
    static CooVector a;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24023;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void toMatrixTestCase() {
        double[] expEntries;
        int[][] expIndices;
        Shape expShape;
        CooMatrix exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new CooMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix());

        // ------------------- sub-case 2 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{4, 56, 9903, 14643}, {0, 0, 0, 0}};
        expShape = new Shape(sparseSize, 1);
        exp = new CooMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(true));

        // ------------------- sub-case 3 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{0, 0, 0, 0}, {4, 56, 9903, 14643}};
        expShape = new Shape(1, sparseSize);
        exp = new CooMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, a.toMatrix(false));
    }


    @Test
    void toComplex() {
        Complex128[] expEntries;
        int[] expIndices;
        int expSize;
        CooCVector exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new Complex128[]{
                new Complex128(1.345), new Complex128(-989.234),
                new Complex128(5.15), new Complex128(617.4)};
        expIndices = new int[]{4, 56, 9903, 14643};
        expSize = sparseSize;
        exp = new CooCVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.toComplex());
    }


    @Test
    void toTensor() {
        double[] expEntries;
        int[][] expIndices;
        Shape expShape;
        CooTensor exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new double[]{1.345, -989.234, 5.15, 617.4};
        expIndices = new int[][]{{4}, {56}, {9903}, {14643}};
        expShape = new Shape(sparseSize);
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, a.toTensor());
    }


    @Test
    void toDenseTestCase() {
        double[] expEntries;
        Vector exp;

        // ------------------- sub-case 1 -------------------
        expEntries = new double[sparseSize];
        expEntries[aIndices[0]] = aEntries[0];
        expEntries[aIndices[1]] = aEntries[1];
        expEntries[aIndices[2]] = aEntries[2];
        expEntries[aIndices[3]] = aEntries[3];
        exp = new Vector(expEntries);

        assertEquals(exp, a.toDense());
    }


    @Test
    void fromDenseTestCase() {
        double[] denseEntries;
        Vector denseVector;

        // ------------------- sub-case 1 -------------------
        denseEntries = new double[sparseSize];
        denseEntries[aIndices[0]] = aEntries[0];
        denseEntries[aIndices[1]] = aEntries[1];
        denseEntries[aIndices[2]] = aEntries[2];
        denseEntries[aIndices[3]] = aEntries[3];
        denseVector = new Vector(denseEntries);

        assertEquals(a, CooVector.fromDense(denseVector));
    }
}
