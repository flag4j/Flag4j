package org.flag4j.arrays.dense.vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorOuterProductTests {

    double[] aEntries;
    Vector a;
    int[] sparseIndices;
    int sparseSize;

    @Test
    void realDenseOuterTestCase() {
        double[] bEntries;
        Vector b;
        double[][] expEntries;
        Matrix exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new double[]{16.345, -9.234, 0.000154};
        b = new Vector(bEntries);
        expEntries = new double[][]{
                {16.345, -9.234, 0.000154},
                {91.53199999999998, -51.7104, 0.0008623999999999999},
                {-152.907475, 86.38407000000001, -0.00144067}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void realSparseOuterTestCase() {
        double[] bEntries;
        CooVector b;
        double[][] expEntries;
        Matrix exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new double[]{16.345};
        sparseIndices = new int[]{1};
        sparseSize = 3;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new double[][]{{0.0, 16.345, 0.0},
                {0.0, 91.53199999999998, 0.0},
                {0.0, -152.907475, 0.0}};
        exp = new Matrix(expEntries);
        Matrix act = new Matrix(a.size, b.size,
                RealDenseSparseVectorOps.outerProduct(a.data, b.data, b.indices, b.size));

        assertEquals(exp, act);
    }

    @Test
    void complexDenseOuterTestCase() {
        Complex128[] bEntries;
        CVector b;
        Complex128[][] expEntries;
        CMatrix exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128("0.0+45.0i"), new Complex128("-9.345+2.111105i"), new Complex128("71.5-8.0i")};
        b = new CVector(bEntries);
        expEntries = new Complex128[][]{{new Complex128("0.0-45.0i"), new Complex128("-9.345-2.111105i"), new Complex128("71.5+8.0i")},
                {new Complex128("0.0-251.99999999999997i"), new Complex128("-52.332-11.822187999999999i"), new Complex128("400.4+44.8i")},
                {new Complex128("0.0+420.975i"), new Complex128("87.422475+19.749387275i"), new Complex128("-668.8825-74.84i")}};
        exp = new CMatrix(expEntries);

        Complex128[] actData = new Complex128[a.size*b.size];
        RealFieldDenseVectorOps.outerProduct(a.data, b.data, actData);
        CMatrix act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);
    }


    @Test
    void complexSparseOuterTestCase() {
        Complex128[] bEntries;
        CooCVector b;
        Complex128[][] expEntries;
        CMatrix exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128("71.5-8.0i")};
        sparseIndices = new int[]{2};
        sparseSize = 3;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("0.0"), new Complex128("71.5+8.0i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("400.4+44.8i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("-668.8825-74.84i")}};
        exp = new CMatrix(expEntries);

        Complex128[] actData = new Complex128[a.data.length*b.size];
        RealFieldDenseCooVectorOps.outerProduct(a.data, b.data, b.indices, b.size, actData);
        CMatrix act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);
    }
}
