package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorOuterProductTest {

    int[] bIndices;
    static int sparseSize;
    static CooVector a;

    @BeforeAll
    static void setup() {
        double[] aEntries = {1.0, 5.6, -9.355};
        int[] aIndices = {1, 2, 4};
        sparseSize = 5;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseOuterProdTestCase() {
        double[] bEntries;
        double[][] expEntries;
        CooVector b;
        Matrix exp;

        // -------------------- sub-case 1 --------------------
        bEntries = new double[]{1.34, -99.4};
        bIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, bIndices);
        expEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {1.34, 0.0, -99.4, 0.0, 0.0},
                {7.504, 0.0, -556.64, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {-12.535700000000002, 0.0, 929.8870000000001, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- sub-case 2 --------------------
        bEntries = new double[]{1.34, -99.4};
        bIndices = new int[]{0, 2};
        b = new CooVector(sparseSize+1445, bEntries, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.outer(finalB));
    }


    @Test
    void sparseComplexOuterProdTestCase() {
        Complex128[] bEntries;
        Complex128[][] expEntries;
        CooCVector b;
        CMatrix exp;

        // -------------------- sub-case 1 --------------------
        bEntries = new Complex128[]{new Complex128(1.34, 0.0244), new Complex128(-99, 815.66)};
        bIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, bIndices);
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.34+0.0244i"), new Complex128("0.0"), new Complex128("-99.0+815.66i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("7.504+0.13664i"), new Complex128("0.0"), new Complex128("-554.4+4567.696i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("-12.535700000000002-0.22826200000000002i"), new Complex128("0.0"), new Complex128("926.1450000000001-7630.4993i"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, RealComplexSparseVectorOps.outerProduct(a, b));

        // -------------------- sub-case 2 --------------------
        bEntries = new Complex128[]{new Complex128(1.34, 0.0244), new Complex128(-99, 815.66)};
        bIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize+103, bEntries, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->RealComplexSparseVectorOps.outerProduct(a, finalB));
    }


    @Test
    void denseOuterProdTestCase() {
        double[] bEntries;
        double[][] expEntries;
        Vector b;
        Matrix exp;

        // -------------------- sub-case 1 --------------------
        bEntries = new double[]{1.34, -0.0013, 11.56, 0.0, -13.5};
        b = new Vector(bEntries);
        expEntries = new double[][]{{0.0, 0.0, 0.0, 0.0, 0.0},
                {1.34, -0.0013, 11.56, 0.0, -13.5},
                {7.504, -0.007279999999999999, 64.736, 0.0, -75.6},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {-12.535700000000002, 0.0121615, -108.14380000000001, -0.0, 126.2925}};
        exp = new Matrix(expEntries);
        Matrix act = new Matrix(a.size, b.size,
                RealDenseSparseVectorOps.outerProduct(a.data, a.indices, a.size, b.data));

        assertEquals(exp, act);

        // -------------------- sub-case 2 --------------------
        bEntries = new double[]{1.34, -0.0013, 11.56, 0.0, -13.5, 1.305, 1.556, -1.3413, 772.24};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class,
                ()->RealDenseSparseVectorOps.outerProduct(a.data, a.indices, a.size, finalB.data));
    }


    @Test
    void denseComplexOuterProdTestCase() {
        Complex128[] bEntries;
        Complex128[][] expEntries;
        CVector b;
        CMatrix exp;

        // -------------------- sub-case 1 --------------------
        bEntries = new Complex128[]{new Complex128(24.1, 54.1), new Complex128(-9.245, 3.4), new Complex128(14.5),
                new Complex128(0, 94.14), new Complex128(113, 55.62)};
        b = new CVector(bEntries);
        expEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("-0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("24.1+54.1i"), new Complex128("-9.245+3.4i"), new Complex128("14.5"), new Complex128("0.0+94.14i"), new Complex128("113.0+55.62i")},
                {new Complex128("134.96+302.96i"), new Complex128("-51.77199999999999+19.04i"), new Complex128("81.19999999999999"), new Complex128("0.0+527.184i"), new Complex128("632.8+311.472i")},
                {new Complex128("0.0"), new Complex128("-0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("-225.45550000000003-506.10550000000006i"), new Complex128("86.486975-31.807000000000002i"), new Complex128("-135.6475"), new Complex128("-0.0-880.6797i"), new Complex128("-1057.115-520.3251i")}};
        exp = new CMatrix(expEntries);

        Complex128[] actData = new Complex128[a.size*b.size];
        RealFieldDenseCooVectorOps.outerProduct(a.data, a.indices, a.size, b.data, actData);
        CMatrix act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);

        // -------------------- sub-case 2 --------------------
        bEntries = new Complex128[]{new Complex128(24.1, 54.1), new Complex128(-9.245, 3.4)};
        b = new CVector(bEntries);

        CVector finalB = b;
        int actSize = a.size*b.size;
        assertThrows(IllegalArgumentException.class,
                ()-> RealFieldDenseCooVectorOps.outerProduct(a.data, a.indices, a.size, finalB.data, new Complex128[actSize]));
    }
}
