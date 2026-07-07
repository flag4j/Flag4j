package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorOuterProductTests {

    Complex128[] aEntries;
    CVector a;

    Complex128[] actData;
    Complex128[][] expEntries;
    CMatrix exp;

    int[] sparseIndices;
    int sparseSize;


    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347),
                new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345)};
        a = new CVector(aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{245.6, -99.35};
        b = new Vector(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("357.348+1504630.8232i"), new Complex128("-144.55425-608652.57445i")},
                {new Complex128("-2267.8704+1228.0i"), new Complex128("917.3978999999999-496.75i")},
                {new Complex128("2270.5719999999997-13811.1932i"), new Complex128("-918.4907499999998+5586.897574999999i")}};
        exp = new CMatrix(expEntries);

        actData = new Complex128[a.size*b.size];
        RealFieldDenseVectorOps.outerProduct(a.data, b.data, actData);
        CMatrix act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{245.6, -99.35, 1.55, 626.7};
        b = new Vector(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("357.348+1504630.8232i"), new Complex128("-144.55425-608652.57445i"), new Complex128("2.25525+9495.83785i"), new Complex128("911.8485000000001+3839381.6649i")},
                {new Complex128("-2267.8704+1228.0i"), new Complex128("917.3978999999999-496.75i"), new Complex128("-14.3127+7.75i"), new Complex128("-5786.947800000001+3133.5i")},
                {new Complex128("2270.5719999999997-13811.1932i"), new Complex128("-918.4907499999998+5586.897574999999i"), new Complex128("14.329749999999999-87.16347499999999i"), new Complex128("5793.8414999999995-35242.16115i")}};
        exp = new CMatrix(expEntries);

        actData = new Complex128[a.size*b.size];
        RealFieldDenseVectorOps.outerProduct(a.data, b.data, actData);
        act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);
    }


    @Test
    void sparseRealDenseTestCase() {
        double[] bEntries;
        CooVector b;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{245.6};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[][]{
                {new Complex128("357.348+1504630.8232i"), new Complex128("0.0")},
                {new Complex128("-2267.8704+1228.0i"), new Complex128("-0.0")},
                {new Complex128("2270.5719999999997-13811.1932i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        Complex128[] actData = new Complex128[a.size*b.size];
        RealFieldDenseCooVectorOps.outerProduct(a.data, b.data, b.indices, b.size, actData);
        CMatrix act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{245.6, -99.35};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[][]{
                {new Complex128("357.348+1504630.8232i"), new Complex128("0.0"), new Complex128("-144.55425-608652.57445i"), new Complex128("0.0")},
                {new Complex128("-2267.8704+1228.0i"), new Complex128("-0.0"), new Complex128("917.3978999999999-496.75i"), new Complex128("-0.0")},
                {new Complex128("2270.5719999999997-13811.1932i"), new Complex128("0.0"), new Complex128("-918.4907499999998+5586.897574999999i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        actData = new Complex128[a.size*b.size];
        RealFieldDenseCooVectorOps.outerProduct(a.data, b.data, b.indices, b.size, actData);
        act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{new Complex128("24.566+8.56i"), new Complex128("-9.56-0.0035i")};
        b = new CVector(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("52477.27385+150487.385602i"), new Complex128("-35.352014499999996-58567.8722275i")},
                {new Complex128("-184.042444+201.87304i"), new Complex128("88.25954-47.832319000000005i")},
                {new Complex128("-254.25465000000003-1460.5939269999997i"), new Complex128("-88.18537925+537.6341775i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{new Complex128("24.566+8.56i"), new Complex128("-9.56-0.0035i"), new Complex128("9.35"), new Complex128("-0.001+2.6i")};
        b = new CVector(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("52477.27385+150487.385602i"), new Complex128("-35.352014499999996-58567.8722275i"), new Complex128("13.60425+57281.34445i"), new Complex128("15928.500745-9.909347i")},
                {new Complex128("-184.042444+201.87304i"), new Complex128("88.25954-47.832319000000005i"), new Complex128("-86.33789999999999+46.75i"), new Complex128("13.009234+24.003400000000003i")},
                {new Complex128("-254.25465000000003-1460.5939269999997i"), new Complex128("-88.18537925+537.6341775i"), new Complex128("86.44075-525.7925749999999i"), new Complex128("-146.218945-23.9807655i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCVector b;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{new Complex128("993.356 + 1.6i")};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[][]{
                {new Complex128("11247.48818+6085641.222532i"), new Complex128("0.0")},
                {new Complex128("-9164.649304+4981.5544i"), new Complex128("0.0")},
                {new Complex128("9093.601019999998-55875.669982i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        Complex128[] actData = new Complex128[a.size*b.size];
        DenseCooFieldVectorOps.outerProduct(a.data, b.data, b.indices, b.size, actData);
        CMatrix act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{new Complex128("993.356+1.6i"), new Complex128("0.0+8.35i")};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[][]{
                {new Complex128("11247.48818+6085641.222532i"), new Complex128("0.0"), new Complex128("51154.997449999995-12.14925i"), new Complex128("0.0")},
                {new Complex128("-9164.649304+4981.5544i"), new Complex128("0.0"), new Complex128("41.75+77.1039i"), new Complex128("0.0")},
                {new Complex128("9093.601019999998-55875.669982i"), new Complex128("0.0"), new Complex128("-469.558075-77.19574999999999i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        actData = new Complex128[a.size*b.size];
        DenseCooFieldVectorOps.outerProduct(a.data, b.data, b.indices, b.size, actData);
        act = new CMatrix(a.size, b.size, actData);

        assertEquals(exp, act);
    }
}
