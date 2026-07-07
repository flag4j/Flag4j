package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CVectorSubTests {

    Complex128[] aEntries = new Complex128[]{
            new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
            new Complex128(3.54698), new Complex128(0, 8.356)};
    CVector a = new CVector(aEntries);
    Complex128[] expEntries;
    CVector exp;

    int sparseSize;
    int[] sparseIndices;


    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{
                new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
                new Complex128(3.54698), new Complex128(0, 8.356)};
        a = new CVector(aEntries);
    }

    @Test
    void scalDoubleTestCase() {
        double b;

        // ------------------ sub-case 1 ------------------
        b = 45.15;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 2 ------------------
        b = -2384.526;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 3 ------------------
        b = Double.POSITIVE_INFINITY;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 4 ------------------
        b = Double.NaN;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        CVector act = a.sub(b);

        for(int i=0; i<act.size; i++) {
            assertTrue(Double.isNaN(act.get(i).re));
            assertEquals(exp.get(i).im, act.get(i).im);
        }
    }


    @Test
    void scalComplex128TestCase() {
        Complex128 b;

        // ------------------ sub-case 1 ------------------
        b = new Complex128(9.145, -523923.15965);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 2 ------------------
        b = new Complex128(0, -14.36);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 3 ------------------
        b = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 4 ------------------
        b = new Complex128(6.24, Double.POSITIVE_INFINITY);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 5 ------------------
        b = new Complex128(Double.NEGATIVE_INFINITY, 135.5);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 6 ------------------
        b = new Complex128(Double.NaN, Double.NaN);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        CVector act = a.sub(b);

        for (int i = 0; i < act.size; i++) {
            assertTrue(Double.isNaN(act.get(i).re));
            assertTrue(Double.isNaN(act.get(i).im));
        }
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ------------------ sub-case 1 ------------------
        bEntries = new double[]{54.1354, -99.2344, 0, 0.023};
        b = new Vector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 2 ------------------
        bEntries = new double[]{-54.1354, -99.2344, 0, Double.NEGATIVE_INFINITY};
        b = new Vector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 3 ------------------
        bEntries = new double[]{-54.1354, -99.2344, 0, 14, 1.5};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB));

        // ------------------ sub-case 4 ------------------
        bEntries = new double[]{-54.1354, -99.2344};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB2));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;

        // ------------------ sub-case 1 ------------------
        bEntries = new double[]{54.1354, -1.4};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(54.1354), new Complex128(-24.565, 9.3),
                new Complex128(3.54698).sub(-1.4), new Complex128(0, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 2 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
                new Complex128(3.54698), new Complex128(0, 8.356).sub(-1.4)};
        exp = new CVector(expEntries);

        CVector act = a.sub(b);
        assertEquals(exp, act);

        // ------------------ sub-case 3 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB));

        // ------------------ sub-case 4 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 3;
        sparseIndices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB2));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        // ------------------ sub-case 1 ------------------
        bEntries = new Complex128[]{new Complex128(2.45, -99.24), new Complex128(9),
                new Complex128(0, -8.35), new Complex128(-9924.5, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 2 ------------------
        bEntries = new Complex128[]{new Complex128(2455, 0.0001424), new Complex128(-9),
                new Complex128(-0.0, Double.NEGATIVE_INFINITY), new Complex128(Double.POSITIVE_INFINITY, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 3 ------------------
        bEntries = new Complex128[]{new Complex128(2.45, -99.24), new Complex128(9),
                new Complex128(0, -8.35), new Complex128(-9924.5, 24.656),
                new Complex128(9.345, 1344)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB));

        // ------------------ sub-case 4 ------------------
        bEntries = new Complex128[]{new Complex128(2.45, -99.24), new Complex128(9),
                new Complex128(0, -8.35)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB2));
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCVector b;

        // ------------------ sub-case 1 ------------------
        bEntries = new Complex128[]{new Complex128(-9.24, 8.14), new Complex128(0, 22455.6126)};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(bEntries[0]), new Complex128(-24.565, 9.3),
                new Complex128(3.54698).sub(bEntries[1]), new Complex128(0, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sub(b));

        // ------------------ sub-case 2 ------------------
        bEntries = new Complex128[]{new Complex128(4.5, 0.00245)};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
                new Complex128(3.54698), new Complex128(0, 8.356).sub(bEntries[0])};
        exp = new CVector(expEntries);

        CVector act = a.sub(b);

        assertEquals(exp, act);

        // ------------------ sub-case 3 ------------------
        bEntries = new Complex128[]{new Complex128(9.3455, 15.6)};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB));

        // ------------------ sub-case 4 ------------------
        bEntries = new Complex128[]{new Complex128(9.3455, 15.6)};
        sparseSize = 3;
        sparseIndices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB2));
    }

    // ----------------- SubEq Tests -----------------

    @Test
    void scalDoubleEqTestCase() {
        double b;

        // ------------------ sub-case 1 ------------------
        setup();
        b = 45.15;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        setup();
        b = -2384.526;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        setup();
        b = Double.POSITIVE_INFINITY;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 4 ------------------
        setup();
        b = Double.NaN;
        expEntries = new Complex128[]{
                new Complex128(2.566-b, -9.24), new Complex128(-24.565-b, 9.3),
                new Complex128(3.54698-b), new Complex128(0-b, 8.356)};
        exp = new CVector(expEntries);

        a.subEq(b);

        for(int i=0; i<a.size; i++) {
            assertTrue(Double.isNaN(a.get(i).re));
            assertEquals(exp.get(i).im, a.get(i).im);
        }
    }


    @Test
    void scalComplex128EqTestCase() {
        Complex128 b;

        // ------------------ sub-case 1 ------------------
        setup();
        b = new Complex128(9.145, -523923.15965);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        setup();
        b = new Complex128(0, -14.36);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        setup();
        b = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 4 ------------------
        setup();
        b = new Complex128(6.24, Double.POSITIVE_INFINITY);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 5 ------------------
        setup();
        b = new Complex128(Double.NEGATIVE_INFINITY, 135.5);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 6 ------------------
        setup();
        b = new Complex128(Double.NaN, Double.NaN);
        expEntries = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b), new Complex128(-24.565, 9.3).sub(b),
                new Complex128(3.54698).sub(b), new Complex128(0, 8.356).sub(b)};
        exp = new CVector(expEntries);

        a.subEq(b);

        for (int i = 0; i < a.size; i++) {
            assertTrue(Double.isNaN(a.get(i).re));
            assertTrue(Double.isNaN(a.get(i).im));
        }
    }


    @Test
    void realDenseEqTestCase() {
        double[] bEntries;
        Vector b;

        // ------------------ sub-case 1 ------------------
        setup();
        bEntries = new double[]{54.1354, -99.2344, 0, 0.023};
        b = new Vector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        setup();
        bEntries = new double[]{-54.1354, -99.2344, 0, Double.NEGATIVE_INFINITY};
        b = new Vector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        setup();
        bEntries = new double[]{-54.1354, -99.2344, 0, 14, 1.5};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.subEq(finalB));

        // ------------------ sub-case 4 ------------------
        setup();
        bEntries = new double[]{-54.1354, -99.2344};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.subEq(finalB2));
    }


    @Test
    void realSparseEqTestCase() {
        double[] bEntries;
        CooVector b;

        // ------------------ sub-case 1 ------------------
        setup();
        bEntries = new double[]{54.1354, -1.4};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(54.1354), new Complex128(-24.565, 9.3),
                new Complex128(3.54698).sub(-1.4), new Complex128(0, 8.356)};
        exp = new CVector(expEntries);

        RealFieldDenseCooVectorOps.subEq(a, b);
        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        setup();
        bEntries = new double[]{-1.4};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
                new Complex128(3.54698), new Complex128(0, 8.356).sub(-1.4)};
        exp = new CVector(expEntries);

        RealFieldDenseCooVectorOps.subEq(a, b);
        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        setup();
        bEntries = new double[]{-1.4};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        final CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> RealFieldDenseCooVectorOps.subEq(a, finalB));

        // ------------------ sub-case 4 ------------------
        setup();
        bEntries = new double[]{-1.4};
        sparseSize = 3;
        sparseIndices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        final CooVector finalB1 = b;
        assertThrows(LinearAlgebraException.class, ()-> RealFieldDenseCooVectorOps.subEq(a, finalB1));
    }


    @Test
    void complexDenseEqTestCase() {
        Complex128[] bEntries;
        CVector b;

        // ------------------ sub-case 1 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(2.45, -99.24), new Complex128(9),
                new Complex128(0, -8.35), new Complex128(-9924.5, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(2455, 0.0001424), new Complex128(-9),
                new Complex128(-0.0, Double.NEGATIVE_INFINITY), new Complex128(Double.POSITIVE_INFINITY, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(b.get(0)), new Complex128(-24.565, 9.3).sub(b.get(1)),
                new Complex128(3.54698).sub(b.get(2)), new Complex128(0, 8.356).sub(b.get(3))};
        exp = new CVector(expEntries);

        a.subEq(b);
        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(2.45, -99.24), new Complex128(9),
                new Complex128(0, -8.35), new Complex128(-9924.5, 24.656),
                new Complex128(9.345, 1344)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.subEq(finalB));

        // ------------------ sub-case 4 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(2.45, -99.24), new Complex128(9),
                new Complex128(0, -8.35)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.subEq(finalB2));
    }


    @Test
    void complexSparseEqTestCase() {
        Complex128[] bEntries;
        CooCVector b;

        // ------------------ sub-case 1 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(-9.24, 8.14), new Complex128(0, 22455.6126)};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24).sub(bEntries[0]), new Complex128(-24.565, 9.3),
                new Complex128(3.54698).sub(bEntries[1]), new Complex128(0, 8.356)};
        exp = new CVector(expEntries);

        DenseCooFieldVectorOps.subEq(a, b);
        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(4.5, 0.00245)};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new Complex128[]{
                new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
                new Complex128(3.54698), new Complex128(0, 8.356).sub(bEntries[0])};
        exp = new CVector(expEntries);

        DenseCooFieldVectorOps.subEq(a, b);
        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(9.3455, 15.6)};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        final CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> DenseCooFieldVectorOps.subEq(a, finalB));

        // ------------------ sub-case 4 ------------------
        setup();
        bEntries = new Complex128[]{new Complex128(9.3455, 15.6)};
        sparseSize = 3;
        sparseIndices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        final CooCVector finalB1 = b;
        assertThrows(LinearAlgebraException.class, ()-> DenseCooFieldVectorOps.subEq(a, finalB1));
    }
}
