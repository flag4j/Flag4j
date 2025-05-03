package org.flag4j.arrays.sparse.coo_vector;


import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorAddTests {
    CooVector a;

    @Test
    void sparseAddTestCase() {
        CooVector b, exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new CooVector(size, aValues, aIndices);

        double[] bValues = {44, -5.66, 22.445, -0.994, 10.5};
        int[] bIndices = {1, 5, 11, 67, 200};
        b = new CooVector(size, bValues, bIndices);

        // --------------------- sub-case 1 ---------------------
        double[] expValues = {1.34, 44, 51.6-5.66, 22.445, -0.994, -0.00245, 10.5};
        int[] expIndices = {0, 1, 5, 11, 67, 103, 200};
        exp = new CooVector(size, expValues, expIndices);

        assertEquals(exp, a.add(b));

        // --------------------- sub-case 2 ---------------------
        bValues = new double[]{44, -5.66, 22.445, -0.994, 10.5};
        bIndices = new int[]{1, 5, 11, 67, 200};
        b = new CooVector(size+13, bValues, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));
    }


    @Test
    void sparseComplexAddTestCase() {
        CooCVector b, exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new CooVector(size, aValues, aIndices);

        Complex128[] bValues = {new Complex128(1, -0.024),
                new Complex128(99.24, 1.5), new Complex128(0, 1.4)};
        int[] bIndices = {1, 5, 6};
        b = new CooCVector(size, bValues, bIndices);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(1.34), new Complex128(1, -0.024),
                new Complex128(99.24+51.6, 1.5), new Complex128(0, 1.4), new Complex128(-0.00245)};
        int[] expIndices = {0, 1, 5, 6, 103};
        exp = new CooCVector(size, expValues, expIndices);

        assertEquals(exp, a.add(b));

        // --------------------- sub-case 2 ---------------------
        bValues = new Complex128[]{new Complex128(1, -0.024),
                new Complex128(99.24, 1.5), new Complex128(0, 1.4)};
        bIndices = new int[]{1, 5, 6};
        b = new CooCVector(size+13, bValues, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));
    }


    @Test
    void denseTestCase() {
        Vector b, exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new CooVector(size, aValues, aIndices);

        double[] bValues = {1, 5, -0.0024, 1, 2001.256, 61, -99.24, 1.5};
        b = new Vector(bValues);

        // --------------------- sub-case 1 ---------------------
        double[] expValues = {1+1.34, 5, -0.0024+51.6, 1, 2001.256, 61-0.00245, -99.24, 1.5};
        exp = new Vector(expValues);

        assertEquals(exp, RealDenseSparseVectorOps.add(b, a));

        // --------------------- sub-case 2 ---------------------
        bValues = new double[]{1, 5, -0.0024, 1, 2001.256, 61};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->RealDenseSparseVectorOps.add(finalB, a));
    }


    @Test
    void denseComplexTestCase() {
        CVector b, exp;

        double[] aValues = {1.34, 51.6};
        int[] aIndices = {0, 2};
        int size = 5;
        a = new CooVector(size, aValues, aIndices);

        Complex128[] bValues = {new Complex128(1.445, -9.24), new Complex128(1.45),
        new Complex128(0, -99.145), new Complex128(4.51, 8.456), new Complex128(11.34, -0.00245)};
        b = new CVector(bValues);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(1.445+1.34, -9.24), new Complex128(1.45),
                new Complex128(51.6, -99.145), new Complex128(4.51, 8.456), new Complex128(11.34, -0.00245)};
        exp = new CVector(expValues);

        assertEquals(exp, RealFieldDenseCooVectorOps.add(b, a));

        // --------------------- sub-case 2 ---------------------
        bValues = new Complex128[]{new Complex128(1.445, -9.24), new Complex128(1.45),
                new Complex128(0, -99.145), new Complex128(4.51, 8.456),
                new Complex128(11.34, -0.00245), new Complex128(34.5, 0.0014)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> RealFieldDenseCooVectorOps.add(finalB, a));
    }


    @Test
    void scalarTestCase() {
        double b;
        Vector exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new CooVector(size, aValues, aIndices);

        b = 2.345;

        // --------------------- sub-case 1 ---------------------
        double[] expValues = {1.34+2.345, 0, 51.6+2.345, 0, 0, -0.00245+2.345, 0, 0};
        exp = new Vector(expValues);

        assertEquals(exp.toCoo(), a.add(b));
    }


    @Test
    void complexScalarTestCase() {
        Complex128 b;
        CooCVector exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 3};
        int size = 5;
        a = new CooVector(size, aValues, aIndices);

        b = new Complex128(13.455, -1459.4521);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(13.455+1.34, -1459.4521), new Complex128(0),
                new Complex128(13.455+51.6, -1459.4521), new Complex128(13.455-0.00245, -1459.4521),
                new Complex128(0)};
        exp = new CVector(expValues).toCoo();

        assertEquals(exp, a.add(b));
    }
}
