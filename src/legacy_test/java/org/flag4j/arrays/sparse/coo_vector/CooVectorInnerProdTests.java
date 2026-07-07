package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.common.semiring_ops.AggregateSemiring;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorInnerProdTests {
    int[] bIndices;
    static int sparseSize;
    static CooVector a;

    @BeforeAll
    static void setup() {
        double[] aEntries = {1.0, 5.6, -9.355, 215.0};
        int[] aIndices = {1, 2, 8, 13};
        sparseSize = 15;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseInnerProdTestCase() {
        double[] bEntries;
        CooVector b;
        double exp;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{1.34, 55.15, -41.13};
        bIndices = new int[]{0, 2, 8};
        b = new CooVector(sparseSize, bEntries, bIndices);

        exp = 55.15*5.6 + -9.355*-41.13;

        assertEquals(exp, a.inner(b));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{1.34, 55.15, -41.13};
        bIndices = new int[]{0, 2, 8};
        b = new CooVector(sparseSize+23, bEntries, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.inner(finalB));
    }


    @Test
    void denseInnerProdTestCase() {
        double[] bEntries;
        Vector b;
        double exp;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{
                1.34, 55.15, -41.13, 1, 3.45,
                -99.14, 551.15, 51.5, 0, 0.134,
                0.0245, -0.0, 14.45, 6.133, 4.5};
        b = new Vector(bEntries);

        exp = 55.15 + 5.6*-41.13 + 215.0*6.133;

        assertEquals(exp, RealDenseSparseVectorOps.inner(b.data, a.data, a.indices, a.size));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{
                1.34, 55.15, -41.13, 1, 3.45,
                -99.14, 551.15, 51.5, 0, 0.134,
                0.0245, -0.0, 14.45};
        b = new Vector(bEntries);
        Vector finalB = b;
        assertThrows(IllegalArgumentException.class,
                ()->RealDenseSparseVectorOps.inner(finalB.data, a.data, a.indices, a.size));
    }


    @Test
    void sparseComplexInnerProdTestCase() {
        Complex128[] bEntries;
        CooCVector b;
        Complex128 exp;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{new Complex128(1.334, 9.4), new Complex128(-67,14), new Complex128(24,-56.134)};
        bIndices = new int[]{0, 2, 8};
        b = new CooCVector(sparseSize, bEntries, bIndices);

        exp = AggregateSemiring.sum(new Complex128[]{
                new Complex128(-67,14).conj().mult(5.6), new Complex128(24,-56.134).conj().mult(-9.355)
        });

        assertEquals(exp, RealComplexSparseVectorOps.inner(a, b));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{new Complex128(1.334, 9.4), new Complex128(-67,14), new Complex128(24,-56.134)};
        bIndices = new int[]{0, 2, 8};
        b = new CooCVector(sparseSize-1, bEntries, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->RealComplexSparseVectorOps.inner(a, finalB));
    }


    @Test
    void denseComplexInnerProdTestCase() {
        Complex128[] bEntries;
        CVector b;
        Complex128 exp;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{
                new Complex128(24.1, 54.1), new Complex128(-9.245, 3.4), new Complex128(14.5),
                new Complex128(0, 94.14), Complex128.ZERO, new Complex128(113, 55.62),
                new Complex128(54.13, 5.1), new Complex128(0.0013), new Complex128(-0.924, -994.15),
                new Complex128(24.5516, -0.415), new Complex128(0, 13.46), Complex128.ZERO,
                new Complex128(5.2, 0.924), new Complex128(0.15, .135), new Complex128(25591, 13.5),
                };
        b = new CVector(bEntries);

        exp = AggregateSemiring.sum(new Complex128[]{
                new Complex128(-9.245, 3.4).conj().mult(1.0),
                new Complex128(14.5).conj().mult(5.6),
                new Complex128(-0.924, -994.15).conj().mult(-9.355),
                new Complex128(0.15, .135).conj().mult(215.0)
        });

        assertEquals(exp, RealFieldDenseCooVectorOps.inner(a.data, a.indices, a.size, b.data));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{
                new Complex128(24.1, 54.1), new Complex128(-9.245, 3.4), new Complex128(14.5),
                new Complex128(0, 94.14), Complex128.ZERO, new Complex128(113, 55.62),
                new Complex128(54.13, 5.1), new Complex128(0.0013), new Complex128(-0.924, -994.15),
                new Complex128(24.5516, -0.415), new Complex128(0, 13.46), Complex128.ZERO,
                new Complex128(5.2, 0.924), new Complex128(0.15, .135), new Complex128(25591, 13.5),
                new Complex128(1.15, 4.55), new Complex128(91)
        };
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class,
                ()-> RealFieldDenseCooVectorOps.inner(a.data, a.indices, a.size, finalB.data));
    }


    @Test
    void normalizeTestCase() {
        // ----------------------- sub-case 1 -----------------------
        double[] expEntries = {0.0046451435284722955, 0.026012803759444852, -0.043455317708858326, 0.9987058586215435};
        int[] expIndices = {1, 2, 8, 13};
        CooVector exp = new CooVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.normalize());
    }
}
