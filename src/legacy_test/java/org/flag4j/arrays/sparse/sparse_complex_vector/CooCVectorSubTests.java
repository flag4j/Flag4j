package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOps;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorSubTests {
    CooCVector a;

    @Test
    void sparseSubTestCase() {
        CooVector b;
        CooCVector exp;

        Complex128[] aValues = {new Complex128(32.5, 98), new Complex128(-8.2, 55.1), new Complex128(0, 14.5)};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new CooCVector(size, aValues, aIndices);

        double[] bValues = {44, -5.66, 22.445, -0.994, 10.5};
        int[] bIndices = {1, 5, 11, 67, 200};
        b = new CooVector(size, bValues, bIndices);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(32.5, 98), new Complex128(44).negate(), new Complex128(-8.2, 55.1).sub(-5.66),
                new Complex128(22.445).negate(), new Complex128(-0.994).negate(), new Complex128(0, 14.5), new Complex128(10.50).negate()};
        int[] expIndices = {0, 1, 5, 11, 67, 103, 200};
        exp = new CooCVector(size, expValues, expIndices);

        assertEquals(exp, RealComplexSparseVectorOps.sub(a, b));

        // --------------------- sub-case 2 ---------------------
        bValues = new double[]{44, -5.66, 22.445, -0.994, 10.5};
        bIndices = new int[]{1, 5, 11, 67, 200};
        b = new CooVector(size+13, bValues, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->RealComplexSparseVectorOps.sub(a, finalB));
    }


    @Test
    void sparseComplexSubTestCase() {
        CooCVector b, exp;

        Complex128[] aValues = {new Complex128(32.5, 98), new Complex128(-8.2, 55.1), new Complex128(0, 14.5)};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new CooCVector(size, aValues, aIndices);

        Complex128[] bValues = {new Complex128(1, -0.024),
                new Complex128(99.24, 1.5), new Complex128(0, 1.4)};
        int[] bIndices = {1, 5, 6};
        b = new CooCVector(size, bValues, bIndices);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(32.5, 98), new Complex128(1, -0.024).negate(),
                new Complex128(-8.2, 55.1).sub(new Complex128(99.24, 1.5)), new Complex128(0, 1.4).negate(),
                new Complex128(0, 14.5)
        };
        int[] expIndices = {0, 1, 5, 6, 103};
        exp = new CooCVector(size, expValues, expIndices);

        assertEquals(exp, a.sub(b));

        // --------------------- sub-case 2 ---------------------
        bValues = new Complex128[]{new Complex128(1, -0.024),
                new Complex128(99.24, 1.5), new Complex128(0, 1.4)};
        bIndices = new int[]{1, 5, 6};
        b = new CooCVector(size+13, bValues, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.sub(finalB));
    }


    @Test
    void denseTestCase() {
        Vector b;
        CVector exp;

        Complex128[] aValues = {new Complex128(32.5, 98), new Complex128(-8.2, 55.1), new Complex128(0, 14.5)};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new CooCVector(size, aValues, aIndices);

        double[] bValues = {1, 5, -0.0024, 1, 2001.256, 61, -99.24, 1.5};
        b = new Vector(bValues);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {
                aValues[0].sub(new Complex128(1)), new Complex128(5).negate(),
                aValues[1].sub(new Complex128(-0.0024)), new Complex128(1).negate(),
                new Complex128(2001.256).negate(), aValues[2].sub(new Complex128(61)),
                new Complex128(-99.24).negate(), new Complex128(1.5).negate()};
        exp = new CVector(expValues);

        assertEquals(exp, RealComplexDenseSparseVectorOps.sub(a, b));

        // --------------------- sub-case 2 ---------------------
        bValues = new double[]{1, 5, -0.0024, 1, 2001.256, 61};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->RealComplexDenseSparseVectorOps.sub(a, finalB));
    }


    @Test
    void denseComplexTestCase() {
        CVector b, exp;

        Complex128[] aValues = {new Complex128(32.5, 98), new Complex128(-8.2, 55.1)};
        int[] aIndices = {0, 2};
        int size = 5;
        a = new CooCVector(size, aValues, aIndices);

        Complex128[] bValues = {new Complex128(1.445, -9.24), new Complex128(1.45),
                new Complex128(0, -99.145), new Complex128(4.51, 8.456), new Complex128(11.34, -0.00245)};
        b = new CVector(bValues);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(32.5, 98).sub(new Complex128(1.445, -9.24)), new Complex128(1.45).negate(),
                new Complex128(-8.2, 55.1).sub(new Complex128(0, -99.145)), new Complex128(4.51, 8.456).negate(),
                new Complex128(11.34, -0.00245).negate()};
        exp = new CVector(expValues);

        assertEquals(exp, DenseCooFieldVectorOps.sub(a, b));

        // --------------------- sub-case 2 ---------------------
        bValues = new Complex128[]{new Complex128(1.445, -9.24), new Complex128(1.45),
                new Complex128(0, -99.145), new Complex128(4.51, 8.456),
                new Complex128(11.34, -0.00245), new Complex128(34.5, 0.0014)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> DenseCooFieldVectorOps.sub(a, finalB));
    }


    @Test
    void scalarTestCase() {
        double b;
        CVector exp;

        Complex128[] aValues = {new Complex128(32.5, 98), new Complex128(-8.2, 55.1), new Complex128(0, 14.5)};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new CooCVector(size, aValues, aIndices);

        b = 2.345;

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {aValues[0].sub(b), Complex128.ZERO, aValues[1].sub(b),
                Complex128.ZERO, Complex128.ZERO, aValues[2].sub(b), Complex128.ZERO, Complex128.ZERO};
        exp = new CVector(expValues);

        assertEquals(exp.toCoo(), a.sub(b));
    }


    @Test
    void complexScalarTestCase() {
        Complex128 b;
        CVector exp;

        Complex128[] aValues = {new Complex128(32.5, 98), new Complex128(-8.2, 55.1), new Complex128(0, 14.5)};
        int[] aIndices = {0, 2, 3};
        int size = 5;
        a = new CooCVector(size, aValues, aIndices);

        b = new Complex128(13.455, -1459.4521);

        // --------------------- sub-case 1 ---------------------
        Complex128[] expValues = {new Complex128(32.5, 98).sub(b), Complex128.ZERO, new Complex128(-8.2, 55.1).sub(b),
                new Complex128(0, 14.5).sub(b), Complex128.ZERO};
        exp = new CVector(expValues);

        assertEquals(exp.toCoo(), a.sub(b));
    }
}
