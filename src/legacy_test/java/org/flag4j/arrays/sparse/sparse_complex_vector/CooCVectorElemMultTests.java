package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorElemMultTests {

    CooCVector a;
    int size;

    @Test
    void denseElemMultTestCase() {
        Complex128[] aValues = {
                new Complex128(1.3345, -9.25), new Complex128(0, -45.62),
                new Complex128(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new CooCVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices;
        Vector b;
        Complex128[] expValues;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 0.14};
        b = new Vector(bValues);

        expValues = new Complex128[]{new Complex128(1.3345, -9.25).mult(1.223), new Complex128(0, -45.62).mult(3.4),
                new Complex128(25.612, 0.0245).mult(-14.51)};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, RealFieldDenseCooVectorOps.elemMult(b, a));


        // -------------------- sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> RealFieldDenseCooVectorOps.elemMult(finalB, a));
    }


    @Test
    void sparseElemMultTestCase() {
        Complex128[] aValues = {
                new Complex128(1.3345, -9.25), new Complex128(0, -45.62),
                new Complex128(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new CooCVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices, bIndices;
        CooVector b;
        Complex128[] expValues;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3};
        bIndices = new int[]{1, 2, 3, 5};
        b = new CooVector(size, bValues, bIndices);

        expValues = new Complex128[]{aValues[1].mult(bValues[1]), aValues[2].mult(bValues[3])};
        expIndices = new int[]{2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, RealComplexSparseVectorOps.elemMult(a, b));

        // -------------------- sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        bIndices = new int[]{1, 2, 5, 6, 105, 132};
        b = new CooVector(140, bValues, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->RealComplexSparseVectorOps.elemMult(a, finalB));
    }


    @Test
    void denseComplexElemMultTestCase() {
        Complex128[] aValues = {
                new Complex128(1.3345, -9.25), new Complex128(0, -45.62),
                new Complex128(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new CooCVector(size, aValues, aIndices);

        Complex128[] bValues, expValues;
        int[] expIndices;
        CVector b;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        bValues = new Complex128[]{new Complex128(24.3, -0.013), new Complex128(0, 13.6),
                new Complex128(2.4), new Complex128(-994.1 ,1.45), new Complex128(1495, 13.4),
                new Complex128(9924.515, 51.5), new Complex128(24.56, -88.351)};
        b = new CVector(bValues);

        expValues = new Complex128[]{
                new Complex128(1.3345, -9.25).mult(new Complex128(24.3, -0.013)),
                new Complex128(0, -45.62).mult(new Complex128(2.4)),
                new Complex128(25.612, 0.0245).mult(new Complex128(9924.515, 51.5))};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, DenseCooFieldVectorOps.elemMult(b, a));

        // -------------------- sub-case 2 --------------------
        bValues = new Complex128[]{new Complex128(24.3, -0.013), new Complex128(0, 13.6), new Complex128(24),
                new Complex128(2.4), new Complex128(-994.1 ,1.45), new Complex128(1495, 13.4)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> DenseCooFieldVectorOps.elemMult(finalB, a));
    }


    @Test
    void sparseComplexElemMultTestCase() {
        Complex128[] aValues = {
                new Complex128(1.3345, -9.25), new Complex128(0, -45.62),
                new Complex128(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new CooCVector(size, aValues, aIndices);

        Complex128[] bValues;
        int[] expIndices, bIndices;
        CooCVector b;
        Complex128[] expValues;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        bValues = new Complex128[]{
                new Complex128(24.3, -0.013), new Complex128(0, 13.6),
                new Complex128(2.4), new Complex128(-994.1 ,1.45)
        };
        bIndices = new int[]{1, 2, 3, 5};
        b = new CooCVector(size, bValues, bIndices);

        expValues = new Complex128[]{aValues[1].mult(bValues[1]), aValues[2].mult(bValues[3])};
        expIndices = new int[]{2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- sub-case 2 --------------------
        bValues = new Complex128[]{
                new Complex128(24.3, -0.013), new Complex128(0, 13.6),
                new Complex128(2.4), new Complex128(-994.1 ,1.45),
                new Complex128(3249.56, 2122.2), new Complex128(-926.6, 324.67)
        };
        bIndices = new int[]{1, 2, 5, 6, 105, 132};
        b = new CooCVector(140, bValues, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexScalarMultTestCase() {
        Complex128[] aValues = {
                new Complex128(1.3345, -9.25), new Complex128(0, -45.62),
                new Complex128(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new CooCVector(size, aValues, aIndices);

        Complex128[] expValues;
        int[] expIndices;
        Complex128 b;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        b = new Complex128(23.55, -984.2);

        expValues = new Complex128[]{
                new Complex128(1.3345, -9.25).mult(b),
                new Complex128(0, -45.62).mult(b),
                new Complex128(25.612, 0.0245).mult(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }


    @Test
    void realScalarMultTestCase() {
        Complex128[] aValues = {
                new Complex128(1.3345, -9.25), new Complex128(0, -45.62),
                new Complex128(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new CooCVector(size, aValues, aIndices);

        Complex128[] expValues;
        int[] expIndices;
        double b;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        b = 24.5;

        expValues = new Complex128[]{
                new Complex128(1.3345, -9.25).mult(b),
                new Complex128(0, -45.62).mult(b),
                new Complex128(25.612, 0.0245).mult(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }
}
