package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorElemDivTests {

    CooCVector a;
    int size;


    @Test
    void denseDivTestCase() {
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

        expValues = new Complex128[]{new Complex128(1.3345, -9.25).div(1.223), new Complex128(0, -45.62).div(3.4),
                new Complex128(25.612, 0.0245).div(-14.51)};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, RealFieldDenseCooVectorOps.elemDiv(a, b));

        // -------------------- sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> RealFieldDenseCooVectorOps.elemDiv(a, finalB));
    }


    @Test
    void denseComplexDivTestCase() {
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
                new Complex128(1.3345, -9.25).div(new Complex128(24.3, -0.013)),
                new Complex128(0, -45.62).div(new Complex128(2.4)),
                new Complex128(25.612, 0.0245).div(new Complex128(9924.515, 51.5))};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, DenseCooFieldVectorOps.elemDiv(a, b));

        // -------------------- sub-case 2 --------------------
        bValues = new Complex128[]{new Complex128(24.3, -0.013), new Complex128(0, 13.6), new Complex128(24),
                new Complex128(2.4), new Complex128(-994.1 ,1.45), new Complex128(1495, 13.4)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()-> DenseCooFieldVectorOps.elemDiv(a, finalB));
    }


    @Test
    void complexScalarDivTestCase() {
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
                new Complex128(1.3345, -9.25).div(b),
                new Complex128(0, -45.62).div(b),
                new Complex128(25.612, 0.0245).div(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, a.div(b));
    }


    @Test
    void realScalarDivTestCase() {
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
                new Complex128(1.3345, -9.25).div(b),
                new Complex128(0, -45.62).div(b),
                new Complex128(25.612, 0.0245).div(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, a.div(b));
    }
}
