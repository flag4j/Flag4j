package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorSubTests {
    static Complex128[] aEntries, expEntries;
    static CTensor A, exp;
    static Shape aShape, bShape, expShape;

    int[][] sparseIndices;

    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(235.61, 1.45), new Complex128(0, -0.00024),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(1.345, -781.2),
                new Complex128(0.014, -2.45),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(51.0, 24.56),  new Complex128(6.1, -0.03),  new Complex128(-0.00014, 1.34),};
        aShape = new Shape(2, 3, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Tensor B;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new Complex128[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));

        // ----------------------- sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB1));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooTensor B;

        // ------------------------- sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.get1DIndex(sparseIndices[0])] = expEntries[expShape.get1DIndex(sparseIndices[0])].sub(bEntries[0]);
        expEntries[expShape.get1DIndex(sparseIndices[1])] = expEntries[expShape.get1DIndex(sparseIndices[1])].sub(bEntries[1]);
        expEntries[expShape.get1DIndex(sparseIndices[2])] = expEntries[expShape.get1DIndex(sparseIndices[2])].sub(bEntries[2]);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------------- sub-case 2 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        CooTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CTensor B;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{
                new Complex128(-0.00234, 2.452), new Complex128(15.6), new Complex128(99.2442, 9.1),
                new Complex128(100.252, 1235), new Complex128(-78.2556, -99.1441), new Complex128(0.111134, -772.4),
                new Complex128(671.455, 15.56), new Complex128(-0.00024), new Complex128(515.667, 895.52),
                new Complex128(14.515), new Complex128(100.135), new Complex128(0, 1)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);
        expEntries = new Complex128[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{
                new Complex128(-0.00234, 2.452), new Complex128(15.6), new Complex128(99.2442, 9.1),
                new Complex128(100.252, 1235), new Complex128(-78.2556, -99.1441), new Complex128(0.111134, -772.4),
                new Complex128(671.455, 15.56), new Complex128(-0.00024), new Complex128(515.667, 895.52),
                new Complex128(14.515), new Complex128(100.135), new Complex128(0, 1)
        };
        bShape = new Shape(12);
        B = new CTensor(bShape, bEntries);

        CTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCTensor B;

        // ------------------------- sub-case 1 -------------------------
        bEntries = new Complex128[]{
                new Complex128(1, -0.2045), new Complex128(-800.145, 3204.5)
        };
        bShape = new Shape( 2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        expShape = new Shape( 2, 3, 2);
        expEntries[expShape.get1DIndex(sparseIndices[0])] = expEntries[expShape.get1DIndex(sparseIndices[0])].sub(bEntries[0]);
        expEntries[expShape.get1DIndex(sparseIndices[1])] = expEntries[expShape.get1DIndex(sparseIndices[1])].sub(bEntries[1]);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------------- sub-case 2 -------------------------
        bEntries = new Complex128[]{
                new Complex128(1, -0.2045), new Complex128(-800.145, 3204.5)
        };
        bShape = new Shape(13, 89, 14576);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        CooCTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void subRealScalarTestCase() {
        double b;

        // ----------------------- sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new Complex128[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void subComplexScalar() {
        Complex128[] expEntries;
        CTensor exp;
        Complex128 b;

        // ----------------------- sub-case 1 -----------------------
        b = new Complex128(234.5, -364.00);
        expEntries = new Complex128[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void realDenseSubEqTestCase() {
        double[] bEntries;
        Tensor B;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new Complex128[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));

        // ----------------------- sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB1));
    }


    @Test
    void subEqRealScalarTestCase() {
        double b;

        // ----------------------- sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new Complex128[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(b);
        assertEquals(exp, A);
    }


    @Test
    void complexDenseSubEqTestCase() {
        Complex128[] bEntries;
        CTensor B;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.34, -0.324), new Complex128(0.134), new Complex128(0, 2.501),
                new Complex128(-994.1, 0.0234), new Complex128(9.4, 0.14), new Complex128(5.2, 1104.5),
                new Complex128(103.45, 6), new Complex128(-23.45, 1.4), new Complex128(-2, -0.4),
                new Complex128(3.55), Complex128.ZERO, new Complex128(100.2456),
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);
        expEntries = new Complex128[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.34, -0.324), new Complex128(0.134), new Complex128(0, 2.501),
                new Complex128(-994.1, 0.0234), new Complex128(9.4, 0.14), new Complex128(5.2, 1104.5),
                new Complex128(103.45, 6), new Complex128(-23.45, 1.4), new Complex128(-2, -0.4),
                new Complex128(3.55), Complex128.ZERO, new Complex128(100.2456),
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new CTensor(bShape, bEntries);

        CTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));

        // ----------------------- sub-case 3 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.34, -0.324), new Complex128(0.134), new Complex128(0, 2.501),
                new Complex128(-994.1, 0.0234), new Complex128(9.4, 0.14), new Complex128(5.2, 1104.5),
                new Complex128(103.45, 6), new Complex128(-23.45, 1.4), new Complex128(-2, -0.4),
                new Complex128(3.55), Complex128.ZERO, new Complex128(100.2456),
                new Complex128(1.344), new Complex128(0.924, 55.6)
        };
        bShape = new Shape(7, 2);
        B = new CTensor(bShape, bEntries);

        CTensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB1));
    }


    @Test
    void subEqComplexScalarTestCase() {
        Complex128 b;

        // ----------------------- sub-case 1 -----------------------
        b = new Complex128(234.25256, -1451.13455);
        expEntries = new Complex128[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(b);
        assertEquals(exp, A);
    }
}
