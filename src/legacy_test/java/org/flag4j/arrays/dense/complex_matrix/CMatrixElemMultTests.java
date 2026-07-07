package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.io.PrintOptions;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixElemMultTests {
    Shape sparseShape;
    int[] rowIndices, colIndies;

    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realTestCase() {
        double[][] bEntries;
        Matrix B;

        // ------------------- sub-case 1 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {12.3, 4.45, -878.2},
                {3.456, 3.45, -65.44},
                {4.566, 0, 37.45},
                {-1, -0.0000002, 94.3}};
        B = new Matrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(1519.0500000000002,-114.39000000000001), new Complex128(201.14000000000001, -0.148185), new Complex128(-4742.280000000001)},
                {new Complex128(3.456), new Complex128(0.0, -2563.695), new Complex128(2257.68, 6085.92)},
                {new Complex128(34781.25387), new Complex128(0.0), new Complex128(0.0)},
                {new Complex128(-3.141592653589793, -3.141592653589793), new Complex128(-0.0000018429324647, -0.00000302), new Complex128(-377.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.elemMult(B));


        // ------------------- sub-case 2 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {12.3, 4.45, -878.2},
                {3.456, 3.45, -65.44},
                {4.566, 0, 37.45},
                {-1, -0.0000002, 94.3}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }


    @Test
    void complexTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ------------------- sub-case 1 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1519.0500000000002,-114.39000000000001), new Complex128(201.14000000000001, -0.148185), new Complex128(-4742.280000000001)},
                {new Complex128(3.456), new Complex128(0.0, -2563.695), new Complex128(2257.68, 6085.92)},
                {new Complex128(34781.25387), new Complex128(0.0), new Complex128(0.0)},
                {new Complex128(-3.141592653589793, -3.141592653589793), new Complex128(-0.0000018429324647, -0.00000302), new Complex128(-377.2)}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{{new Complex128(186538.84800000003,-28254.330000000005), new Complex128(9091.523065439502, -13.395924000000003), new Complex128(-25608.312000000005)},
                {new Complex128(3.456), new Complex128(-1905081.7545000003), new Complex128(488100.6000000001, -419928.48)},
                {new Complex128(264944288.38576216), new Complex128(0.0), new Complex128(0.0)},
                {new Complex128(0.0, -19.739208802178716), new Complex128(2.8619999652773907e-05, -5.565656043394e-05), new Complex128(1508.8)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1519.0500000000002,-114.39000000000001), new Complex128(201.14000000000001, -0.148185)},
                {new Complex128(3.456), new Complex128(0.0, -2563.695)},
                {new Complex128(34781.25387), new Complex128(0.0)},
                {new Complex128(-3.141592653589793, -3.141592653589793), new Complex128(-0.0000018429324647, -0.00000302)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }


    @Test
    void sparseRealTestCase() {
        double[] bEntries;
        CooMatrix B;

        // ------------------- sub-case 1 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.1)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.34, -994.1, 34.5};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = A.shape;
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndies);
        expEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("60.568000000000005-0.04462200000000001i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("-0.0"), new Complex128("0.0")},
                {new Complex128("108.38494654884786+108.38494654884786i"), new Complex128("0.0"), new Complex128("-0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo().set(0, 2, 1), A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.34, -994.1, 34.5};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = new Shape(56, 191114);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndies);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }


    @Test
    void sparseComplexTestCase() {
        Complex128[] bEntries;
        CooCMatrix B;

        // ------------------- sub-case 1 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.1)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(345.6, 94.1), new Complex128(-9.4, 34), new Complex128(4.4)};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = A.shape;
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndies);
        expEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("15624.253530000002+4241.811519999999i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("-0.0"), new Complex128("0.0")},
                {new Complex128("13.823007675795091+13.823007675795091i"), new Complex128("0.0"), new Complex128("-0.0")}};
        exp = new CMatrix(expEntries);

        PrintOptions.setPrecision(100);
        assertEquals(exp.toCoo().set(Complex128.ZERO, 2, 1), A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(345.6, 94.1), new Complex128(-9.4, 34), new Complex128(4.4)};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = new Shape(56, 191114);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndies);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }
}
