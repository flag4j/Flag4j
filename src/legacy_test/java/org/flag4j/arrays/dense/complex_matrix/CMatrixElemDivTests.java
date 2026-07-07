package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.CustomAssertions;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixElemDivTests {
    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realTestCase() {
        double[][] bEntries;
        Matrix B;

        // ------------------- sub-case 1 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.0)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {12.3, 4.45, -878.2},
                {3.456, 3.45, -65.44},
                {4.566, 0, 37.45},
                {-1, -0.0000002, 94.3}};
        B = new Matrix(bEntries);
        expEntries = new Complex128[][]{{new Complex128("10.040650406504064-0.7560975609756098i"), new Complex128("10.157303370786517-0.0074831460674157305i"), new Complex128("-0.00614894101571396")},
                {new Complex128("0.28935185185185186"), new Complex128("0.0-215.3913043478261i"), new Complex128("0.527200488997555+1.4211491442542787i")},
                {new Complex128("1668.2971966710468"), new Complex128(Double.NaN, Double.NaN), new Complex128("0.0")},
                {new Complex128("-3.141592653589793-3.141592653589793i"), new Complex128("-46073311.61750001-75500000.0i"), new Complex128("-0.042417815482502653")}};
        exp = new CMatrix(expEntries);

        CustomAssertions.assertEqualsNaN(exp, A.div(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.div(finalB));
    }


    @Test
    void complexTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ------------------- sub-case 1 -------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), new Complex128(1)},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1519.0500000000002,-114.39000000000001), new Complex128(201.14000000000001, -0.148185), new Complex128(-4742.280000000001)},
                {new Complex128(3.456), new Complex128(0.0, -2563.695), new Complex128(2257.68, 6085.92)},
                {new Complex128(34781.25387), new Complex128(0.0), new Complex128(0.0)},
                {new Complex128(-3.141592653589793, -3.141592653589793), new Complex128(-0.0000018429324647, -0.00000302), new Complex128(-377.2)}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{{new Complex128("0.08130081300813005"), new Complex128("0.22471910112359553"), new Complex128("-0.0011386927806877705")},
                {new Complex128("0.28935185185185186"), new Complex128("0.2898550724637681"), new Complex128(-0.01528117359413203)},
                {new Complex128("0.2190100744634253"), new Complex128(Double.NaN, Double.NaN), new Complex128(Double.NaN, Double.NaN)},
                {new Complex128("-1.0"), new Complex128("-5000000.0"), new Complex128("0.010604453870625663")}};
        exp = new CMatrix(expEntries);

        CustomAssertions.assertEqualsNaN(exp, A.div(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.div(finalB));
    }
}
