package org.flag4j.arrays.dense.matrix;

import org.flag4j.CustomAssertions;
import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MatrixElemOppTests {

    static double[][] aEntries;
    static Matrix A;

    @BeforeAll
    static void setup() {
        aEntries = new double[][]{
                {1.123, 2, -0.01, 0.0, -0.0, 15},
                {104.51, -64, 54, 100.455, 0.00024, 1024},
                {-9.2245, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, 10.4, -158.14}
        };
        A = new Matrix(aEntries);
    }


    @Test
    void sqrtComplexTestCase() {
        Complex128[][] expEntries;
        CMatrix exp;

        // --------------------- sub-case 1 ---------------------
        expEntries = new Complex128[][]{
                {Complex128.sqrt(1.123), Complex128.sqrt(2), Complex128.sqrt(-0.01), Complex128.sqrt(0.0), Complex128.sqrt(-0.0), Complex128.sqrt(15)},
                {Complex128.sqrt(104.51), Complex128.sqrt(-64), Complex128.sqrt(54), Complex128.sqrt(100.455), Complex128.sqrt(0.00024), Complex128.sqrt(1024)},
                {Complex128.sqrt(-9.2245), Complex128.sqrt(Double.POSITIVE_INFINITY), Complex128.sqrt(Double.NEGATIVE_INFINITY), Complex128.sqrt(Double.NaN), Complex128.sqrt(10.4), Complex128.sqrt(-158.14)}
        };

        exp = new CMatrix(expEntries);
        CustomAssertions.assertEqualsNaN(exp, A.sqrtComplex());
    }
}
