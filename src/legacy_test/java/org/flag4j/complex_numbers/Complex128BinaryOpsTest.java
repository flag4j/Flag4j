package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * This class contains various tests for ops with complex numbers.
 */
class Complex128BinaryOpsTest {
    Complex128 a;
    Complex128 b;
    Complex128 result;
    Complex128 expResult;
    double bDouble;

    @Test
    void addRealTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(2);
        b = new Complex128(5);
        expResult = new Complex128(2+5);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(102.31);
        b = new Complex128(1.3435e3);
        expResult = new Complex128(102.31+1.3435e3);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(-123.45);
        b = new Complex128(234.09264001);
        expResult = new Complex128(-123.45 + 234.09264001);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        b = new Complex128(1);
        expResult = new Complex128(Double.POSITIVE_INFINITY+1);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY);
        expResult = new Complex128(Double.POSITIVE_INFINITY+Double.NEGATIVE_INFINITY);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.NaN);
        b = new Complex128(-1234.123);
        expResult = new Complex128(Double.NaN + -1234.123);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- sub-case 6 ---------------
        a = new Complex128(1.234);
        b = new Complex128(Double.NaN);
        expResult = new Complex128(1.234 + Double.NaN);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);
    }


    @Test
    void addComplexTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(2, 10);
        b = new Complex128(5, 345);
        expResult = new Complex128(2+5, 10+345);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(4589.12398, 1.35);
        b = new Complex128(0, 124.5);
        expResult = new Complex128(4589.12398+0, 1.35+124.5);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(-45, 62759.2);
        b = new Complex128(-1.34e-15, 1.3);
        expResult = new Complex128(-45 + -1.34e-15, 62759.2 + 1.3);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, 6);
        b = new Complex128(1, Double.NEGATIVE_INFINITY);
        expResult = new Complex128(Double.POSITIVE_INFINITY+1, 6+Double.NEGATIVE_INFINITY);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = new Complex128(Double.POSITIVE_INFINITY+Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY+Double.POSITIVE_INFINITY);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.NaN, 1);
        b = new Complex128(-1234.123, Double.NaN);
        expResult = new Complex128(Double.NaN + -1234.123, 1+Double.NaN);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- sub-case 6 ---------------
        a = new Complex128(Double.NaN, Double.NaN);
        b = new Complex128(Double.NaN, Double.NaN);
        expResult = new Complex128(1.234 + Double.NaN);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));
    }


    @Test
    void addDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(34);
        bDouble = 435.234;
        expResult = new Complex128(34 + 435.234);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(-123.34);
        bDouble = 94.9492;
        expResult = new Complex128(-123.34 + 94.9492);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(50.3, -34.98165);
        bDouble = 10.5;
        expResult = new Complex128(50.3 + 10.5, -34.98165);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(50.3, Double.NEGATIVE_INFINITY);
        bDouble = 10.5;
        expResult = new Complex128(50.3 + 10.5, Double.NEGATIVE_INFINITY);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.NaN, 5.4);
        bDouble = 234.435;
        expResult = new Complex128(234.435 + Double.NaN, 5.4);

        result = a.add(bDouble);
        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(5.4, result.im);
    }


    @Test
    void subRealTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(2);
        b = new Complex128(5);
        expResult = new Complex128(2-5);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(102.31);
        b = new Complex128(1.3435e3);
        expResult = new Complex128(102.31-1.3435e3);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(-123.45);
        b = new Complex128(234.09264001);
        expResult = new Complex128(-123.45 - 234.09264001);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        b = new Complex128(1);
        expResult = new Complex128(Double.POSITIVE_INFINITY-1);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY);
        expResult = new Complex128(Double.POSITIVE_INFINITY-Double.NEGATIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.NaN);
        b = new Complex128(-1234.123);
        expResult = new Complex128(Double.NaN - -1234.123);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- sub-case 6 ---------------
        a = new Complex128(1.234);
        b = new Complex128(Double.NaN);
        expResult = new Complex128(1.234 - Double.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);
    }


    @Test
    void subComplexTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(2, 10);
        b = new Complex128(5, 345);
        expResult = new Complex128(2-5, 10-345);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(4589.12398, 1.35);
        b = new Complex128(0, 124.5);
        expResult = new Complex128(4589.12398-0, 1.35-124.5);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(-45, 62759.2);
        b = new Complex128(-1.34e-15, 1.3);
        expResult = new Complex128(-45 - -1.34e-15, 62759.2 - 1.3);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, 6);
        b = new Complex128(1, Double.NEGATIVE_INFINITY);
        expResult = new Complex128(Double.POSITIVE_INFINITY-1, 6-Double.NEGATIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = new Complex128(Double.POSITIVE_INFINITY-Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY-Double.POSITIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.NaN, 1);
        b = new Complex128(-1234.123, Double.NaN);
        expResult = new Complex128(Double.NaN - -1234.123, 1-Double.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- sub-case 6 ---------------
        a = new Complex128(Double.NaN, Double.NaN);
        b = new Complex128(Double.NaN, Double.NaN);
        expResult = new Complex128(1.234 - Double.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));
    }


    @Test
    void subDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(34);
        bDouble = 435.234;
        expResult = new Complex128(34 - 435.234);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex128(-123.34);
        bDouble = 94.9492;
        expResult = new Complex128(-123.34 - 94.9492);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(50.3, -34.98165);
        bDouble = 10.5;
        expResult = new Complex128(50.3 - 10.5, -34.98165);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(50.3, Double.NEGATIVE_INFINITY);
        bDouble = 10.5;
        expResult = new Complex128(50.3 - 10.5, Double.NEGATIVE_INFINITY);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.NaN, 5.4);
        bDouble = 234.435;
        expResult = new Complex128(234.435 - Double.NaN, 5.4);

        result = a.sub(bDouble);
        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(5.4, result.im);
    }


    @Test
    void multTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(09.3241, -93.13);
        b = new Complex128(1.355, 297e4);
        expResult = new Complex128(9.3241*1.355-(-93.13)*297e4, 9.3241*297e4-93.13*1.355);

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex128(0, -93.13);
        b = new Complex128(1.355, 0);
        expResult = new Complex128(0, -93.13*1.355);

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(1.345, -93.13);
        b = new Complex128(0, 0);
        expResult = Complex128.ZERO;

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 4 ---------------
        a = new Complex128(0, 0);
        b = new Complex128(1.345, -93.13);
        expResult = Complex128.ZERO;

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

        result = a.mult(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(Double.POSITIVE_INFINITY, result.im);


        // --------------- sub-case 6 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.mult(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(Double.NEGATIVE_INFINITY, result.im);
    }


    @Test
    void multDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(09.3241, -93.13);
        bDouble = 1.355;
        expResult = new Complex128(09.3241*1.355, -93.13*1.355);

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex128(0, 0);
        bDouble = 9.234e10;

        expResult = Complex128.ZERO;

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(1.345, -93.13);
        bDouble = 0;
        expResult = Complex128.ZERO;

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        bDouble = 9.234e10;
        expResult = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        bDouble = Double.POSITIVE_INFINITY;

        result = a.mult(bDouble);

        expResult = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);

        Assertions.assertEquals(expResult, result);
    }


    @Test
    void divTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(09.3241, -93.13);
        b = new Complex128(1.355, 297e4);
        expResult = new Complex128(-3.135690092459805e-5, -3.139441915353789e-6);

        result = a.div(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex128(0, -93.13);
        b = new Complex128(1.355, 0);
        expResult = new Complex128(0, -68.73062730627306);

        result = a.div(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex128(1.345, -93.13);
        b = new Complex128(0, 0);
        expResult = Complex128.NaN;

        result = a.div(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));


        // --------------- sub-case 4 ---------------
        a = new Complex128(0, 0);
        b = new Complex128(1.345, -93.13);
        expResult = Complex128.ZERO;

        result = a.div(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

        result = a.div(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        b = new Complex128(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.div(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));
    }


    @Test
    void divDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex128(09.3241, -93.13);
        bDouble = 1.3452;
        expResult = new Complex128(09.3241/1.3452, -93.13/1.3452);

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex128(0, -93.13);
        bDouble = 0;

        result = a.div(bDouble);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(Double.NEGATIVE_INFINITY, result.im);

        // --------------- sub-case 3 ---------------
        a = new Complex128(1.345, -93.13);
        b = new Complex128(0, 0);
        expResult = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(0, 0);
        bDouble = 24.134;
        expResult = Complex128.ZERO;

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        bDouble = Double.POSITIVE_INFINITY;

        result = a.div(bDouble);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- sub-case 5 ---------------
        a = new Complex128(Double.POSITIVE_INFINITY, 2);
        bDouble = Double.NEGATIVE_INFINITY;

        result = a.div(bDouble);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(-0.0, result.im);
    }
}
