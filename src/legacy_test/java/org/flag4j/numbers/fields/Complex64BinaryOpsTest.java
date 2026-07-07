package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * This class contains various tests for ops with complex numbers.
 */
class Complex64BinaryOpsTest {
    Complex64 a;
    Complex64 b;
    Complex64 result;
    Complex64 expResult;
    float bDouble;

    @Test
    void addRealTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(2);
        b = new Complex64(5);
        expResult = new Complex64(2+5);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(102.31f);
        b = new Complex64(1.3435e3f);
        expResult = new Complex64(102.31f+1.3435e3f);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(-123.45f);
        b = new Complex64(234.09264001f);
        expResult = new Complex64(-123.45f + 234.09264001f);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        b = new Complex64(1);
        expResult = new Complex64(Float.POSITIVE_INFINITY+1);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY);
        expResult = new Complex64(Float.POSITIVE_INFINITY+Float.NEGATIVE_INFINITY);

        result = a.add(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.NaN);
        b = new Complex64(-1234.123f);
        expResult = new Complex64(Float.NaN + -1234.123f);

        result = a.add(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- sub-case 6 ---------------
        a = new Complex64(1.234f);
        b = new Complex64(Float.NaN);
        expResult = new Complex64(1.234f + Float.NaN);

        result = a.add(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(0, result.im);
    }


    @Test
    void addComplexTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(2, 10);
        b = new Complex64(5, 345);
        expResult = new Complex64(2+5, 10+345);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(4589.12398f, 1.35f);
        b = new Complex64(0, 124.5f);
        expResult = new Complex64(4589.12398f+0, 1.35f+124.5f);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(-45, 62759.2f);
        b = new Complex64(-1.34e-15f, 1.3f);
        expResult = new Complex64(-45f + -1.34e-15f, 62759.2f + 1.3f);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, 6);
        b = new Complex64(1, Float.NEGATIVE_INFINITY);
        expResult = new Complex64(Float.POSITIVE_INFINITY+1, 6+Float.NEGATIVE_INFINITY);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = new Complex64(Float.POSITIVE_INFINITY+Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY+Float.POSITIVE_INFINITY);

        result = a.add(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.NaN, 1);
        b = new Complex64(-1234.123f, Float.NaN);
        expResult = new Complex64(Float.NaN + -1234.123f, 1+Float.NaN);

        result = a.add(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));

        // --------------- sub-case 6 ---------------
        a = new Complex64(Float.NaN, Float.NaN);
        b = new Complex64(Float.NaN, Float.NaN);
        expResult = new Complex64(1.234f + Float.NaN);

        result = a.add(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));
    }


    @Test
    void addDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(34);
        bDouble = 435.234f;
        expResult = new Complex64(34 + 435.234f);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(-123.34f);
        bDouble = 94.9492f;
        expResult = new Complex64(-123.34f + 94.9492f);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(50.3f, -34.98165f);
        bDouble = 10.5f;
        expResult = new Complex64(50.3f + 10.5f, -34.98165f);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(50.3f, Float.NEGATIVE_INFINITY);
        bDouble = 10.5f;
        expResult = new Complex64(50.3f + 10.5f, Float.NEGATIVE_INFINITY);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.NaN, 5.4f);
        bDouble = 234.435f;
        expResult = new Complex64(234.435f + Float.NaN, 5.4f);

        result = a.add(bDouble);
        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(5.4f, result.im);
    }


    @Test
    void subRealTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(2);
        b = new Complex64(5);
        expResult = new Complex64(2-5);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(102.31f);
        b = new Complex64(1.3435e3f);
        expResult = new Complex64(102.31f-1.3435e3f);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(-123.45f);
        b = new Complex64(234.09264001f);
        expResult = new Complex64(-123.45f - 234.09264001f);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        b = new Complex64(1);
        expResult = new Complex64(Float.POSITIVE_INFINITY-1);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY);
        expResult = new Complex64(Float.POSITIVE_INFINITY-Float.NEGATIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.NaN);
        b = new Complex64(-1234.123f);
        expResult = new Complex64(Float.NaN - -1234.123f);

        result = a.sub(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- sub-case 6 ---------------
        a = new Complex64(1.234f);
        b = new Complex64(Float.NaN);
        expResult = new Complex64(1.234f - Float.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(0, result.im);
    }


    @Test
    void subComplexTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(2, 10);
        b = new Complex64(5, 345);
        expResult = new Complex64(2-5, 10-345);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(4589.12398f, 1.35f);
        b = new Complex64(0, 124.5f);
        expResult = new Complex64(4589.12398f-0, 1.35f-124.5f);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(-45, 62759.2f);
        b = new Complex64(-1.34e-15f, 1.3f);
        expResult = new Complex64(-45 - -1.34e-15f, 62759.2f - 1.3f);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, 6);
        b = new Complex64(1, Float.NEGATIVE_INFINITY);
        expResult = new Complex64(Float.POSITIVE_INFINITY-1, 6-Float.NEGATIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = new Complex64(Float.POSITIVE_INFINITY-Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY-Float.POSITIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.NaN, 1);
        b = new Complex64(-1234.123f, Float.NaN);
        expResult = new Complex64(Float.NaN - -1234.123f, 1-Float.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));

        // --------------- sub-case 6 ---------------
        a = new Complex64(Float.NaN, Float.NaN);
        b = new Complex64(Float.NaN, Float.NaN);
        expResult = new Complex64(1.234f - Float.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));
    }


    @Test
    void subDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(34);
        bDouble = 435.234f;
        expResult = new Complex64(34 - 435.234f);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 2 ---------------
        a = new Complex64(-123.34f);
        bDouble = 94.9492f;
        expResult = new Complex64(-123.34f - 94.9492f);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(50.3f, -34.98165f);
        bDouble = 10.5f;
        expResult = new Complex64(50.3f - 10.5f, -34.98165f);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(50.3f, Float.NEGATIVE_INFINITY);
        bDouble = 10.5f;
        expResult = new Complex64(50.3f - 10.5f, Float.NEGATIVE_INFINITY);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.NaN, 5.4f);
        bDouble = 234.435f;
        expResult = new Complex64(234.435f - Float.NaN, 5.4f);

        result = a.sub(bDouble);
        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(5.4f, result.im);
    }


    @Test
    void multTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(09.3241f, -93.13f);
        b = new Complex64(1.355f, 297e4f);
        expResult = new Complex64(9.3241f*1.355f-(-93.13f)*297e4f, 9.3241f*297e4f-93.13f*1.355f);

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex64(0, -93.13f);
        b = new Complex64(1.355f, 0);
        expResult = new Complex64(0, -93.13f*1.355f);

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(1.345f, -93.13f);
        b = new Complex64(0, 0);
        expResult = Complex64.ZERO;

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 4 ---------------
        a = new Complex64(0, 0);
        b = new Complex64(1.345f, -93.13f);
        expResult = Complex64.ZERO;

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);

        result = a.mult(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(Float.POSITIVE_INFINITY, result.im);


        // --------------- sub-case 6 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);

        result = a.mult(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(Float.NEGATIVE_INFINITY, result.im);
    }


    @Test
    void multDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(09.3241f, -93.13f);
        bDouble = 1.355f;
        expResult = new Complex64(09.3241f*1.355f, -93.13f*1.355f);

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex64(0, 0);
        bDouble = 9.234e10f;

        expResult = Complex64.ZERO;

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(1.345f, -93.13f);
        bDouble = 0;
        expResult = Complex64.ZERO;

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        bDouble = 9.234e10f;
        expResult = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        bDouble = Float.POSITIVE_INFINITY;

        result = a.mult(bDouble);

        expResult = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);

        Assertions.assertEquals(expResult, result);
    }


    @Test
    void divTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(09.3241f, -93.13f);
        b = new Complex64(1.355f, 297e4f);
        expResult = new Complex64(-3.135690092459805e-5f, -3.1394417874253122e-6f);

        result = a.div(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex64(0, -93.13f);
        b = new Complex64(1.355f, 0);
        expResult = new Complex64(0, -68.73062133789062f);

        result = a.div(b);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 3 ---------------
        a = new Complex64(1.345f, -93.13f);
        b = new Complex64(0, 0);
        expResult = Complex64.NaN;

        result = a.div(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));


        // --------------- sub-case 4 ---------------
        a = new Complex64(0, 0);
        b = new Complex64(1.345f, -93.13f);
        expResult = Complex64.ZERO;

        result = a.div(b);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);

        result = a.div(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
        b = new Complex64(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);

        result = a.div(b);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));
    }


    @Test
    void divDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        a = new Complex64(09.3241f, -93.13f);
        bDouble = 1.3452f;
        expResult = new Complex64(09.3241f/1.3452f, -93.13f/1.3452f);

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- sub-case 2 ---------------
        a = new Complex64(0, -93.13f);
        bDouble = 0;

        result = a.div(bDouble);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(Float.NEGATIVE_INFINITY, result.im);

        // --------------- sub-case 3 ---------------
        a = new Complex64(1.345f, -93.13f);
        b = new Complex64(0, 0);
        expResult = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(0, 0);
        bDouble = 24.134f;
        expResult = Complex64.ZERO;

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- sub-case 4 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        bDouble = Float.POSITIVE_INFINITY;

        result = a.div(bDouble);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertTrue(Float.isNaN(result.im));

        // --------------- sub-case 5 ---------------
        a = new Complex64(Float.POSITIVE_INFINITY, 2);
        bDouble = Float.NEGATIVE_INFINITY;

        result = a.div(bDouble);

        Assertions.assertTrue(Float.isNaN(result.re));
        Assertions.assertEquals(-0.0f, result.im);
    }
}
