package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64UnaryOpsTest {
    Complex64 a;
    Complex64 expValue, value;
    float expValueFloat, valueFloat;
    double expValueDouble, valueDouble;


    @Test
    void magTestCase() {
        // ----------- sub-case 1 --------------
        a = new Complex64(0);
        expValueDouble = 0;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- sub-case 2 --------------
        a = new Complex64(2.4f);
        expValueDouble = 2.4f;

        valueDouble= a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 3 --------------
        a = new Complex64(-10.394f);
        expValueDouble = 10.394f;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- sub-case 4 --------------
        a = new Complex64(2, 8);
        expValueDouble = Math.sqrt(4+64);

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(-8.42f, 1.94f);
        expValueDouble = Math.sqrt(Math.pow(-8.42f, 2) + Math.pow(1.94f, 2));

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        expValueDouble = Float.POSITIVE_INFINITY;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(Float.NaN, 2.3f);
        valueDouble = a.mag();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void magDoubleTestCase() {
        // ----------- sub-case 1 --------------
        a = new Complex64(0);
        expValueDouble = 0;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- sub-case 2 --------------
        a = new Complex64(2.4f);
        expValueDouble = 2.4f;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 3 --------------
        a = new Complex64(-10.394f);
        expValueDouble = 10.394f;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- sub-case 4 --------------
        a = new Complex64(2, 8);
        expValueDouble = Math.sqrt(4+64);
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(-8.42f, 1.94f);
        expValueDouble = Math.sqrt(Math.pow(-8.42f, 2) + Math.pow(1.94f, 2));
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        expValueDouble = Float.POSITIVE_INFINITY;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(Float.NaN, 2.3f);
        valueDouble = a.mag();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void absTestCase() {
        // ----------- sub-case 1 --------------
        a = new Complex64(0);
        expValueDouble = 0;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- sub-case 2 --------------
        a = new Complex64(2.4f);
        expValueDouble = 2.4f;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 3 --------------
        a = new Complex64(-10.394f);
        expValueDouble = 10.394f;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- sub-case 4 --------------
        a = new Complex64(2, 8);
        expValueDouble = Math.sqrt(4+64);
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(-8.42f, 1.94f);
        expValueDouble = Math.sqrt(Math.pow(-8.42f, 2) + Math.pow(1.94f, 2));
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        expValueDouble = Float.POSITIVE_INFINITY;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- sub-case 5 --------------
        a = new Complex64(Float.NaN, 2.3f);
        valueDouble = a.abs();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void negateTestCase() {
        // ---------- sub-case 1 ------------
        a = new Complex64(4);
        expValue = new Complex64(-4);
        value = a.negate();
        Assertions.assertEquals(expValue, value);

        // ---------- sub-case 2 ------------
        a = new Complex64(-2.445f);
        expValue = new Complex64(2.445f);
        value = a.negate();
        Assertions.assertEquals(expValue, value);


        // ---------- sub-case 3 ------------
        a = new Complex64(13.4f, -123);
        expValue = new Complex64(-13.4f, 123);
        value = a.negate();
        Assertions.assertEquals(expValue, value);


        // ---------- sub-case 4 ------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        expValue = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        value = a.negate();
        Assertions.assertEquals(expValue, value);

        // ---------- sub-case 5 ------------
        a = Complex64.NaN;
        value = a.negate();
        Assertions.assertTrue(Float.isNaN(value.re));
        Assertions.assertTrue(Float.isNaN(value.im));
    }


    @Test
    void invertTestCase() {
        // ---------- sub-case 1 ------------
        a = new Complex64(4);
        expValue = new Complex64(1.0f/4.0f);
        value = a.invert();
        Assertions.assertEquals(expValue, value);

        // ---------- sub-case 2 ------------
        a = new Complex64(-2.445f);
        expValue = new Complex64(-0.40899795501022496f);
        value = a.invert();
        Assertions.assertEquals(expValue, value);


        // ---------- sub-case 3 ------------
        a = new Complex64(13.4f, -123);
        expValue = new Complex64(8.753272678814991E-4f, 0.00803472101688385f);
        value = a.invert();
        Assertions.assertEquals(expValue, value);

        // ---------- sub-case 4 ------------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
        value = a.invert();
        Assertions.assertTrue(Float.isNaN(value.re));
        Assertions.assertTrue(Float.isNaN(value.im));

        // ---------- sub-case 5 ------------
        a = Complex64.NaN;
        value = a.invert();
        Assertions.assertTrue(Float.isNaN(value.re));
        Assertions.assertTrue(Float.isNaN(value.im));
    }


    @Test
    void conjTestCase() {
        // --------- sub-case 1 -----------
        a = new Complex64(0, 0);
        expValue = new Complex64(0, 0);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 2 -----------
        a = new Complex64(14.234f, 0);
        expValue = new Complex64(14.234f, 0);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 3 -----------
        a = new Complex64(1.451f, -9.3f);
        expValue = new Complex64(1.451f, 9.3f);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 4 -----------
        a = new Complex64(24, Float.POSITIVE_INFINITY);
        expValue = new Complex64(24, Float.NEGATIVE_INFINITY);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 5 -----------
        a = new Complex64(123.3f, Float.NEGATIVE_INFINITY);
        expValue = new Complex64(123.3f, Float.POSITIVE_INFINITY);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 6 -----------
        a = Complex64.NaN;
        value = a.conj();
        Assertions.assertTrue(Float.isNaN(value.re));
        Assertions.assertTrue(Float.isNaN(value.im));
    }


    @Test
    void sgnTestCase() {
        // --------- sub-case 1 -----------
        a = new Complex64(0);
        expValue = new Complex64(0);
        value = Complex64.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 2 -----------
        a = new Complex64(1.23f);
        expValue = new Complex64(1);
        value = Complex64.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 3 -----------
        a = new Complex64(-32974.234f);
        expValue = new Complex64(-1);
        value = Complex64.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 4 -----------
        a = new Complex64(1.4f, 13.4f);
        expValue = a.div((float) a.mag());
        value = Complex64.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 5 -----------
        a = new Complex64(-13.13f, 4141.2f);
        expValue = a.div((float) a.mag());
        value = Complex64.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- sub-case 6 -----------
        a = new Complex64(Float.POSITIVE_INFINITY, 4141.2f);
        value = Complex64.sgn(a);
        Assertions.assertEquals(0, value.im);
        Assertions.assertTrue(Float.isNaN(value.re));

        // --------- sub-case 7 -----------
        a = Complex64.NaN;
        value = Complex64.sgn(a);
        Assertions.assertTrue(Float.isNaN(value.im));
        Assertions.assertTrue(Float.isNaN(value.re));
    }
}
