package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


class Complex64PropertiesTest {
    Complex64 a;
    boolean expResult;
    boolean result;

    @Test
    void isIntTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = true;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = true;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(23.5f);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);
    }

    @Test
    void isDoubleTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = true;
        result = a.isFloat();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = true;
        result = a.isFloat();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = false;
        result = a.isFloat();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expResult = true;
        result = a.isFloat();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = true;
        result = a.isFloat();

        assertEquals(expResult, result);
    }

    @Test
    void isNaNTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 6 --------------
        a = new Complex64(223.54268f, Float.NaN);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 7 --------------
        a = new Complex64(223.54268f,12434.33f);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- sub-case 8 --------------
        a = new Complex64(Float.NaN,12434.33f);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);


        // ------------- sub-case 9 --------------
        a = new Complex64(Float.NaN,Float.NaN);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);
    }

    @Test
    void isFiniteTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 6 --------------
        a = new Complex64(223.54268f, Float.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 7 --------------
        a = new Complex64(223.54268f,12434.33f);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- sub-case 8 --------------
        a = new Complex64(Float.NaN,12434.33f);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);


        // ------------- sub-case 9 --------------
        a = new Complex64(Float.NaN,Float.NaN);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);


        // ------------- sub-case 10 --------------
        a = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);
    }


    @Test
    void isInfiniteTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 6 --------------
        a = new Complex64(223.54268f, Float.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 7 --------------
        a = new Complex64(223.54268f,12434.33f);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- sub-case 8 --------------
        a = new Complex64(Float.NaN,12434.33f);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);


        // ------------- sub-case 9 --------------
        a = new Complex64(Float.NaN,Float.NaN);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);


        // ------------- sub-case 10 --------------
        a = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);
    }


    @Test
    void isRealTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 6 --------------
        a = new Complex64(223.54268f, Float.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 7 --------------
        a = new Complex64(223.54268f,12434.33f);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 8 --------------
        a = new Complex64(Float.NaN,12434.33f);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 9 --------------
        a = new Complex64(Float.NaN,Float.NaN);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- sub-case 10 --------------
        a = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);
    }


    @Test
    void isImaginaryTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(0, -342);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 6 --------------
        a = new Complex64(223.54268f, Float.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 7 --------------
        a = new Complex64(0,12434.33f);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 8 --------------
        a = new Complex64(Float.NaN,12434.33f);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);


        // ------------- sub-case 9 --------------
        a = new Complex64(Float.NaN,Float.NaN);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);


        // ------------- sub-case 10 --------------
        a = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- sub-case 11 --------------
        a = new Complex64(0, Float.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);
    }


    @Test
    void isComplexTestCase() {
        // ------------- sub-case 1 --------------
        a = new Complex64(5);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 2 --------------
        a = new Complex64(-4);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 3 --------------
        a = new Complex64(2,-1);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 4 --------------
        a = new Complex64(0, -342);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 5 --------------
        a = new Complex64(223.54268f);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 6 --------------
        a = new Complex64(223.54268f, Float.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 7 --------------
        a = new Complex64(0,12434.33f);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 8 --------------
        a = new Complex64(Float.NaN,12434.33f);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 9 --------------
        a = new Complex64(Float.NaN,Float.NaN);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 10 --------------
        a = new Complex64(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- sub-case 11 --------------
        a = new Complex64(0, Float.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);
    }

}
