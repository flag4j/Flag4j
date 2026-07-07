package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64ExponentialTest {
    float a, b;
    Complex64 aComplex, bComplex;
    Complex64 expResult, actResult;

    @Test
    void powerOneDoubleTestCase() {
        // ------------ sub-case 1 ---------------
        a = 4;
        bComplex = new Complex64(6, 9);
        expResult = new Complex64(4079.525f, -367.0057067871094f);
        actResult = Complex64.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        a = 7.243867f;
        bComplex = new Complex64(-4.3f, 13.45f);
        expResult = new Complex64( 1.4114029E-5f, 2.000083914026618E-4f);
        actResult = Complex64.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        a = Float.NaN;
        bComplex = new Complex64(-4.3f, 13.45f);
        actResult = Complex64.pow(a, bComplex);
        Assertions.assertTrue(Float.isNaN(actResult.re));
        Assertions.assertTrue(Float.isNaN(actResult.im));

        // ------------ sub-case 4 ---------------
        a = 4.545f;
        bComplex = new Complex64(2.34f);
        expResult = new Complex64((float) Math.pow(a, 2.34f));
        actResult = Complex64.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void powTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex64(5);
        bComplex = new Complex64(3);
        expResult = new Complex64((float) Math.pow(5, 3));
        actResult = Complex64.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex64(3.4113f);
        bComplex = new Complex64(-6.133f, 1.3f);
        expResult = new Complex64(-1.3164215E-5f, 5.388559657149017E-4f);
        actResult = Complex64.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex64(5, 1.34f);
        bComplex = new Complex64(3, 4);
        expResult = new Complex64(22.98758f, 42.894046783447266f);
        actResult = Complex64.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        aComplex = new Complex64(-8.4f, 2.234f);
        bComplex = new Complex64(1.65901f, -4.192436f);
        expResult = new Complex64(-2644103.5f, 5805822.5f);
        actResult = Complex64.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void expTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex64(5);
        expResult = new Complex64((float) Math.exp(5));
        actResult = Complex64.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex64(5, 1.34f);
        expResult = new Complex64(33.949924f, 144.47792053222656f);
        actResult = Complex64.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex64(-23.23f, -13.32f);
        expResult = new Complex64(5.9455524E-11f, -5.579295359048331E-11f);
        actResult = Complex64.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void lnTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex64(1);
        expResult = new Complex64(0);
        actResult = Complex64.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex64(-1);
        expResult = new Complex64((float) 0, (float) Math.PI);
        actResult = Complex64.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex64(0);
        expResult = new Complex64(Float.NEGATIVE_INFINITY);
        actResult = Complex64.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        aComplex = new Complex64(146.1417912f);
        expResult = new Complex64(4.984577323028071f);
        actResult = Complex64.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        aComplex = new Complex64(142.18623f, -92.394356f);
        expResult = new Complex64(5.133259841229789f, -0.5762432330428644f);
        actResult = Complex64.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 ---------------
        aComplex = new Complex64(-8.5464f, -9.72352f);
        expResult = new Complex64(2.5607536790655163f, -2.2918540198902058f);
        actResult = Complex64.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void lnDoubleTestCase() {
        // ------------ sub-case 1 ---------------
        a = 1;
        expResult = new Complex64(0);
        actResult = Complex64.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        a = (float) Math.E;
        expResult = new Complex64(0.99999994f);
        actResult = Complex64.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        a = 0;
        expResult = new Complex64(Float.NEGATIVE_INFINITY);
        actResult = Complex64.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        a = 146.1417912f;
        expResult = new Complex64(4.984577323028071f);
        actResult = Complex64.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        a = -1;
        expResult = new Complex64((float) 0, (float) Math.PI);
        actResult = Complex64.ln(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex64(1);
        expResult = new Complex64(0);
        actResult = Complex64.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex64(-1);
        expResult = new Complex64(0, 1.364376425743103f);
        actResult = Complex64.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex64(0);
        expResult = new Complex64(Float.NEGATIVE_INFINITY);
        actResult = Complex64.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        aComplex = new Complex64(146.1417912f);
        expResult = new Complex64(2.164774426011174f);
        actResult = Complex64.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        aComplex = new Complex64(142.18623f, -92.394356f);
        expResult = new Complex64(2.2293463f, -0.25025925040245056f);
        actResult = Complex64.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 ---------------
        aComplex = new Complex64(-8.5464f, -9.72352f);
        expResult = new Complex64(1.1121211f, -0.9953395128250122f);
        actResult = Complex64.log(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logDoubleTestCase() {
        // ------------ sub-case 1 ---------------
        a = 1;
        expResult = new Complex64(0);
        actResult = Complex64.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        a = -1;
        expResult = new Complex64(0, 1.364376425743103f);
        actResult = Complex64.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        a = 0;
        expResult = new Complex64(Float.NEGATIVE_INFINITY);
        actResult = Complex64.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        a = 146.1417912f;
        expResult = new Complex64(2.164774426011174f);
        actResult = Complex64.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        a = -984.593465f;
        expResult = new Complex64( 2.9932568f, 1.364376425743103f);
        actResult = Complex64.log(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logDoubleBaseDoubleTestCase() {
        // ------------ sub-case 1 ------------
        a = 10;
        b = 12.23423f;
        expResult = new Complex64(1.0875766408496945f);
        actResult = Complex64.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ------------
        a = 985.343242f;
        b = 34.532f;
        expResult = new Complex64((float) (Math.log(b)/Math.log(a)));
        actResult = Complex64.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ------------
        a = -985.343242f;
        b = 34.532f;
        expResult = new Complex64(0.4254609128939375f, -0.1939107511743641f);
        actResult = Complex64.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ------------
        a = 98.4715f;
        b = -0.3096712f;
        expResult = new Complex64(-0.25540384666493776f, 0.6844776272773743f);
        actResult = Complex64.log(a, b);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logBaseDoubleTestCase() {
        // ------------ sub-case 1 ------------
        a = 2;
        bComplex = new Complex64(14.32f, 785.234981f);
        expResult = new Complex64(9.61722f,2.239873170852661f);
        actResult = Complex64.log(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ------------
        a = -42;
        bComplex = new Complex64(0.23423f, -18.343f);
        expResult = new Complex64(0.25081712f,-0.627661943435669f);
        actResult = Complex64.log(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ------------
        a = -23.123f;
        bComplex = new Complex64(-123.34f, 895);
        expResult = new Complex64(1.3551072366805246f,-0.8117131305115636f);
        actResult = Complex64.log(a, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logBaseTestCase() {
        // ------------ sub-case 1 ------------
        aComplex = new Complex64(93.23487f, -6.32465f);
        bComplex = new Complex64(-345.2f, 14.556f);
        expResult = new Complex64(1.2776989f, 0.7021597623825073f);
        actResult = Complex64.log(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ------------
        aComplex = new Complex64(12.1843f);
        bComplex = new Complex64(0);
        expResult = new Complex64(Float.NEGATIVE_INFINITY);
        actResult = Complex64.log(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }
}
