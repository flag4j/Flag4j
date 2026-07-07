package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128ExponentialTest {
    double a, b;
    Complex128 aComplex, bComplex;
    Complex128 expResult, actResult;

    @Test
    void powerOneDoubleTestCase() {
        // ------------ sub-case 1 ---------------
        a = 4;
        bComplex = new Complex128(6, 9);
        expResult = new Complex128(4079.524813717923, -367.0058504434513);
        actResult = Complex128.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        a = 7.243867;
        bComplex = new Complex128(-4.3, 13.45);
        expResult = new Complex128( 1.4114030703306451E-5, 2.0000844699788985E-4);
        actResult = Complex128.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        a = Double.NaN;
        bComplex = new Complex128(-4.3, 13.45);
        actResult = Complex128.pow(a, bComplex);
        Assertions.assertTrue(Double.isNaN(actResult.re));
        Assertions.assertTrue(Double.isNaN(actResult.im));

        // ------------ sub-case 4 ---------------
        a = 4.545;
        bComplex = new Complex128(2.34);
        expResult = new Complex128(Math.pow(a, 2.34));
        actResult = Complex128.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void powTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex128(5);
        bComplex = new Complex128(3);
        expResult = new Complex128(Math.pow(5, 3));
        actResult = Complex128.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex128(3.4113);
        bComplex = new Complex128(-6.133, 1.3);
        expResult = new Complex128(-1.3164257696727862E-5, 5.388558542480906E-4);
        actResult = Complex128.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex128(5, 1.34);
        bComplex = new Complex128(3, 4);
        expResult = new Complex128(22.987590087916242, 42.89401981891137);
        actResult = Complex128.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        aComplex = new Complex128(-8.4, 2.234);
        bComplex = new Complex128(1.65901, -4.192436);
        expResult = new Complex128(-2644100.3854441093, 5805824.714161132);
        actResult = Complex128.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void expTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex128(5);
        expResult = new Complex128(Math.exp(5));
        actResult = Complex128.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex128(5, 1.34);
        expResult = new Complex128(33.94992686043801, 144.4779161705263);
        actResult = Complex128.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex128(-23.23, -13.32);
        expResult = new Complex128(5.945547809631131E-11, -5.579294435010688E-11);
        actResult = Complex128.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void lnTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex128(1);
        expResult = new Complex128(0);
        actResult = Complex128.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex128(-1);
        expResult = new Complex128(0, Math.PI);
        actResult = Complex128.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex128(0);
        expResult = new Complex128(Double.NEGATIVE_INFINITY);
        actResult = Complex128.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        aComplex = new Complex128(146.1417912);
        expResult = new Complex128(4.984577323028071);
        actResult = Complex128.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        aComplex = new Complex128(142.18623, -92.394356);
        expResult = new Complex128(5.133259841229789, -0.5762432330428644);
        actResult = Complex128.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 ---------------
        aComplex = new Complex128(-8.5464, -9.72352);
        expResult = new Complex128(2.5607536790655163, -2.2918540198902058);
        actResult = Complex128.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void lnDoubleTestCase() {
        // ------------ sub-case 1 ---------------
        a = 1;
        expResult = new Complex128(0);
        actResult = Complex128.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        a = Math.E;
        expResult = new Complex128(1);
        actResult = Complex128.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        a = 0;
        expResult = new Complex128(Double.NEGATIVE_INFINITY);
        actResult = Complex128.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        a = 146.1417912;
        expResult = new Complex128(4.984577323028071);
        actResult = Complex128.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        a = -1;
        expResult = new Complex128(0, Math.PI);
        actResult = Complex128.ln(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logTestCase() {
        // ------------ sub-case 1 ---------------
        aComplex = new Complex128(1);
        expResult = new Complex128(0);
        actResult = Complex128.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        aComplex = new Complex128(-1);
        expResult = new Complex128(0, 1.3643763538418412);
        actResult = Complex128.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        aComplex = new Complex128(0);
        expResult = new Complex128(Double.NEGATIVE_INFINITY);
        actResult = Complex128.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        aComplex = new Complex128(146.1417912);
        expResult = new Complex128(2.164774426011174);
        actResult = Complex128.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        aComplex = new Complex128(142.18623, -92.394356);
        expResult = new Complex128(2.2293464232216595, -0.25025925634460555);
        actResult = Complex128.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 ---------------
        aComplex = new Complex128(-8.5464, -9.72352);
        expResult = new Complex128(1.1121211923316043, -0.9953395541661019);
        actResult = Complex128.log(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logDoubleTestCase() {
        // ------------ sub-case 1 ---------------
        a = 1;
        expResult = new Complex128(0);
        actResult = Complex128.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ---------------
        a = -1;
        expResult = new Complex128(0, 1.3643763538418412);
        actResult = Complex128.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ---------------
        a = 0;
        expResult = new Complex128(Double.NEGATIVE_INFINITY);
        actResult = Complex128.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 ---------------
        a = 146.1417912;
        expResult = new Complex128(2.164774426011174);
        actResult = Complex128.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 ---------------
        a = -984.593465;
        expResult = new Complex128( 2.9932569489221543, 1.3643763538418412);
        actResult = Complex128.log(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logDoubleBaseDoubleTestCase() {
        // ------------ sub-case 1 ------------
        a = 10;
        b = 12.23423;
        expResult = new Complex128(1.0875766408496945);
        actResult = Complex128.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ------------
        a = 985.343242;
        b = 34.532;
        expResult = new Complex128(Math.log(b)/Math.log(a));
        actResult = Complex128.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ------------
        a = -985.343242;
        b = 34.532;
        expResult = new Complex128(0.4254609128939375, -0.1939107511743641);
        actResult = Complex128.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ------------
        a = 98.4715;
        b = -0.3096712;
        expResult = new Complex128(-0.25540384666493776, 0.6844775649445937);
        actResult = Complex128.log(a, b);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logBaseDoubleTestCase() {
        // ------------ sub-case 1 ------------
        a = 2;
        bComplex = new Complex128(14.32, 785.234981);
        expResult = new Complex128(9.617220494534935,2.2398731644571033);
        actResult = Complex128.log(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ------------
        a = -42;
        bComplex = new Complex128(0.23423, -18.343);
        expResult = new Complex128(0.25081712428118125,-0.6276618980737457);
        actResult = Complex128.log(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 ------------
        a = -23.123;
        bComplex = new Complex128(-123.34, 895);
        expResult = new Complex128(1.3551072366805246,-0.8117131305115636);
        actResult = Complex128.log(a, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logBaseTestCase() {
        // ------------ sub-case 1 ------------
        aComplex = new Complex128(93.23487, -6.32465);
        bComplex = new Complex128(-345.2, 14.556);
        expResult = new Complex128(1.277698990918928, 0.7021597301463025);
        actResult = Complex128.log(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 ------------
        aComplex = new Complex128(12.1843);
        bComplex = new Complex128(0);
        expResult = new Complex128(Double.NEGATIVE_INFINITY);
        actResult = Complex128.log(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }
}
