package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128SqrtTest {
    double a;
    Complex128 aComplex;
    Complex128 expResult, actResult;

    @Test
    void sqrtDoubleTestCase() {
        // ------------- sub-case 1 -------------
        a = 1;
        expResult = new Complex128(1);
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 2 -------------
        a = 4;
        expResult = new Complex128(2);
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 3 -------------
        a = 2;
        expResult = Complex128.ROOT_TWO;
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);


        // ------------- sub-case 4 -------------
        a = 3;
        expResult = Complex128.ROOT_THREE;
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 5 -------------
        a = 763.3422;
        expResult = new Complex128(Math.sqrt(a));
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        a = 0;
        expResult = new Complex128(0);
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        a = -1;
        expResult = new Complex128(0, 1);
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        a = -56.3947;
        expResult = new Complex128(0, Math.sqrt(56.3947));
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 7 -------------
        a = -0.0;
        expResult = new Complex128(0, -0.0);
        actResult = Complex128.sqrt(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void sqrtTestCase() {
        // ------------- sub-case 1 -------------
        aComplex = new Complex128(1);
        expResult = new Complex128(1);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 2 -------------
        aComplex = new Complex128(4);
        expResult = new Complex128(2);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 3 -------------
        aComplex = new Complex128(2);
        expResult = Complex128.ROOT_TWO;
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);


        // ------------- sub-case 4 -------------
        aComplex = new Complex128(3);
        expResult = Complex128.ROOT_THREE;
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 5 -------------
        aComplex = new Complex128(763.3422);
        expResult = new Complex128(Math.sqrt(763.3422));
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        aComplex = new Complex128(0);
        expResult = new Complex128(0);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        aComplex = new Complex128(-1);
        expResult = new Complex128(0, 1);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        aComplex = new Complex128(-56.3947);
        expResult = new Complex128(0, Math.sqrt(56.3947));
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 7 -------------
        aComplex = new Complex128(-0.0);
        expResult = new Complex128(0, -0.0);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 7 -------------
        aComplex = new Complex128(14.3, 7683.453);
        expResult = new Complex128(62.0393677722319, 61.92401112313579);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 8 -------------
        aComplex = new Complex128(-84.3453, 32.337847);
        expResult = new Complex128(1.7301266590439461, 9.345514338779571);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 9 -------------
        aComplex = new Complex128(0.34534, -9753246.45756);
        expResult = new Complex128(2208.307814017331, -2208.307735826237);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 10 -------------
        aComplex = new Complex128(-74.2346, -634.2146);
        expResult = new Complex128(16.797466883551945, -18.87828100500743);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);
    }
}
