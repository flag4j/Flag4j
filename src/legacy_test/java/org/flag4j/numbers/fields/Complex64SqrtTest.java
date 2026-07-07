package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64SqrtTest {
    float a;
    Complex64 aComplex;
    Complex64 expResult, actResult;

    @Test
    void sqrtDoubleTestCase() {
        // ------------- sub-case 1 -------------
        a = 1;
        expResult = new Complex64(1);
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 2 -------------
        a = 4;
        expResult = new Complex64(2);
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 3 -------------
        a = 2;
        expResult = Complex64.ROOT_TWO;
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);


        // ------------- sub-case 4 -------------
        a = 3;
        expResult = Complex64.ROOT_THREE;
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 5 -------------
        a = 763.3422f;
        expResult = new Complex64((float) Math.sqrt(a));
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        a = 0;
        expResult = new Complex64(0);
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        a = -1;
        expResult = new Complex64(0, 1);
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        a = -56.3947f;
        expResult = new Complex64(0, (float) Math.sqrt(56.3947f));
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 7 -------------
        a = -0.0f;
        expResult = new Complex64(0, -0.0f);
        actResult = Complex64.sqrt(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void sqrtTestCase() {
        // ------------- sub-case 1 -------------
        aComplex = new Complex64(1);
        expResult = new Complex64(1);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 2 -------------
        aComplex = new Complex64(4);
        expResult = new Complex64(2);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 3 -------------
        aComplex = new Complex64(2);
        expResult = Complex64.ROOT_TWO;
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);


        // ------------- sub-case 4 -------------
        aComplex = new Complex64(3);
        expResult = Complex64.ROOT_THREE;
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 5 -------------
        aComplex = new Complex64(763.3422f);
        expResult = new Complex64((float) Math.sqrt(763.3422f));
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        aComplex = new Complex64(0);
        expResult = new Complex64(0);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        aComplex = new Complex64(-1);
        expResult = new Complex64(0, 1);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 6 -------------
        aComplex = new Complex64(-56.3947f);
        expResult = new Complex64(0, (float) Math.sqrt(56.3947f));
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 7 -------------
        aComplex = new Complex64(-0.0f);
        expResult = new Complex64(0, -0.0f);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 7 -------------
        aComplex = new Complex64(14.3f, 7683.453f);
        expResult = new Complex64(62.0393677722319f, 61.92401112313579f);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 8 -------------
        aComplex = new Complex64(-84.3453f, 32.337847f);
        expResult = new Complex64(1.7301267f, 9.345514338779571f);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 9 -------------
        aComplex = new Complex64(0.34534f, -9753246.45756f);
        expResult = new Complex64(2208.307814017331f, -2208.307735826237f);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);

        // ------------- sub-case 10 -------------
        aComplex = new Complex64(-74.2346f, -634.2146f);
        expResult = new Complex64(16.797466883551945f, -18.87828100500743f);
        actResult = aComplex.sqrt();
        Assertions.assertEquals(expResult, actResult);
    }
}
