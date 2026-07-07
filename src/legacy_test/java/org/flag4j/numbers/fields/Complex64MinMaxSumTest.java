package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64MinMaxSumTest {

    Complex64 n1, n2, n3, n4, n5;
    Complex64 sum, min, max;
    Complex64 expSum, expMin, expMax;
    int expArg, arg;

    @Test
    void sumTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1.3f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(0, -2);

        sum = Complex64.sum(n1, n2, n3, n4);

        expSum = new Complex64(93.13f + 1.3f + -4.2e-8f + 0,
                -6456.331f + 7.5f + 0 + -2);

        Assertions.assertEquals(sum, expSum);

        // ------------ sub-case 2 ------------
        sum = Complex64.sum();
        expSum = Complex64.ZERO;
        Assertions.assertEquals(sum, expSum);
    }


    @Test
    void minTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1.3f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(0, -2);

        min = Complex64.min(n1, n2, n3, n4);

        expMin = new Complex64(-4.2e-8f);

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        min = Complex64.min();
        Assertions.assertNull(min);
    }


    @Test
    void minReal() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1.3f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(0, -2);
        n5 = new Complex64(-9347, 100);

        min = Complex64.minRe(n1, n2, n3, n4, n5);

        expMin = new Complex64(-9347);

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        min = Complex64.minRe();
        Assertions.assertTrue(Double.isNaN(min.re));
    }


    @Test
    void argminTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1.3f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(0, -2);

        arg = Complex64.argmin(n1, n2, n3, n4);

        expArg = 2;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex64.argmin();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void argminRealTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1.3f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(-122, -2);
        n5 = new Complex64(0, -2);

        arg = Complex64.argminReal(n1, n2, n3, n4);

        expArg = 3;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex64.argminReal();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void maxTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(1.3f, 7.5f);
        n2 = new Complex64(93.13f, -6456.331f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(0, -2);

        max = Complex64.max(n1, n2, n3, n4);

        expMax = new Complex64(93.13f, -6456.331f);

        Assertions.assertEquals(expMax, max);

        // ------------ sub-case 2 ------------
        max = Complex64.max();
        Assertions.assertNull(max);
    }


    @Test
    void maxReal() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1.3f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(104.43f, -2);
        n5 = new Complex64(0, 100);

        max = Complex64.maxRe(n1, n2, n3, n4, n5);

        expMax = new Complex64(104.43f);

        Assertions.assertEquals(expMax, max);

        // ------------ sub-case 2 ------------
        max = Complex64.maxRe();
        Assertions.assertTrue(Double.isNaN(max.re));
    }


    @Test
    void argmaxTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(1.3f, 7.5f);
        n2 = new Complex64(93.13f, -6456.331f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(0, -2);

        arg = Complex64.argmax(n1, n2, n3, n4);

        expArg = 1;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex64.argmax();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void argmaxRealTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex64(93.13f, -6456.331f);
        n2 = new Complex64(1e10f, 7.5f);
        n3 = new Complex64(-4.2e-8f);
        n4 = new Complex64(-122, -2);
        n5 = new Complex64(0, -2);

        arg = Complex64.argmaxReal(n1, n2, n3, n4);

        expArg = 1;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex64.argmaxReal();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }
}
