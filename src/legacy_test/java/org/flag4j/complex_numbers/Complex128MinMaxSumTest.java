package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128MinMaxSumTest {

    Complex128 n1, n2, n3, n4, n5;
    Complex128 sum, min, max;
    Complex128 expSum, expMin, expMax;
    int expArg, arg;

    @Test
    void sumTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1.3, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(0, -2);

        sum = Complex128.sum(n1, n2, n3, n4);

        expSum = new Complex128(93.13 + 1.3 + -4.2e-8 + 0,
                -6456.331 + 7.5 + 0 + -2);

        Assertions.assertEquals(sum, expSum);

        // ------------ sub-case 2 ------------
        sum = Complex128.sum();
        expSum = Complex128.ZERO;
        Assertions.assertEquals(sum, expSum);
    }


    @Test
    void minTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1.3, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(0, -2);

        min = Complex128.min(n1, n2, n3, n4);

        expMin = new Complex128(-4.2e-8);

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        min = Complex128.min();
        Assertions.assertNull(min);
    }


    @Test
    void minReal() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1.3, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(0, -2);
        n5 = new Complex128(-9347, 100);

        min = Complex128.minRe(n1, n2, n3, n4, n5);

        expMin = new Complex128(-9347);

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        min = Complex128.minRe();
        Assertions.assertTrue(Double.isNaN(min.re));
    }


    @Test
    void argminTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1.3, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(0, -2);

        arg = Complex128.argmin(n1, n2, n3, n4);

        expArg = 2;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex128.argmin();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void argminRealTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1.3, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(-122, -2);
        n5 = new Complex128(0, -2);

        arg = Complex128.argminReal(n1, n2, n3, n4);

        expArg = 3;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex128.argminReal();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void maxTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(1.3, 7.5);
        n2 = new Complex128(93.13, -6456.331);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(0, -2);

        max = Complex128.max(n1, n2, n3, n4);

        expMax = new Complex128(93.13, -6456.331);

        Assertions.assertEquals(expMax, max);

        // ------------ sub-case 2 ------------
        max = Complex128.max();
        Assertions.assertNull(max);
    }


    @Test
    void maxReal() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1.3, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(104.43, -2);
        n5 = new Complex128(0, 100);

        max = Complex128.maxRe(n1, n2, n3, n4, n5);

        expMax = new Complex128(104.43);

        Assertions.assertEquals(expMax, max);

        // ------------ sub-case 2 ------------
        max = Complex128.maxRe();
        Assertions.assertTrue(Double.isNaN(max.re));
    }


    @Test
    void argmaxTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(1.3, 7.5);
        n2 = new Complex128(93.13, -6456.331);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(0, -2);

        arg = Complex128.argmax(n1, n2, n3, n4);

        expArg = 1;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex128.argmax();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void argmaxRealTestCase() {
        // ------------ sub-case 1 ------------
        n1 = new Complex128(93.13, -6456.331);
        n2 = new Complex128(1e10, 7.5);
        n3 = new Complex128(-4.2e-8);
        n4 = new Complex128(-122, -2);
        n5 = new Complex128(0, -2);

        arg = Complex128.argmaxReal(n1, n2, n3, n4);

        expArg = 1;

        Assertions.assertEquals(expMin, min);

        // ------------ sub-case 2 ------------
        arg = Complex128.argmaxReal();
        expArg = -1;
        Assertions.assertEquals(expMin, min);
    }
}
