package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Complex128ConstructorTest {
    Complex128 a, b;
    double expRe;
    double expIm;


    @Test
    void defaultConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = Complex128.ZERO;
        expRe=0;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);
    }


    @Test
    void reConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = new Complex128(1023);
        expRe=1023;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 2 ----------
        a = new Complex128(9991.02331);
        expRe=9991.02331;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 3 ----------
        a = new Complex128(Double.NEGATIVE_INFINITY);
        expRe=Double.NEGATIVE_INFINITY;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 4 ----------
        a = new Complex128(Double.NaN);
        expRe=Double.NaN;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 4 ----------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expRe=Double.POSITIVE_INFINITY;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 5 ----------
        a = new Complex128(-9356e7);
        expRe=-9356e7;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);
    }


    @Test
    void reImConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = new Complex128(1023);
        expRe=1023;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 2 ----------
        a = new Complex128(9991.02331, 1304);
        expRe=9991.02331;
        expIm=1304;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 3 ----------
        a = new Complex128(494, Double.NEGATIVE_INFINITY);
        expRe=494;
        expIm=Double.NEGATIVE_INFINITY;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 4 ----------
        a = new Complex128(Double.NaN, Double.NaN);
        expRe=Double.NaN;
        expIm=Double.NaN;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 4 ----------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        expRe=Double.POSITIVE_INFINITY;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 5 ----------
        a = new Complex128(1.2391, -9356e7);
        expRe=1.2391;
        expIm=-9356e7;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);
    }


    @Test
    void copyConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = Complex128.ZERO;
        expRe = a.re;
        expIm = a.im;
        b = a;

        assertEquals(expRe, b.re);
        assertEquals(expIm, b.im);

        // ---------- sub-case 2 ----------
        a = new Complex128(1023.343);
        expRe = a.re;
        expIm = a.im;
        b = a;

        assertEquals(expRe, b.re);
        assertEquals(expIm, b.im);


        // ---------- sub-case 3 ----------
        a = new Complex128(Double.POSITIVE_INFINITY, 03.3210003);
        expRe = a.re;
        expIm = a.im;
        b = a;

        assertEquals(expRe, b.re);
        assertEquals(expIm, b.im);
    }


    @Test
    void stringConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = new Complex128("3");
        expRe = 3;
        expIm = 0;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 2 ----------
        a = new Complex128("009.343442");
        expRe = 9.343442;
        expIm = 0;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);


        // ---------- sub-case 3 ----------
        a = new Complex128("3934.22- i");
        expRe = 3934.22;
        expIm = -1;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);


        // ---------- sub-case 4 ----------
        a = new Complex128("94 - i");
        expRe = 94;
        expIm = -1;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 5 ----------
        a = new Complex128("0.001 - 0.313i");
        expRe = 0.001;
        expIm = -0.313;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 6 ----------
        a = new Complex128("  2394   +    i  ");
        expRe = 2394;
        expIm = 1;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 7 ----------
        a = new Complex128("2i");
        expRe = 0;
        expIm = 2;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 8 ----------
        a = new Complex128("009.324i");
        expRe = 0;
        expIm = 9.324;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 9 ----------
        a = new Complex128(" -1254i   ");
        expRe = 0;
        expIm = -1254;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 9 ----------
        a = new Complex128("0");
        expRe = 0;
        expIm = 0;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 10 ----------
        assertThrows(RuntimeException.class, () -> new Complex128("sdf"));

        // ---------- sub-case 11 ----------
        assertThrows(RuntimeException.class, () -> new Complex128("1.023*i"));

        // ---------- sub-case 12 ----------
        assertThrows(RuntimeException.class, () -> new Complex128("1.13- 2ei"));
    }
}
