package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Complex64ConstructorTest {
    Complex64 a, b;
    float expRe;
    float expIm;


    @Test
    void defaultConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = Complex64.ZERO;
        expRe=0;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);
    }


    @Test
    void reConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = new Complex64(1023);
        expRe=1023;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 2 ----------
        a = new Complex64(9991.02331f);
        expRe=9991.02331f;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 3 ----------
        a = new Complex64(Float.NEGATIVE_INFINITY);
        expRe=Float.NEGATIVE_INFINITY;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 4 ----------
        a = new Complex64(Float.NaN);
        expRe=Float.NaN;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 4 ----------
        a = new Complex64(Float.POSITIVE_INFINITY);
        expRe=Float.POSITIVE_INFINITY;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 5 ----------
        a = new Complex64(-9356e7f);
        expRe=-9356e7f;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);
    }


    @Test
    void reImConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = new Complex64(1023);
        expRe=1023;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 2 ----------
        a = new Complex64(9991.02331f, 1304);
        expRe=9991.02331f;
        expIm=1304;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 3 ----------
        a = new Complex64(494, Float.NEGATIVE_INFINITY);
        expRe=494;
        expIm=Float.NEGATIVE_INFINITY;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 4 ----------
        a = new Complex64(Float.NaN, Float.NaN);
        expRe=Float.NaN;
        expIm=Float.NaN;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);

        // ---------- sub-case 4 ----------
        a = new Complex64(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
        expRe=Float.POSITIVE_INFINITY;
        expIm=0;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);


        // ---------- sub-case 5 ----------
        a = new Complex64(1.2391f, -9356e7f);
        expRe=1.2391f;
        expIm=-9356e7f;

        assertEquals(a.re, expRe);
        assertEquals(a.re, expRe);
    }


    @Test
    void copyConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = Complex64.ZERO;
        expRe = a.re;
        expIm = a.im;
        b = a;

        assertEquals(expRe, b.re);
        assertEquals(expIm, b.im);

        // ---------- sub-case 2 ----------
        a = new Complex64(1023.343f);
        expRe = a.re;
        expIm = a.im;
        b = a;

        assertEquals(expRe, b.re);
        assertEquals(expIm, b.im);


        // ---------- sub-case 3 ----------
        a = new Complex64(Float.POSITIVE_INFINITY, 03.3210003f);
        expRe = a.re;
        expIm = a.im;
        b = a;

        assertEquals(expRe, b.re);
        assertEquals(expIm, b.im);
    }


    @Test
    void stringConstructorTestCase() {
        // ---------- sub-case 1 ----------
        a = new Complex64("3");
        expRe = 3;
        expIm = 0;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 2 ----------
        a = new Complex64("009.343442");
        expRe = 9.343442f;
        expIm = 0;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);


        // ---------- sub-case 3 ----------
        a = new Complex64("3934.22 - i");
        expRe = 3934.22f;
        expIm = -1;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);


        // ---------- sub-case 4 ----------
        a = new Complex64("94 - i");
        expRe = 94;
        expIm = -1;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 5 ----------
        a = new Complex64("0.001 - 0.313i");
        expRe = 0.001f;
        expIm = -0.313f;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 6 ----------
        a = new Complex64("  2394   +    i  ");
        expRe = 2394;
        expIm = 1;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 7 ----------
        a = new Complex64("2i");
        expRe = 0;
        expIm = 2;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 8 ----------
        a = new Complex64("009.324i");
        expRe = 0;
        expIm = 9.324f;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 9 ----------
        a = new Complex64(" -1254i   ");
        expRe = 0;
        expIm = -1254;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 9 ----------
        a = new Complex64("0");
        expRe = 0;
        expIm = 0;

        assertEquals(expRe, a.re);
        assertEquals(expIm, a.im);

        // ---------- sub-case 10 ----------
        assertThrows(RuntimeException.class, () -> new Complex64("sdf"));

        // ---------- sub-case 11 ----------
        assertThrows(RuntimeException.class, () -> new Complex64("1.023*i"));

        // ---------- sub-case 12 ----------
        assertThrows(RuntimeException.class, () -> new Complex64("1.13 - 2ei"));
    }
}
