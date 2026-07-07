package org.flag4j.complex_numbers;

import org.flag4j.io.PrintOptions;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class Complex128ToStringTest {
    Complex128 a;
    String expStr;

    @BeforeAll
    static void setUp() {
        // Ensure print options are set to default before any test is run.
        PrintOptions.resetAll();
    }


    @AfterEach
    void tearDown() {
        // Ensure print options are reset after each test is run.
        PrintOptions.resetAll();
    }


    @Test
    void realToStringTestCase() {
        // ---------- sub-case 1 ------------
        a = new Complex128(1);
        expStr = "1";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 2 ------------
        a = new Complex128(93.234);
        expStr = "93.234";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 3 ------------
        a = new Complex128(-1.23e-5);
        expStr = "-1.23E-5";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 4 ------------
        a = Complex128.ZERO;
        expStr = "0";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));
    }


    @Test
    void imaginaryToStringTestCase() {
        // ---------- sub-case 1 ------------
        a = new Complex128(0, 1);
        expStr = "i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 2 ------------
        a = new Complex128(0, 93.234);
        expStr = "93.234i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 3 ------------
        a = new Complex128(0, -1.23e-5);
        expStr = "-1.23E-5i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 4 ------------
        a = new Complex128(0, -1);
        expStr = "-i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 5 ------------
        a = new Complex128(0, 24);
        expStr = "24i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 6 ------------
        a = new Complex128(0, -56);
        expStr = "-56i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));
    }


    @Test
    void complexToStringTestCase() {
        // ---------- sub-case 1 ------------
        a = new Complex128(234.3, 1);
        expStr = "234.3+i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 2 ------------
        a = new Complex128(1.341, 93.234);
        expStr = "1.341+93.234i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 3 ------------
        a = new Complex128(-9.324, -1.23e-5);
        expStr = "-9.324-1.23E-5i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));

        // ---------- sub-case 4 ------------
        a = new Complex128(994.242, -1);
        expStr = "994.242-i";
        Assertions.assertEquals(expStr, a.toString());
        Assertions.assertEquals(expStr.length(), Complex128.length(a));
    }
}
