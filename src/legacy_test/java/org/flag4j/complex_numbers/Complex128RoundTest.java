package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128RoundTest {

    Complex128 n, expRound, actRound;
    boolean expNearZero, actNearZero;

    @Test
    void roundTestCase() {
        // -------------- sub-case 1 --------------
        n = new Complex128(0);
        expRound = new Complex128(0);
        actRound = Complex128.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 2 --------------
        n = new Complex128(13, 4);
        expRound = new Complex128(13, 4);
        actRound = Complex128.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 3 --------------
        n = new Complex128(-0.133, 13.413);
        expRound = new Complex128(0, 13);
        actRound = Complex128.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 4 --------------
        n = new Complex128(-0.893, 16.5);
        expRound = new Complex128(-1, 17);
        actRound = Complex128.round(n);
        Assertions.assertEquals(expRound, actRound);


        // -------------- sub-case 5 --------------
        n = new Complex128(9.3E10, 0.1993312);
        expRound = new Complex128(9.3E10, 0);
        actRound = Complex128.round(n);
        Assertions.assertEquals(expRound, actRound);
    }


    @Test
    void roundDecimalsTestCase() {
        // -------------- sub-case 1 --------------
        n = new Complex128(0);
        expRound = new Complex128(0);
        actRound = Complex128.round(n, 1);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 2 --------------
        n = new Complex128(13, 4);
        expRound = new Complex128(13, 4);
        actRound = Complex128.round(n, 2);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 3 --------------
        n = new Complex128(-0.133, 13.41562);
        expRound = new Complex128(-0.13, 13.42);
        actRound = Complex128.round(n, 2);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 4 --------------
        n = new Complex128(-0.89242993, 16.99999999);
        expRound = new Complex128(-0.89243, 17);
        actRound = Complex128.round(n, 6);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 5 --------------
        n = new Complex128(9.3E10, 0.1993312);
        expRound = new Complex128(9.3E10, 0.1993);
        actRound = Complex128.round(n, 4);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 9 --------------
        n = new Complex128(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex128.round(n, -1));

        // -------------- sub-case 10 --------------
        n = new Complex128(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex128.round(n, -100));
    }


    @Test
    void nearZeroTestCase() {
        // -------------- sub-case 1 --------------
        n = new Complex128(24.25, 0.3422);
        expNearZero = false;
        actNearZero = Complex128.nearZero(n, 0.001);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 2 --------------
        n = new Complex128(13, 4);
        expNearZero = true;
        actNearZero = Complex128.nearZero(n, 15);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 3 --------------
        n = new Complex128(-0.133, 13.41562);
        expNearZero = false;
        actNearZero = Complex128.nearZero(n, 1);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 4 --------------
        n = new Complex128(-0.0001231, 0.0000001313);
        expNearZero = true;
        actNearZero = Complex128.nearZero(n, 0.0005);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 5 --------------
        n = new Complex128(9.3E10, 0.1993312);
        expNearZero = false;
        actNearZero = Complex128.nearZero(n, 13100);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 6 --------------
        n = new Complex128(Double.POSITIVE_INFINITY);
        expNearZero = false;
        actNearZero = Complex128.nearZero(n, 13100);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 7 --------------
        n = new Complex128(Double.NEGATIVE_INFINITY);
        expNearZero = false;
        actNearZero = Complex128.nearZero(n, 1000000);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 8 --------------
        n = new Complex128(Double.NaN);
        expNearZero = false;
        actNearZero = Complex128.nearZero(n, 13);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 9 --------------
        n = new Complex128(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex128.nearZero(n, -1));

        // -------------- sub-case 10 --------------
        n = new Complex128(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex128.nearZero(n, -100));
    }
}
