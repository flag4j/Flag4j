package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64RoundTest {

    Complex64 n, expRound, actRound;
    boolean expNearZero, actNearZero;

    @Test
    void roundTestCase() {
        // -------------- sub-case 1 --------------
        n = new Complex64(0);
        expRound = new Complex64(0);
        actRound = Complex64.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 2 --------------
        n = new Complex64(13, 4);
        expRound = new Complex64(13, 4);
        actRound = Complex64.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 3 --------------
        n = new Complex64(-0.133f, 13.413f);
        expRound = new Complex64(0, 13);
        actRound = Complex64.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 4 --------------
        n = new Complex64(-0.893f, 16.5f);
        expRound = new Complex64(-1, 17);
        actRound = Complex64.round(n);
        Assertions.assertEquals(expRound, actRound);


        // -------------- sub-case 5 --------------
        n = new Complex64(9.3E10f, 0.1993312f);
        expRound = new Complex64(9.3E10f, 0);
        actRound = Complex64.round(n);
        Assertions.assertEquals(expRound, actRound);
    }


    @Test
    void roundDecimalsTestCase() {
        // -------------- sub-case 1 --------------
        n = new Complex64(0);
        expRound = new Complex64(0);
        actRound = Complex64.round(n, 1);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 2 --------------
        n = new Complex64(13, 4);
        expRound = new Complex64(13, 4);
        actRound = Complex64.round(n, 2);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 3 --------------
        n = new Complex64(-0.133f, 13.41562f);
        expRound = new Complex64(-0.13f, 13.42f);
        actRound = Complex64.round(n, 2);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 4 --------------
        n = new Complex64(-0.89242993f, 16.99999999f);
        expRound = new Complex64(-0.89243f, 17);
        actRound = Complex64.round(n, 6);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 5 --------------
        n = new Complex64(9.3E10f, 0.1993312f);
        expRound = new Complex64(9.3E10f, 0.1993f);
        actRound = Complex64.round(n, 4);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 9 --------------
        n = new Complex64(8234.5f, 123.34f);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex64.round(n, -1));

        // -------------- sub-case 10 --------------
        n = new Complex64(8234.5f, 123.34f);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex64.round(n, -100));
    }


    @Test
    void nearZeroTestCase() {
        // -------------- sub-case 1 --------------
        n = new Complex64(24.25f, 0.3422f);
        expNearZero = false;
        actNearZero = Complex64.nearZero(n, 0.001f);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 2 --------------
        n = new Complex64(13, 4);
        expNearZero = true;
        actNearZero = Complex64.nearZero(n, 15);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 3 --------------
        n = new Complex64(-0.133f, 13.41562f);
        expNearZero = false;
        actNearZero = Complex64.nearZero(n, 1);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 4 --------------
        n = new Complex64(-0.0001231f, 0.0000001313f);
        expNearZero = true;
        actNearZero = Complex64.nearZero(n, 0.0005f);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 5 --------------
        n = new Complex64(9.3E10f, 0.1993312f);
        expNearZero = false;
        actNearZero = Complex64.nearZero(n, 13100);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 6 --------------
        n = new Complex64(Float.POSITIVE_INFINITY);
        expNearZero = false;
        actNearZero = Complex64.nearZero(n, 13100);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 7 --------------
        n = new Complex64(Float.NEGATIVE_INFINITY);
        expNearZero = false;
        actNearZero = Complex64.nearZero(n, 1000000);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 8 --------------
        n = new Complex64(Float.NaN);
        expNearZero = false;
        actNearZero = Complex64.nearZero(n, 13);
        Assertions.assertEquals(expRound, actRound);

        // -------------- sub-case 9 --------------
        n = new Complex64(8234.5f, 123.34f);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex64.nearZero(n, -1));

        // -------------- sub-case 10 --------------
        n = new Complex64(8234.5f, 123.34f);
        Assertions.assertThrows(IllegalArgumentException.class, () -> Complex64.nearZero(n, -100));
    }
}
