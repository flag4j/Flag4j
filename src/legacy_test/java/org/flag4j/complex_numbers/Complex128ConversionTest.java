package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128ConversionTest {
    Complex128 n;
    double[] expPolar, actPolar;
    Complex128 expRect, actRect;
    double[] polar;

    /*
        Note: These test take into consideration precision errors from double floating point errors.
     */

    @Test
    void toPolarTestCase() {
        // --------------- sub-case 1 ---------------
        n = new Complex128(0);
        expPolar = new double[]{0, 0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 2 ---------------
        n = new Complex128(1, 3);
        expPolar = new double[]{Math.sqrt(10), Math.atan(3)};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 3 ---------------
        n = new Complex128(2.42, -1.35);
        expPolar = new double[]{2.771082820848197, -0.5088510437828061};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 4 ---------------
        n = new Complex128(1, 1);
        expPolar = new double[]{Math.sqrt(2), Math.PI/4.0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 5 ---------------
        n = new Complex128(-Math.sqrt(3.0)/2.0, -1.0/2.0);
        expPolar = new double[]{0.9999999999999999, -5.0*Math.PI/6.0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);
    }


    @Test
    void fromPolarTestCase() {
        // --------------- sub-case 1 ---------------
        expRect = new Complex128(0);
        polar = new double[]{0, 0};
        actRect = Complex128.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 2 ---------------
        expRect = new Complex128(1, 3);
        polar = new double[]{Math.sqrt(10), Math.atan(3)};
        actRect = Complex128.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 3 ---------------
        expRect = new Complex128(2.42, -1.3499999999999999);
        polar = new double[]{2.771082820848197, -0.5088510437828061};
        actRect = Complex128.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 4 ---------------
        expRect = new Complex128(1.0000000000000002, 1);
        polar = new double[]{Math.sqrt(2), Math.PI/4.0};
        actRect = Complex128.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 5 ---------------
        expRect = new Complex128(-0.8660254037844387, -0.49999999999999994);
        polar = new double[]{1, -5.0*Math.PI/6.0};
        actRect = Complex128.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);
    }
}
