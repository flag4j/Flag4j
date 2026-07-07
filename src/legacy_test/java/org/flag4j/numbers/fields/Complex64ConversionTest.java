package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64ConversionTest {
    Complex64 n;
    float[] expPolar, actPolar;
    Complex64 expRect, actRect;
    float[] polar;

    /*
        Note: These test take into consideration precision errors from double floating point errors.
     */

    @Test
    void toPolarTestCase() {
        // --------------- sub-case 1 ---------------
        n = new Complex64(0);
        expPolar = new float[]{0, 0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 2 ---------------
        n = new Complex64(1, 3);
        expPolar = new float[]{(float) Math.sqrt(10), (float) Math.atan(3)};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 3 ---------------
        n = new Complex64(2.42f, -1.35f);
        expPolar = new float[]{2.771082820848197f, -0.5088510437828061f};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 4 ---------------
        n = new Complex64(1, 1);
        expPolar = new float[]{(float) Math.sqrt(2), (float) (Math.PI/4.0)};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- sub-case 5 ---------------
        n = new Complex64((float) (-Math.sqrt(3.0)/2.0), (float) (-1.0/2.0));
        expPolar = new float[]{0.9999999999999999f, (float) (-5.0*Math.PI/6.0)};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);
    }


    @Test
    void fromPolarTestCase() {
        // --------------- sub-case 1 ---------------
        expRect = new Complex64(0);
        polar = new float[]{0, 0};
        actRect = Complex64.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 2 ---------------
        expRect = new Complex64(1.0000001f, 3);
        polar = new float[]{(float) Math.sqrt(10), (float) Math.atan(3)};
        actRect = Complex64.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 3 ---------------
        expRect = new Complex64(2.42f, -1.3499999999999999f);
        polar = new float[]{2.771082820848197f, -0.5088510437828061f};
        actRect = Complex64.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 4 ---------------
        expRect = new Complex64(0.99999994f, 0.9999999403953552f);
        polar = new float[]{(float) Math.sqrt(2), (float) (Math.PI/4.0)};
        actRect = Complex64.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- sub-case 5 ---------------
        expRect = new Complex64(-0.8660254037844387f, -0.5000000596046448f);
        polar = new float[]{1, (float) (-5.0*Math.PI/6.0)};
        actRect = Complex64.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);
    }
}
