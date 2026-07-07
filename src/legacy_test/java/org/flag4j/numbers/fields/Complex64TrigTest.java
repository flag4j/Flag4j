package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

// Note, all tests account for floating point precision errors.
class Complex64TrigTest {
    double a;
    Complex64 aComplex;
    Complex64 expResult, actResult;

    @Test
    void sinTestCase() {
        // ------------ sub-case 1 --------------
        aComplex = new Complex64((float) Math.PI);
        expResult = new Complex64(-8.742278E-8f);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 --------------
        aComplex = new Complex64((float) Math.PI/2);
        expResult = new Complex64(1);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 --------------
        aComplex = new Complex64((float) (3*Math.PI/2));
        expResult = new Complex64(-1);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 --------------
        aComplex = new Complex64(0);
        expResult = new Complex64(0);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 --------------
        aComplex = new Complex64((float) (-600*Math.PI));
        expResult = new Complex64(2.5747626E-5f);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 --------------
        aComplex = new Complex64((float) Math.PI/4);
        expResult = new Complex64(0.7071067811865475f);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(63425.234432673f);
        expResult = new Complex64(0.37064958f);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(355.34f, (float) Math.PI);
        expResult = new Complex64( -3.8660564f, -10.887526512145996f);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(-2.3f, 8.099867543f);
        expResult = new Complex64( -1228.1888f, -1097.3673095703125f);
        actResult = Complex64.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void cosTestCase() {
        // ------------ sub-case 1 --------------
        aComplex = new Complex64((float) Math.PI);
        expResult = new Complex64(-1);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 --------------
        aComplex = new Complex64((float) Math.PI/2);
        expResult = new Complex64(-4.371139E-8f);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 --------------
        aComplex = new Complex64((float) (3*Math.PI/2));
        expResult = new Complex64(1.1924881E-8f);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 --------------
        aComplex = new Complex64(0);
        expResult = new Complex64(1);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 --------------
        aComplex = new Complex64((float) (-600*Math.PI));
        expResult = new Complex64(1);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 --------------
        aComplex = new Complex64((float) Math.PI/4);
        expResult = new Complex64((float) Math.sqrt(2)/2);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(63425.234432673f);
        expResult = new Complex64(-0.9287728f);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(355.34f, (float) Math.PI);
        expResult = new Complex64( -10.928267f, -3.851644277572632f);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(-2.3f, 8.099867543f);
        expResult = new Complex64( -1097.3676f, -1228.1885986328125f);
        actResult = Complex64.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void tanTestCase() {
        // ------------ sub-case 1 --------------
        aComplex = new Complex64((float) Math.PI);
        expResult = new Complex64(8.742278E-8f);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 --------------
        aComplex = new Complex64(0);
        expResult = new Complex64(0);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 --------------
        aComplex = new Complex64((float) (-600*Math.PI));
        expResult = new Complex64(2.5747626E-5f);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 --------------
        aComplex = new Complex64((float) Math.PI/4);
        expResult = new Complex64(0.9999999999999999f);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 --------------
        aComplex = new Complex64(63425.234432673f);
        expResult = new Complex64(-0.39907455f);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 --------------
        aComplex = new Complex64(355.34f, (float) Math.PI);
        expResult = new Complex64( 0.002341809f, 0.9970974326133728f);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex64(-2.3f, 8.099867543f);
        expResult = new Complex64( 1.587595E-7f, 1.0000000206720314f);
        actResult = Complex64.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }
}
