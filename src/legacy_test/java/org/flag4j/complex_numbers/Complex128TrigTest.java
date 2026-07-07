package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

// Note, all tests account for floating point precision errors.
class Complex128TrigTest {
    double a;
    Complex128 aComplex;
    Complex128 expResult, actResult;

    @Test
    void sinTestCase() {
        // ------------ sub-case 1 --------------
        aComplex = new Complex128(Math.PI);
        expResult = new Complex128(1.2246467991473532E-16);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 --------------
        aComplex = new Complex128(Math.PI/2);
        expResult = new Complex128(1);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 --------------
        aComplex = new Complex128(3*Math.PI/2);
        expResult = new Complex128(-1);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 --------------
        aComplex = new Complex128(0);
        expResult = new Complex128(0);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 --------------
        aComplex = new Complex128(-600*Math.PI);
        expResult = new Complex128(1.019005173792452E-13);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 --------------
        aComplex = new Complex128(Math.PI/4);
        expResult = new Complex128(0.7071067811865475);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(63425.234432673);
        expResult = new Complex128(0.3705960190335554);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(355.34, Math.PI);
        expResult = new Complex128( -3.866096216916842, -10.887511808457687);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(-2.3, 8.099867543);
        expResult = new Complex128( -1228.1884278205741, -1097.367075991759);
        actResult = Complex128.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void cosTestCase() {
        // ------------ sub-case 1 --------------
        aComplex = new Complex128(Math.PI);
        expResult = new Complex128(-1);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 --------------
        aComplex = new Complex128(Math.PI/2);
        expResult = new Complex128(6.123233995736766E-17);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 --------------
        aComplex = new Complex128(3*Math.PI/2);
        expResult = new Complex128(-1.8369701987210297E-16);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 --------------
        aComplex = new Complex128(0);
        expResult = new Complex128(1);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 --------------
        aComplex = new Complex128(-600*Math.PI);
        expResult = new Complex128(1);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 --------------
        aComplex = new Complex128(Math.PI/4);
        expResult = new Complex128(Math.sqrt(2)/2);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(63425.234432673);
        expResult = new Complex128(-0.9287941594758661);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(355.34, Math.PI);
        expResult = new Complex128( -10.928251497079271, -3.8516837048969297);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(-2.3, 8.099867543);
        expResult = new Complex128( -1097.367278259398, -1228.188201439873);
        actResult = Complex128.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void tanTestCase() {
        // ------------ sub-case 1 --------------
        aComplex = new Complex128(Math.PI);
        expResult = new Complex128(-1.2246467991473532E-16);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 2 --------------
        aComplex = new Complex128(0);
        expResult = new Complex128(0);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 3 --------------
        aComplex = new Complex128(-600*Math.PI);
        expResult = new Complex128(1.019005173792452E-13);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 4 --------------
        aComplex = new Complex128(Math.PI/4);
        expResult = new Complex128(0.9999999999999999);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 5 --------------
        aComplex = new Complex128(63425.234432673);
        expResult = new Complex128(-0.39900769751038145);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 6 --------------
        aComplex = new Complex128(355.34, Math.PI);
        expResult = new Complex128( 0.0023418361407621328, 0.9970974609682846);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ sub-case 7 --------------
        aComplex = new Complex128(-2.3, 8.099867543);
        expResult = new Complex128( 1.831579637156239E-7, 1.0000000206720314);
        actResult = Complex128.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }
}
