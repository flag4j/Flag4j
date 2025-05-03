package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128GettersSettersTest {
    Complex128 num;
    long expReLong, expImLong;
    int expReInt, expImInt;
    float expReFloat, expImFloat;
    double expReDouble, expImDouble;


    @Test
    void gettersTestCase() {
        num = new Complex128(692.13, -9673.134);
        expReDouble = 692.13;
        expImDouble = -9673.134;

        Assertions.assertEquals(expReDouble, num.getReal());
        Assertions.assertEquals(expImDouble, num.getImag());
    }
}
