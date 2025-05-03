package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64GettersSettersTest {
    Complex64 num;
    long expReLong, expImLong;
    int expReInt, expImInt;
    float expReFloat, expImFloat;


    @Test
    void gettersTestCase() {
        num = new Complex64(692.13f, -9673.134f);
        expReFloat = 692.13f;
        expImFloat = -9673.134f;

        Assertions.assertEquals(expReFloat, num.getReal());
        Assertions.assertEquals(expImFloat, num.getImag());
    }
}
