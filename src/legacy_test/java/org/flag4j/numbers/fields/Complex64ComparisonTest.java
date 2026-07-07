package org.flag4j.numbers.fields;

import org.flag4j.numbers.Complex64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex64ComparisonTest {
    Complex64 a;
    Complex64 b;
    Object obj;
    Integer integer;
    Float bFloat;
    float d;
    int compare;
    int expCompare;

    @Test
    void equalsTestCase() {
        // ------------ sub-case 1 ---------------
        a = new Complex64(5.123f, 331.4f);
        obj = new Complex64(5.123f, 331.4f);
        Assertions.assertEquals(a, obj);

        // ------------ sub-case 2 ---------------
        a = new Complex64(5.123f, 331.4f);
        b = new Complex64(5.123f, 331.4f);
        Assertions.assertEquals(a, b);

        // ------------ sub-case 3 ---------------
        a = new Complex64(5.123f, 331.4f);
        b = new Complex64(5.123f, 331.4001f);
        Assertions.assertNotEquals(a, b);

        // ------------ sub-case 4 ---------------
        a = new Complex64(5.123f, 331.4f);
        b = new Complex64(5.123f, 331.401f);
        Assertions.assertNotEquals(a, b);

        // ------------ sub-case 5 ---------------
        a = new Complex64(-5.123f);
        bFloat = -5.123f;
        Assertions.assertEquals(a, new Complex64(bFloat));

        // ------------ sub-case 6 ---------------
        a = new Complex64(-5.123f, 1);
        bFloat = -5.123f;
        Assertions.assertNotEquals(a, new Complex64(bFloat));

        // ------------ sub-case 7 ---------------
        a = new Complex64(2);
        integer = 2;
        Assertions.assertEquals(a, new Complex64(integer));

        // ------------ sub-case 8 ---------------
        a = new Complex64(2);
        integer = 0;
        Assertions.assertNotEquals(a, new Complex64(integer));

        // ------------ sub-case 9 ---------------
        a = new Complex64(2.09124f);
        d = 2.09124f;
        Assertions.assertTrue(a.equals(d));

        // ------------ sub-case 10 ---------------
        a = new Complex64(7);
        d = 7;
        Assertions.assertTrue(a.equals(d));

        // ------------ sub-case 11 ---------------
        a = new Complex64(3.024f);
        d = 2.09124f;
        Assertions.assertFalse(a.equals(d));

        // ------------ sub-case 12 ---------------
        a = new Complex64(9);
        d = 7;
        Assertions.assertFalse(a.equals(d));
    }


    @Test
    void hashCodeTestCase() {
        a = new Complex64(52344.13f, -941.339615f);
        int hashPrime = 31;
        int expHash = 17;
        expHash = hashPrime*expHash + java.lang.Float.hashCode(a.re);
        expHash = hashPrime*expHash + java.lang.Float.hashCode(a.im);

        Assertions.assertEquals(expHash, a.hashCode());
    }


    @Test
    void compareTestCase() {
        // -------- sub-case 1 ----------
        a = new Complex64(400);
        b = new Complex64(3, 1.21f);
        expCompare = 1;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);

        // -------- sub-case 2 ----------
        a = new Complex64(4);
        b = new Complex64(3, 3.4214f);
        expCompare = -1;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);

        // -------- sub-case 3 ----------
        a = new Complex64(34.13f, 11.44f);
        b = new Complex64(11.44f, 34.13f);
        expCompare = 0;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);
    }
}
