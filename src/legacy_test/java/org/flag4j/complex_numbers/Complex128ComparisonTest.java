package org.flag4j.complex_numbers;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128ComparisonTest {
    Complex128 a;
    Complex128 b;
    Object obj;
    Integer integer;
    Double Double;
    double d;
    int compare;
    int expCompare;

    @Test
    void equalsTestCase() {
        // ------------ sub-case 1 ---------------
        a = new Complex128(5.123, 331.4);
        obj = new Complex128(5.123, 331.4);
        Assertions.assertEquals(a, obj);

        // ------------ sub-case 2 ---------------
        a = new Complex128(5.123, 331.4);
        b = new Complex128(5.123, 331.4);
        Assertions.assertEquals(a, b);

        // ------------ sub-case 3 ---------------
        a = new Complex128(5.123, 331.4);
        b = new Complex128(5.123, 331.4000000000001);
        Assertions.assertNotEquals(a, b);

        // ------------ sub-case 4 ---------------
        a = new Complex128(5.123, 331.4);
        b = new Complex128(5.123, 331.4000000000001);
        Assertions.assertNotEquals(a, b);

        // ------------ sub-case 5 ---------------
        a = new Complex128(-5.123);
        Double = -5.123;
        Assertions.assertEquals(a, new Complex128(Double));

        // ------------ sub-case 6 ---------------
        a = new Complex128(-5.123, 1);
        Double = -5.123;
        Assertions.assertNotEquals(a, new Complex128(Double));

        // ------------ sub-case 7 ---------------
        a = new Complex128(2);
        integer = 2;
        Assertions.assertEquals(a, new Complex128(integer));

        // ------------ sub-case 8 ---------------
        a = new Complex128(2);
        integer = 0;
        Assertions.assertNotEquals(a, new Complex128(integer));

        // ------------ sub-case 9 ---------------
        a = new Complex128(2.09124);
        d = 2.09124;
        Assertions.assertTrue(a.equals(d));

        // ------------ sub-case 10 ---------------
        a = new Complex128(7);
        d = 7;
        Assertions.assertTrue(a.equals(d));

        // ------------ sub-case 11 ---------------
        a = new Complex128(3.024);
        d = 2.09124;
        Assertions.assertFalse(a.equals(d));

        // ------------ sub-case 12 ---------------
        a = new Complex128(9);
        d = 7;
        Assertions.assertFalse(a.equals(d));
    }


    @Test
    void hashCodeTestCase() {
        a = new Complex128(52344.13, -941.339615);
        int hashPrime = 31;
        int expHash = 17;
        expHash = hashPrime*expHash + java.lang.Double.hashCode(a.re);
        expHash = hashPrime*expHash + java.lang.Double.hashCode(a.im);

        Assertions.assertEquals(expHash, a.hashCode());
    }


    @Test
    void compareTestCase() {
        // -------- sub-case 1 ----------
        a = new Complex128(400);
        b = new Complex128(3, 1.21);
        expCompare = 1;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);

        // -------- sub-case 2 ----------
        a = new Complex128(4);
        b = new Complex128(3, 3.4214);
        expCompare = -1;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);

        // -------- sub-case 3 ----------
        a = new Complex128(34.13, 11.44);
        b = new Complex128(11.44, 34.13);
        expCompare = 0;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);
    }
}
