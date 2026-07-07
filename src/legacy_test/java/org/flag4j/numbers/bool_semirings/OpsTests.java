package org.flag4j.numbers.bool_semirings;

import org.flag4j.numbers.BoolSemiringElement;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class OpsTests {

    @Test
    void orTestCase() {
        BoolSemiringElement a;
        BoolSemiringElement b;
        BoolSemiringElement exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));

        // --------------- sub-case 4 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));
    }


    @Test
    void xorTestCase() {
        BoolSemiringElement a;
        BoolSemiringElement b;
        BoolSemiringElement exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.xor(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.xor(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.xor(b));

        // --------------- sub-case 4 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.xor(b));
    }


    @Test
    void andTestCase() {
        BoolSemiringElement a;
        BoolSemiringElement b;
        BoolSemiringElement exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));

        // --------------- sub-case 4 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));
    }


    @Test
    void notTestCase() {
        BoolSemiringElement a;
        BoolSemiringElement exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiringElement(true);
        exp = new BoolSemiringElement(false);

        assertEquals(exp, a.not());

        // --------------- sub-case 2 ---------------
        a = new BoolSemiringElement(false);
        exp = new BoolSemiringElement(true);

        assertEquals(exp, a.not());
    }


    @Test
    void equalsCompareTestCase() {
        BoolSemiringElement a;
        BoolSemiringElement b;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(true);

        assertTrue(a.equals(b));
        assertTrue(a.equals(a));
        assertFalse(a.equals(null));
        assertEquals(0, a.compareTo(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(false);

        assertTrue(a.equals(b));
        assertTrue(a.equals(a));
        assertFalse(a.equals(null));
        assertEquals(0, a.compareTo(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiringElement(true);
        b = new BoolSemiringElement(false);

        assertFalse(a.equals(b));
        assertTrue(a.compareTo(b) > 0);

        // --------------- sub-case 4 ---------------
        a = new BoolSemiringElement(false);
        b = new BoolSemiringElement(true);

        assertFalse(a.equals(b));
        assertTrue(a.compareTo(b) < 0);
    }
}
