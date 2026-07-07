package org.flag4j.numbers.bool_semirings;

import org.flag4j.numbers.BoolSemiringElement;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConstructorConstantsTests {


    @Test
    void constructorTestCase() {
        BoolSemiringElement act;
        boolean exp;

        // --------------- sub-case 1 ---------------
        act = new BoolSemiringElement(true);
        assertTrue(act.getValue());

        // --------------- sub-case 2 ---------------
        act = new BoolSemiringElement(false);
        assertFalse(act.getValue());

        // --------------- sub-case 3 ---------------
        act = new BoolSemiringElement(1);
        assertTrue(act.getValue());

        // --------------- sub-case 4 ---------------
        act = new BoolSemiringElement(0);
        assertFalse(act.getValue());

        // --------------- sub-case 5 ---------------
        assertThrows(IllegalArgumentException.class, () -> new BoolSemiringElement(2));
        assertThrows(IllegalArgumentException.class, () -> new BoolSemiringElement(-1));
        assertThrows(IllegalArgumentException.class, () -> new BoolSemiringElement(1002));
    }


    @Test
    void constantTestCase() {
        // --------------- sub-case 1 ---------------
        assertTrue(BoolSemiringElement.TRUE.getValue());
        assertTrue(BoolSemiringElement.ONE.getValue());
        assertTrue(new BoolSemiringElement(0).getOne().getValue());
        assertTrue(new BoolSemiringElement(1).isOne());
        assertTrue(new BoolSemiringElement(0).isZero());
        assertFalse(BoolSemiringElement.ZERO.getValue());
        assertFalse(BoolSemiringElement.FALSE.getValue());
        assertFalse(new BoolSemiringElement(1).getZero().getValue());
    }
}
