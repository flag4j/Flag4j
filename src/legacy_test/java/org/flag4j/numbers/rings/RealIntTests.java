package org.flag4j.numbers.rings;

import org.flag4j.numbers.RealI16;
import org.flag4j.numbers.RealI32;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RealIntTests {

    @Test
    void addTests() {
        RealI16 a16;
        RealI16 b16;
        RealI16 exp16;

        RealI32 a32;
        RealI32 b32;
        RealI32 exp32;

        // ------------------- sub-case 1 -------------------
        a16 = new RealI16((short) 1);
        b16 = new RealI16((short) 152);
        exp16 = new RealI16((short) 153);

        a32 = new RealI32(1);
        b32 = new RealI32(152);
        exp32 = new RealI32(153);

        assertEquals(exp16, a16.add(b16));
        assertEquals(exp32, a32.add(b32));

        // ------------------- sub-case 2 -------------------
        a16 = new RealI16((short) 15);
        b16 = new RealI16((short) 0);
        exp16 = new RealI16((short) 15);

        a32 = new RealI32(15);
        b32 = new RealI32(0);
        exp32 = new RealI32(15);

        assertEquals(exp16, a16.add(b16));
        assertEquals(exp32, a32.add(b32));

        // ------------------- sub-case 3 -------------------
        a16 = new RealI16((short) -71);
        b16 = new RealI16((short) 31);
        exp16 = new RealI16((short) (-71 + 31));

        a32 = new RealI32(-71);
        b32 = new RealI32(31);
        exp32 = new RealI32(-71 + 31);

        assertEquals(exp16, a16.add(b16));
        assertEquals(exp32, a32.add(b32));
    }


    @Test
    void subTests() {
        RealI16 a16;
        RealI16 b16;
        RealI16 exp16;

        RealI32 a32;
        RealI32 b32;
        RealI32 exp32;

        // ------------------- sub-case 1 -------------------
        a16 = new RealI16((short) 1);
        b16 = new RealI16((short) 152);
        exp16 = new RealI16((short) -151);

        a32 = new RealI32(1);
        b32 = new RealI32(152);
        exp32 = new RealI32(-151);

        assertEquals(exp16, a16.sub(b16));
        assertEquals(exp32, a32.sub(b32));

        // ------------------- sub-case 2 -------------------
        a16 = new RealI16((short) 15);
        b16 = new RealI16((short) 0);
        exp16 = new RealI16((short) 15);

        a32 = new RealI32(15);
        b32 = new RealI32(0);
        exp32 = new RealI32(15);

        assertEquals(exp16, a16.sub(b16));
        assertEquals(exp32, a32.sub(b32));

        // ------------------- sub-case 3 -------------------
        a16 = new RealI16((short) -71);
        b16 = new RealI16((short) 31);
        exp16 = new RealI16((short) (-71 - 31));

        a32 = new RealI32(-71);
        b32 = new RealI32(31);
        exp32 = new RealI32(-71 - 31);

        assertEquals(exp16, a16.sub(b16));
        assertEquals(exp32, a32.sub(b32));
    }


    @Test
    void multTests() {
        RealI16 a16;
        RealI16 b16;
        RealI16 exp16;

        RealI32 a32;
        RealI32 b32;
        RealI32 exp32;

        // ------------------- sub-case 1 -------------------
        a16 = new RealI16((short) 1);
        b16 = new RealI16((short) 152);
        exp16 = new RealI16((short) 152);

        a32 = new RealI32(1);
        b32 = new RealI32(152);
        exp32 = new RealI32(152);

        assertEquals(exp16, a16.mult(b16));
        assertEquals(exp32, a32.mult(b32));

        // ------------------- sub-case 2 -------------------
        a16 = new RealI16((short) 15);
        b16 = new RealI16((short) 0);
        exp16 = new RealI16((short) 0);

        a32 = new RealI32(15);
        b32 = new RealI32(0);
        exp32 = new RealI32(0);

        assertEquals(exp16, a16.mult(b16));
        assertEquals(exp32, a32.mult(b32));

        // ------------------- sub-case 3 -------------------
        a16 = new RealI16((short) -71);
        b16 = new RealI16((short) 31);
        exp16 = new RealI16((short) (-71 * 31));

        a32 = new RealI32(-71);
        b32 = new RealI32(31);
        exp32 = new RealI32(-71 * 31);

        assertEquals(exp16, a16.mult(b16));
        assertEquals(exp32, a32.mult(b32));
    }


    @Test
    void constantTestCase() {
        assertEquals(new RealI16((short) 1), RealI16.ONE);
        assertEquals(new RealI32(1), RealI32.ONE);
        assertEquals(new RealI16((short) 0), RealI16.ZERO);
        assertEquals(new RealI32(0), RealI32.ZERO);

        assertTrue(new RealI16((short) 1).isOne());
        assertFalse(new RealI16((short) 0).isOne());
        assertFalse(new RealI16((short) 0).isOne());
        assertTrue(new RealI16((short) 1).isOne());

        assertTrue(new RealI32(1).isOne());
        assertFalse(new RealI32(0).isOne());
        assertFalse(new RealI32(0).isOne());
        assertTrue(new RealI32(1).isOne());
    }


    @Test
    void sgnMagTestCase() {
        // Sgn tests
        assertEquals(RealI16.ONE, RealI16.sgn(new RealI16((short) 16)));
        assertEquals(RealI16.ONE, RealI16.sgn(new RealI16((short) 1)));
        assertEquals(RealI16.ZERO, RealI16.sgn(new RealI16((short) 0)));
        assertEquals(RealI16.NEGATIVE_ONE, RealI16.sgn(new RealI16((short) -1)));
        assertEquals(RealI16.NEGATIVE_ONE, RealI16.sgn(new RealI16((short) -251)));

        assertEquals(RealI32.ONE, RealI32.sgn(new RealI32(16)));
        assertEquals(RealI32.ONE, RealI32.sgn(new RealI32(1)));
        assertEquals(RealI32.ZERO, RealI32.sgn(new RealI32(0)));
        assertEquals(RealI32.NEGATIVE_ONE, RealI32.sgn(new RealI32(-1)));
        assertEquals(RealI32.NEGATIVE_ONE, RealI32.sgn(new RealI32(-251)));

        // Magnitude tests
        assertEquals(16.0, new RealI16((short) 16).mag());
        assertEquals(1.0, new RealI16((short) 1).mag());
        assertEquals(0.0, new RealI16((short) 0).mag());
        assertEquals(1.0, new RealI16((short) -1).mag());
        assertEquals(251.0, new RealI16((short) -251).mag());

        assertEquals(16.0, new RealI32(16).mag());
        assertEquals(1.0, new RealI32(1).mag());
        assertEquals(0.0, new RealI32(0).mag());
        assertEquals(1.0, new RealI32(-1).mag());
        assertEquals(251.0, new RealI32(-251).mag());
    }
}
