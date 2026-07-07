package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorNormTests {

    static Complex128[] aEntries;
    static CVector a;
    double expNorm;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void normTestCase() {
        // ------------------ sub-case 1 ------------------
        expNorm = 6126.638392078558;
        assertEquals(expNorm, a.norm());
    }


    @Test
    void pNormTestCase() {
        // ------------------ sub-case 1 ------------------
        expNorm = 6208.346603991548;
        assertEquals(expNorm, a.norm(1));

        // ------------------ sub-case 2 ------------------
        expNorm = 6126.347178284364;
        assertEquals(expNorm, a.norm(4.15), 1.0e-12);

        // ------------------ sub-case 3 ------------------
        expNorm = 6126.347172780367;
        assertEquals(expNorm, a.norm(45), 1.0e-12);

        // ------------------ sub-case 4 ------------------
        expNorm = 6126.347172780367;
        assertEquals(expNorm, a.norm(Double.POSITIVE_INFINITY));

        // ------------------ sub-case 5 ------------------
        expNorm = 0.009241438243709998;
        assertEquals(expNorm, a.norm(-1), 1.0e-12);

        // ------------------ sub-case 6 ------------------
        expNorm = 0.009257;
        assertEquals(expNorm, a.norm(Double.NEGATIVE_INFINITY));
    }


    @Test
    void infNormTestCase() {
        // ------------------ sub-case 1 ------------------
        expNorm = 6126.347172780367;
        assertEquals(expNorm, a.norm(Double.POSITIVE_INFINITY));
    }
}
