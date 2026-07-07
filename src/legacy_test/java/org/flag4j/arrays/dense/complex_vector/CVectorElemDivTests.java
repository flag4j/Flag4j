package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CVectorElemDivTests {
    static Complex128[] aEntries;
    static CVector a;
    Complex128[] expEntries;
    CVector exp;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(4.556, -85.2518), new Complex128(43.1, -99.34551),
                new Complex128(6915.66), new Complex128(0, 9.345)};
        a = new CVector(aEntries);
    }


    /**
     * Checks if all items of two complex vectors are equal where two NaNs are considered equal.
     * @param exp Expected complex vector.
     * @param act Actual complex vector.
     */
    static void assertEqualsNaN(CVector exp, CVector act) {
        assertEquals(exp.size, act.size);

        for(int i=0; i<exp.size; i++) {
            if(Double.isNaN(exp.get(i).re)) {
                assertTrue(Double.isNaN(act.get(i).re));
            } else {
                assertEquals(exp.get(i).re, act.get(i).re);
            }

            if(Double.isNaN(exp.get(i).im)) {
                assertTrue(Double.isNaN(act.get(i).im));
            } else {
                assertEquals(exp.get(i).im, act.get(i).im);
            }
        }
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ------------------- sub-case 1 -------------------
        bEntries = new double[]{2.455, -9.24, 0, 24.50001};
        b = new Vector(bEntries);
        expEntries = new Complex128[]{aEntries[0].div(bEntries[0]), aEntries[1].div(bEntries[1]),
                aEntries[2].div(bEntries[2]), aEntries[3].div(bEntries[3])};
        exp = new CVector(expEntries);

        assertEqualsNaN(exp, a.div(b));

        // ------------------- sub-case 2 -------------------
        bEntries = new double[]{2.455, -9.24};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.div(finalB));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        // ------------------- sub-case 1 -------------------
        bEntries = new Complex128[]{new Complex128(-0.00024), new Complex128(0, 85.234),
                new Complex128(0.00234, 15.6), new Complex128(-0.24, 662.115)};
        b = new CVector(bEntries);
        expEntries = new Complex128[]{aEntries[0].div(bEntries[0]), aEntries[1].div(bEntries[1]),
                aEntries[2].div(bEntries[2]), aEntries[3].div(bEntries[3])};
        exp = new CVector(expEntries);

        assertEqualsNaN(exp, a.div(b));

        // ------------------- sub-case 2 -------------------
        bEntries = new Complex128[]{new Complex128(0, 85.234),
                new Complex128(0.00234, 15.6), new Complex128(-0.24, 662.115)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.div(finalB));
    }
}
