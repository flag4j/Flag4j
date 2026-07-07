package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorAggregateTests {

    static Complex128[] aEntries;
    static CVector a;
    Complex128 expComplex;
    double exp;
    int[] expIndices;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
        new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void sumTestCase() {
        // ------------------ sub-case 1 ------------------
        expComplex = aEntries[0].add(aEntries[1]).add(aEntries[2]).add(aEntries[3]).add(aEntries[4]);
        assertEquals(expComplex, a.sum());
    }


    @Test
    void minTestCase() {
        // ------------------ sub-case 1 ------------------
        exp = -0.009257;
        assertEquals(new Complex128(exp), a.min());
        assertEquals(Math.abs(exp), a.minAbs());
    }


    @Test
    void maxTestCase() {
        // ------------------ sub-case 1 ------------------
        Complex128 expCm = new Complex128(1.455, 6126.347);
        assertEquals(expCm, a.max());
        assertEquals(expCm.mag(), a.maxAbs());
    }


    @Test
    void argminTestCase() {
        // ------------------ sub-case 1 ------------------
        expIndices = new int[]{4};
        assertArrayEquals(expIndices, a.argmin());
    }


    @Test
    void argmaxTestCase() {
        // ------------------ sub-case 1 ------------------
        expIndices = new int[]{0};
        assertArrayEquals(expIndices, a.argmax());
    }
}
