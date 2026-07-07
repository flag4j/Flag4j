package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorRepeatTests {

    static CVector a;
    static Complex128[] aEntries;
    static CMatrix exp;
    static Complex128[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[]{new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)};
        a = new CVector(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.repeat(5, 0));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[]{new Complex128(1), new Complex128(0, 1)};
        a = new CVector(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(1), new Complex128(0, 1)},
                {new Complex128(1), new Complex128(0, 1)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- sub-case 3 ----------------------
        assertThrows(NegativeArraySizeException.class, ()-> a.repeat(-1, 0));
        assertThrows(LinearAlgebraException.class, ()-> a.repeat(13, -2));
        assertThrows(LinearAlgebraException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[]{new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)};
        a = new CVector(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)},
                {new Complex128(20.24, 1), new Complex128(-0.1451), new Complex128(93.1, 515.3),
                        new Complex128(0, -9.245), new Complex128(234.1), new Complex128(1800.24, -9923001.4)}
        };
        exp = new CMatrix(expEntries).T();

        assertEquals(exp, a.repeat(5, 1));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[]{new Complex128(1), new Complex128(0, 1)};
        a = new CVector(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(1), new Complex128(0, 1)},
                {new Complex128(1), new Complex128(0, 1)}
        };
        exp = new CMatrix(expEntries).T();

        assertEquals(exp, a.repeat(2, 1));
    }
}
