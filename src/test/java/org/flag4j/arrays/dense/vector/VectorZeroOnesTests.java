package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorZeroOnesTests {

    double[] aEntries;
    Vector A;

    @Test
    void zerosTestCase(){
        // -------------------- sub-case 1 --------------------
        aEntries = new double[34];
        A = new Vector(aEntries);
        assertTrue(A.isAllZeros());

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{0.0, 0, -0.0};
        A = new Vector(aEntries);
        assertTrue(A.isAllZeros());

        // -------------------- sub-case 3 --------------------
        aEntries = new double[2345];
        aEntries[123] = 3.324;
        A = new Vector(aEntries);
        assertFalse(A.isAllZeros());
    }


    @Test
    void onesTestCase(){
        // -------------------- sub-case 1 --------------------
        aEntries = new double[34];
        Arrays.fill(aEntries, 1);
        A = new Vector(aEntries);
        assertTrue(A.isAllOnes());

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.0, 1, 1.0};
        A = new Vector(aEntries);
        assertTrue(A.isAllOnes());

        // -------------------- sub-case 3 --------------------
        aEntries = new double[2345];
        Arrays.fill(aEntries, 1);
        aEntries[123] = 3.324;
        A = new Vector(aEntries);
        assertFalse(A.isAllOnes());
    }
}
