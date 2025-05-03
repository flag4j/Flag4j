package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorPerpParallelTests {

    double[] aEntries, bEntries;
    Vector a, b;

    @Test
    void isParallelTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new double[]{1, 2, -5, 8};
        a = new Vector(aEntries);
        bEntries = new double[]{4, 8, -20, 32};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // ----------------------- sub-case 2 -----------------------
        aEntries = new double[]{0, 0, -5.234, 8};
        a = new Vector(aEntries);
        bEntries = new double[]{0, 0, -20.936, 32};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // ----------------------- sub-case 3 -----------------------
        aEntries = new double[]{0, 0, 0, 0};
        a = new Vector(aEntries);
        bEntries = new double[]{14, 15568.34435, -20.936, 32};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // ----------------------- sub-case 4 -----------------------
        aEntries = new double[]{0, 0, 0, 1.56};
        a = new Vector(aEntries);
        bEntries = new double[]{0, 0, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // ----------------------- sub-case 5 -----------------------
        aEntries = new double[]{5.234, 56.62, 9.24};
        a = new Vector(aEntries);
        bEntries = new double[]{4.1*5.234, 4.1*56.62, 4.1*9.24, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // ----------------------- sub-case 6 -----------------------
        aEntries = new double[]{5.234};
        a = new Vector(aEntries);
        bEntries = new double[]{0.000234};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // ----------------------- sub-case 7 -----------------------
        aEntries = new double[]{1, 2, -5, 8};
        a = new Vector(aEntries);
        bEntries = new double[]{4, 8, -20, 31};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));
    }


    @Test
    void isPerpTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new double[]{3, 4};
        a = new Vector(aEntries);
        bEntries = new double[]{-8, 6};
        b = new Vector(bEntries);

        assertTrue(a.isPerp(b));

        // ----------------------- sub-case 2 -----------------------
        aEntries = new double[]{0, 0, 1.4415, 135.23, -9234, 0, 1.3};
        a = new Vector(aEntries);
        bEntries = new double[]{1.45, -9.234, 0, 0, 0, 36.7, 0};
        b = new Vector(bEntries);

        assertTrue(a.isPerp(b));

        // ----------------------- sub-case 3 -----------------------
        aEntries = new double[]{1.3, 0, 1.4415, 135.23, -9234, 0, 1.3};
        a = new Vector(aEntries);
        bEntries = new double[]{1.45, -9.234, 0, 0, 0, 36.7, 0};
        b = new Vector(bEntries);

        assertFalse(a.isPerp(b));

        // ----------------------- sub-case 4 -----------------------
        aEntries = new double[]{0, 0, 0, 0, 0, 0, 0};
        a = new Vector(aEntries);
        bEntries = new double[]{1.45, -9.234, 0, 0, 0, 36.7, 0};
        b = new Vector(bEntries);

        assertTrue(a.isPerp(b));

        // ----------------------- sub-case 5 -----------------------
        aEntries = new double[]{1, 0, 0, 1, 0, 1};
        a = new Vector(aEntries);
        bEntries = new double[]{0, 1, 1, 0, 1};
        b = new Vector(bEntries);

        assertFalse(a.isPerp(b));
    }
}
