package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorRepeatTests {

    static CooVector a;
    static double[] aEntries;
    static CooMatrix exp;
    static double[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[]{0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45};
        a = new Vector(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45}
        };
        exp = new Matrix(expEntries).toCoo();

        assertEquals(exp, a.repeat(4, 0));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new Vector(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45},
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45}
        };
        exp = new Matrix(expEntries).toCoo();

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new Vector(aEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 0));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[]{0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45};
        a = new Vector(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45}
        };
        exp = new Matrix(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(4, 1));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new Vector(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45},
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45}
        };
        exp = new Matrix(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(2, 1));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new Vector(aEntries).T().toCoo();

        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 1));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }
}
