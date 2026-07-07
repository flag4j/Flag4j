package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.VectorNorms;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorNormTest {

    double exp;
    double[] aEntries;
    Vector a;

    @Test
    void pNormTestCase() {
        // --------------------- sub-case 1 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 12.395642431310288;

        assertEquals(exp, VectorNorms.norm(a.data, 2));

        // --------------------- sub-case 2 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 18.81343;

        assertEquals(exp, VectorNorms.norm(a.data, 1));

        // --------------------- sub-case 3 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 9.234000000000005;

        assertEquals(exp, VectorNorms.norm(a.data, 234.5));

        // --------------------- sub-case 6 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 9.234;

        assertEquals(exp, VectorNorms.norm(a.data, Double.POSITIVE_INFINITY));
    }

    @Test
    void infNormTestCase() {
        // --------------------- sub-case 1 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 12.395642431310288;

        assertEquals(exp, VectorNorms.norm(a.data));

        // --------------------- sub-case 2 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234, 20243234.235, 1119.234, 5.14, -8.234};
        a = new Vector(aEntries);
        exp = 2.0243234265946947E7;

        assertEquals(exp, VectorNorms.norm(a.data));
    }
}
