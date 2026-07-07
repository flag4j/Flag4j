package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorElementwiseTests {

    double[] aEntries, expEntries;
    Vector a, exp, act;

    @Test
    void sqrtTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[]{1.3345, -99.243, 100.234, 225};
        expEntries = new double[]{Math.sqrt(1.3345), Math.sqrt(-99.243), Math.sqrt(100.234), Math.sqrt(225)};

        a = new Vector(aEntries);
        exp = new Vector(expEntries);
        act = a.sqrt();

        for(int i=0; i<exp.size; i++) {
            if(Double.isNaN(exp.get(i))) {
                assertTrue(Double.isNaN(exp.get(i)));
            } else {
                assertEquals(exp.get(i), act.get(i));
            }
        }
    }


    @Test
    void absTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[]{1.3345, -99.243, 100.234, 225, -0.0};
        expEntries = new double[]{Math.abs(1.3345), Math.abs(-99.243), Math.abs(100.234), Math.abs(225), Math.abs(-0.0)};

        a = new Vector(aEntries);
        exp = new Vector(expEntries);
        act = a.abs();

        assertEquals(exp, act);
    }


    @Test
    void recipTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[]{1.3345, -99.243, 100.234, 225, -0.0};
        expEntries = new double[]{1.0/1.3345, 1.0/-99.243, 1.0/100.234, 1.0/225, 1.0/-0.0};

        a = new Vector(aEntries);
        exp = new Vector(expEntries);
        act = a.recip();

        assertEquals(exp, act);
    }
}
