package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorCrossProductTests {

    double[] aEntries = {1.0, 5.6, -9.355};
    Vector a = new Vector(aEntries);

    @Test
    void realCrossTestCase() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // --------------------- sub-case 1 ---------------------
        bEntries = new double[]{45.6, 7.9, -0.2345};
        b = new Vector(bEntries);
        expEntries = new double[]{72.59130000000002, -426.3535, -247.45999999999998};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- sub-case 2 ---------------------
        bEntries = new double[]{0, 0, 0};
        b = new Vector(bEntries);
        expEntries = new double[]{0.0, -0.0, 0.0};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- sub-case 3 ---------------------
        bEntries = new double[]{0, 0, 1};
        b = new Vector(bEntries);
        expEntries = new double[]{5.6, -1.0, 0.0};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- sub-case 4 ---------------------
        bEntries = new double[]{3, 1, Double.POSITIVE_INFINITY};
        b = new Vector(bEntries);
        expEntries = new double[]{Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -15.799999999999997};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));
    }
}
