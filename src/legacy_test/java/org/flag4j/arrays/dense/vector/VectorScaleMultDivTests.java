package org.flag4j.arrays.dense.vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorScaleMultDivTests {

    double[] aEntries;
    Vector a;

    @Test
    void realScalMultTestCase() {
        double b;
        double[] expEntries;
        Vector exp;

        // ------------------------ sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = -34.5;
        expEntries = new double[]{aEntries[0]*b, aEntries[1]*b, aEntries[2]*b};
        exp = new Vector(expEntries);

        assertEquals(exp, a.mult(b));
    }


    @Test
    void complexScalMultTestCase() {
        Complex128 b;
        Complex128[] expEntries;
        CVector exp;

        // ------------------------ sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = new Complex128(-0.99234, 1.56);
        expEntries = new Complex128[]{b.mult(aEntries[0]), b.mult(aEntries[1]), b.mult(aEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));
    }


    // ---------------------------------------------------------------------------------------------------------------

    @Test
    void realScalDivTestCase() {
        double b;
        double[] expEntries;
        Vector exp;

        // ------------------------ sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = -34.5;
        expEntries = new double[]{aEntries[0]/b, aEntries[1]/b, aEntries[2]/b};
        exp = new Vector(expEntries);

        assertEquals(exp, a.div(b));
    }


    @Test
    void complexScalDivTestCase() {
        Complex128 b;
        Complex128[] expEntries;
        CVector exp;

        // ------------------------ sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = new Complex128(-0.99234, 1.56);
        expEntries = new Complex128[]{new Complex128(aEntries[0]).div(b), new Complex128(aEntries[1]).div(b),
                new Complex128(aEntries[2]).div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));
    }
}
