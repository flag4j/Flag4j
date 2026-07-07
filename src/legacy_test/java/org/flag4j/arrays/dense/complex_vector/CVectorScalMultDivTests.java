package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorScalMultDivTests {

    static Complex128[] aEntries;
    static CVector a;
    Complex128[] expEntries;
    CVector exp;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(2.566, -9.24), new Complex128(-24.565, 9.3),
                new Complex128(3.54698), new Complex128(0, 8.356)};
        a = new CVector(aEntries);
    }


    @Test
    void realScalMultTestCase() {
        double b;

        // ------------------- sub-case 1 -------------------
        b = 129.12354;
        expEntries = new Complex128[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));

        // ------------------- sub-case 2 -------------------
        b = -9.12354;
        expEntries = new Complex128[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));
    }


    @Test
    void complexScalMultTestCase() {
        Complex128 b;

        // ------------------- sub-case 1 -------------------
        b = new Complex128(-99.234, 56.1);
        expEntries = new Complex128[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));

        // ------------------- sub-case 2 -------------------
        b = new Complex128(9.234000014, -56.1);
        expEntries = new Complex128[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));
    }


    @Test
    void realScalDivTestCase() {
        double b;

        // ------------------- sub-case 1 -------------------
        b = 129.12354;
        expEntries = new Complex128[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));

        // ------------------- sub-case 2 -------------------
        b = -9.12354;
        expEntries = new Complex128[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));
    }


    @Test
    void complexScalDivTestCase() {
        Complex128 b;

        // ------------------- sub-case 1 -------------------
        b = new Complex128(-99.234, 56.1);
        expEntries = new Complex128[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));

        // ------------------- sub-case 2 -------------------
        b = new Complex128(9.234000014, -56.1);
        expEntries = new Complex128[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));
    }
}
