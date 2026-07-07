package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorStackJoinExtendTest {

    static Complex128[] aEntries;
    static CVector a;

    int[] sparseIndices;
    int sparseSize;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void joinComplexDenseTestCase() {
        Complex128[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new Complex128[]{new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01)};
        b = new CVector(bEntries);
        expEntries = new Complex128[]{new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257),
                new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01)
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void stackComplexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        Complex128[][] expEntries;
        CMatrix exp;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new Complex128[]{new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01),
        new Complex128(0, 1405.24), new Complex128(9.356), new Complex128(0.245, -8824.5)};
        b = new CVector(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
                        new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257)},
                {new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01),
                        new Complex128(0, 1405.24), new Complex128(9.356), new Complex128(0.245, -8824.5)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));

        // ---------------------- sub-case 2 ----------------------
        bEntries = new Complex128[]{new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01),
                new Complex128(0, 1405.24)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 0));

        // ---------------------- sub-case 3 ----------------------
        bEntries = new Complex128[]{new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01),
                new Complex128(0, 1405.24), new Complex128(9.356), new Complex128(0.245, -8824.5)};
        b = new CVector(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(1.455, 6126.347), new Complex128(2.4656, 9.24)},
                {new Complex128(-9.234, 5.0), new Complex128(-0.9924, -0.01)},
                {new Complex128(9.245, -56.2345), new Complex128(0, 1405.24)},
                {new Complex128(0, 14.5), new Complex128(9.356)},
                {new Complex128(-0.009257),new Complex128(0.245, -8824.5) }};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- sub-case 4 ----------------------
        bEntries = new Complex128[]{new Complex128(2.4656, 9.24), new Complex128(-0.9924, -0.01),
                new Complex128(0, 1405.24)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 1));
    }
}
