package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorCrossTests {

    Complex128[] aEntries;
    CVector a;

    Complex128[] expEntries;
    CVector exp;

    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347),
                new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345)};
        a = new CVector(aEntries);
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        // --------------------- sub-case 1 ---------------------
        bEntries = new Complex128[]{
                new Complex128(993.356, 1.6),
                new Complex128(-0.9935, 8.56),
                new Complex128(0, 8.35)};
        b = new CVector(bEntries);
        expEntries = new Complex128[]{
                new Complex128("-513.9324125-212.11007574999996i"),
                new Complex128("60428.54887-55858.235232i"),
                new Complex128("-43262.3265585-11026.0765445i")
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- sub-case 2 ---------------------
        bEntries = new Complex128[]{
                new Complex128(993.356, 1.6),
                new Complex128(-0.9935, 8.56),
                new Complex128(0, 8.35),
                new Complex128(99.24455, 0.0035)};
        b = new CVector(bEntries);
        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.cross(finalB));

        // --------------------- sub-case 3 ---------------------
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347),
                new Complex128(-9.234, 5.0)};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{
                new Complex128(993.356, 1.6),
                new Complex128(-0.9935, 8.56)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.cross(finalB2));
    }
}
