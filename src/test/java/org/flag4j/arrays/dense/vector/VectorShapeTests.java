package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorShapeTests {

    int[] indices;
    int size;

    double[] aEntries;
    Vector A;

    @Test
    void sizeTestCase() {
        // --------------------- sub-case 1 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        A = new Vector(aEntries);

        assertEquals(aEntries.length, A.length());

        // --------------------- sub-case 2 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234, 20243234.235, 1119.234, 5.14, -8.234};
        A = new Vector(aEntries);

        assertEquals(aEntries.length, A.length());
    }

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector B;

        // ------------------ sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{0, -924.34, 5, 1.34545};
        B = new Vector(bEntries);

        assertTrue(A.hasSameShape(B));
        assertTrue(A.hasSameShape(B));

        // ------------------ sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{0, -924.34, 5, 1.34545, 34.4};
        B = new Vector(bEntries);

        assertFalse(A.hasSameShape(B));
        assertFalse(A.hasSameShape(B));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector B;

        // ------------------ sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34, -0.34), new Complex128(0.445, 15.5), new Complex128(0.455), new Complex128(0, -8.435)};
        B = new CVector(bEntries);

        assertTrue(A.hasSameShape(B));
        assertTrue(A.hasSameShape(B));

        // ------------------ sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(0.455), new Complex128(0, -8.435)};
        B = new CVector(bEntries);

        assertFalse(A.hasSameShape(B));
        assertFalse(A.hasSameShape(B));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector B;

        // ------------------ sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{1.34545};
        indices = new int[]{2};
        size = 4;
        B = new CooVector(size, bEntries, indices);

        assertTrue(A.hasSameShape(B));
        assertTrue(A.hasSameShape(B));

        // ------------------ sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{1.45, 768.2, -1.55, 0.0234};
        indices = new int[]{1, 100, 502, 3405};
        size = 4096;
        B = new CooVector(size, bEntries, indices);

        assertFalse(A.hasSameShape(B));
        assertFalse(A.hasSameShape(B));
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCVector B;

        // ------------------ sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34.5, -9.632)};
        indices = new int[]{2};
        size = 4;
        B = new CooCVector(size, bEntries, indices);

        assertTrue(A.hasSameShape(B));
        assertTrue(A.hasSameShape(B));

        // ------------------ sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34, -0.34), new Complex128(0.445, 15.5), new Complex128(0.455), new Complex128(0, -8.435)};
        indices = new int[]{1, 100, 502, 3405};
        size = 4096;
        B = new CooCVector(size, bEntries, indices);

        assertFalse(A.hasSameShape(B));
        assertFalse(A.hasSameShape(B));
    }
}
