package org.flag4j.arrays.dense.vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class VectorEqualsTests {

    int[] indices;
    int sparseSize;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector B;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 543.354, -0.3456};
        B = new Vector(bEntries);

        assertEquals(A, B);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 123.5, -0.3456};
        B = new Vector(bEntries);

        assertNotEquals(A, B);


        // -------------------- sub-case 3 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 543.354, -0.3456, 0};
        B = new Vector(bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector B;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(1.234), new Complex128(543.354), new Complex128(-0.3456)};
        B = new CVector(bEntries);

        assertEquals(A.toComplex(), B);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(1.234, 13.4), new Complex128(543.354), new Complex128(-0.3456)};
        B = new CVector(bEntries);

        assertNotEquals(A.toComplex(), B);


        // -------------------- sub-case 3 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(1.234), new Complex128(543.354), new Complex128(-0.3456), new Complex128(0, 123.5)};
        B = new CVector(bEntries);

        assertNotEquals(A.toComplex(), B);
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector B;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{1};
        sparseSize = 3;
        B = new CooVector(sparseSize, bEntries, indices);

        assertEquals(A.toCoo(), B);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{2};
        sparseSize = 3;
        B = new CooVector(sparseSize, bEntries, indices);

        assertNotEquals(A.toCoo(), B);

        // -------------------- sub-case 3 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{1};
        sparseSize = 4;
        B = new CooVector(sparseSize, bEntries, indices);

        assertNotEquals(A.toCoo(), B);
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCVector B;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(543.354)};
        indices = new int[]{1};
        sparseSize = 3;
        B = new CooCVector(sparseSize, bEntries, indices);

        assertEquals(A.toCoo().toComplex(), B);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(543.354, -9.34)};
        indices = new int[]{1};
        sparseSize = 3;
        B = new CooCVector(sparseSize, bEntries, indices);

        assertNotEquals(A.toCoo().toComplex(), B);

        // -------------------- sub-case 3 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(543.354)};
        indices = new int[]{1};
        sparseSize = 4;
        B = new CooCVector(sparseSize, bEntries, indices);

        assertNotEquals(A.toCoo().toComplex(), B);
    }


    @Test
    void objectTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{123.4};
        A = new Vector(aEntries);
        Double B = 123.4;

        assertNotEquals(A, B);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        String BString = "hello";

        assertNotEquals(A, BString);
    }
}
