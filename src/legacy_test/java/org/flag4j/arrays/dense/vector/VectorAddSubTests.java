package org.flag4j.arrays.dense.vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorAddSubTests {

    int[] indices;
    int size;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseAddTestCase() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2]};
        exp = new Vector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void ComplexDenseAddTestCase() {
        Complex128[] bEntries, expEntries;
        CVector B, exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34.56, -0.9345), new Complex128(4.666, 1), new Complex128(0, 8.4)};
        B = new CVector(bEntries);
        expEntries = new Complex128[]{bEntries[0].add(aEntries[0]), bEntries[1].add(aEntries[1]), bEntries[2].add(aEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34.56, -0.9345), new Complex128(4.666, 1)};
        B = new CVector(bEntries);

        CVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void realSparseAddTestCase() {
        double[] bEntries, expEntries;
        CooVector B;
        Vector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new CooVector(size, bEntries, indices);
        expEntries = new double[]{1.34+34.677, 6.266, -90.45};
        exp = new Vector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVector(size, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void complexSparseAddTestCase() {
        Complex128[] bEntries, expEntries;
        CooCVector B;
        CVector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(345.66, -1.44559)};
        indices = new int[]{0};
        size = 3;
        B = new CooCVector(size, bEntries, indices);
        expEntries = new Complex128[]{bEntries[0].add(1.34), new Complex128(6.266), new Complex128(-90.45)};
        exp = new CVector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(345.66, -1.44559)};
        indices = new int[]{0};
        size = 201;
        B = new CooCVector(size, bEntries, indices);

        CooCVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void doubleAddTestCase() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34+B, 6.266+B, -90.45+B};
        exp = new Vector(expEntries);

        assertEquals(exp, A.add(B));
    }


    @Test
    void cNumberAddTestCase() {
        Complex128[] expEntries;
        Complex128 B = new Complex128(5.666, 0.975);
        CVector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new Complex128[]{B.add(1.34), B.add(6.266), B.add(-90.45)};
        exp = new CVector(expEntries);

        assertEquals(exp, A.add(B));
    }


    @Test
    void realDenseSubTestCase() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]-bEntries[0], aEntries[1]-bEntries[1], aEntries[2]-bEntries[2]};
        exp = new Vector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void ComplexDenseSubTestCase() {
        Complex128[] bEntries, expEntries;
        CVector B, exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34.56, -0.9345), new Complex128(4.666, 1), new Complex128(0, 8.4)};
        B = new CVector(bEntries);
        expEntries = new Complex128[]{new Complex128(aEntries[0]).sub(bEntries[0]), new Complex128(aEntries[1]).sub(bEntries[1]), new Complex128(aEntries[2]).sub(bEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(34.56, -0.9345), new Complex128(4.666, 1)};
        B = new CVector(bEntries);

        CVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void realSparseSubTestCase() {
        double[] bEntries, expEntries;
        CooVector B;
        Vector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new CooVector(size, bEntries, indices);
        expEntries = new double[]{1.34-34.677, 6.266, -90.45};
        exp = new Vector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVector(size, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void complexSparseSubTestCase() {
        Complex128[] bEntries, expEntries;
        CooCVector B;
        CVector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(345.66, -1.44559)};
        indices = new int[]{0};
        size = 3;
        B = new CooCVector(size, bEntries, indices);
        expEntries = new Complex128[]{new Complex128(1.34).sub(bEntries[0]), new Complex128(6.266), new Complex128(-90.45)};
        exp = new CVector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new Complex128[]{new Complex128(345.66, -1.44559)};
        indices = new int[]{0};
        size = 201;
        B = new CooCVector(size, bEntries, indices);

        CooCVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void doubleSubTestCase() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34-B, 6.266-B, -90.45-B};
        exp = new Vector(expEntries);

        assertEquals(exp, A.sub(B));
    }


    @Test
    void cNumberSubTestCase() {
        Complex128[] expEntries;
        Complex128 B = new Complex128(5.666, 0.975);
        CVector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new Complex128[]{new Complex128(aEntries[0]).sub(B), new Complex128(aEntries[1]).sub(B), new Complex128(aEntries[2]).sub(B)};

        exp = new CVector(expEntries);

        assertEquals(exp, A.sub(B));
    }
}
