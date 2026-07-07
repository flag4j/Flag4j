package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorAddSubEqTests {

    int[] indices;
    int size;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseAddEqTestCase() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2]};
        exp = new Vector(expEntries);

        A.addEq(B);

        assertEquals(exp, A);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.addEq(finalB));
    }


    @Test
    void realSparseAddEqTestCase() {
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

        RealDenseSparseVectorOps.addEq(A, B);
        assertEquals(exp, A);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVector(size, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> RealDenseSparseVectorOps.addEq(A, finalB));
    }


    @Test
    void doubleAddEqTestCase() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34+B, 6.266+B, -90.45+B};
        exp = new Vector(expEntries);

        A.addEq(B);
        assertEquals(exp, A);
    }

    // ----------------------------------------------------------------------------------------------


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

        A.subEq(B);
        assertEquals(exp, A);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.subEq(finalB));
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

        RealDenseSparseVectorOps.subEq(A, B);
        assertEquals(exp, A);

        // -------------------- sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVector(size, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> RealDenseSparseVectorOps.subEq(A, finalB));
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

        A.subEq(B);
        assertEquals(exp, A);
    }
}
