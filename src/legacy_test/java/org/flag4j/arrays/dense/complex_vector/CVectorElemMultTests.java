package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorElemMultTests {

    static Complex128[] aEntries;
    static CVector a;
    Complex128[] expEntries;
    CVector exp;

    int[] sparseIndices;
    int sparseSize;


    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(4.556, -85.2518), new Complex128(43.1, -99.34551),
                new Complex128(6915.66), new Complex128(0, 9.345)};
        a = new CVector(aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ------------------- sub-case 1 -------------------
        bEntries = new double[]{2.455, -9.24, 0, 24.50001};
        b = new Vector(bEntries);
        expEntries = new Complex128[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                aEntries[2].mult(bEntries[2]), aEntries[3].mult(bEntries[3])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- sub-case 2 -------------------
        bEntries = new double[]{2.455, -9.24};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;

        // ------------------- sub-case 1 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{0, 1};
        sparseSize = 4;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                Complex128.ZERO, Complex128.ZERO};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b).toDense());

        // ------------------- sub-case 2 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{1, 3};
        sparseSize = 4;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[]{Complex128.ZERO, aEntries[1].mult(bEntries[0]),
                Complex128.ZERO, aEntries[3].mult(bEntries[1])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b).toDense());

        // ------------------- sub-case 3 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{0, 1};
        sparseSize = 185234;
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        // ------------------- sub-case 1 -------------------
        bEntries = new Complex128[]{new Complex128(-0.00024), new Complex128(0, 85.234),
            new Complex128(0.00234, 15.6), new Complex128(-0.24, 662.115)};
        b = new CVector(bEntries);
        expEntries = new Complex128[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                aEntries[2].mult(bEntries[2]), aEntries[3].mult(bEntries[3])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- sub-case 2 -------------------
        bEntries = new Complex128[]{new Complex128(0, 85.234),
                new Complex128(0.00234, 15.6), new Complex128(-0.24, 662.115)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }

    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCVector b;

        // ------------------- sub-case 1 -------------------
        bEntries = new Complex128[]{new Complex128(234.566, -9.225), new Complex128(0.00024, 15.5)};
        sparseIndices = new int[]{0, 1};
        sparseSize = 4;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                Complex128.ZERO, Complex128.ZERO};
        exp = new CVector(expEntries);

        assertEquals(exp.toCoo(), a.elemMult(b));

        // ------------------- sub-case 2 -------------------
        bEntries = new Complex128[]{new Complex128(-23.566, 0), new Complex128(0, 15.5)};
        sparseIndices = new int[]{1, 3};
        sparseSize = 4;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new Complex128[]{Complex128.ZERO, aEntries[1].mult(bEntries[0]),
                Complex128.ZERO, aEntries[3].mult(bEntries[1])};
        exp = new CVector(expEntries);

        assertEquals(exp.toCoo(), a.elemMult(b));

        // ------------------- sub-case 3 -------------------
        bEntries = new Complex128[]{new Complex128(-23.566, 0), new Complex128(0, 15.5)};
        sparseIndices = new int[]{0, 1};
        sparseSize = 185234;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }
}
