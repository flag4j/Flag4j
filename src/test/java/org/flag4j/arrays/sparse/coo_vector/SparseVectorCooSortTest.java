package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseVectorCooSortTest {
    static int[] aIndices, expIndices;
    static double[] aEntries, expEntries;
    static int sparseSize;
    static CooVector a, exp;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.345, -989.234, 5.15, 617.4, 1.34, 5126.234, 456.3};
        aIndices = new int[]{36, 13, 4, 11345, 3, 645, 3324};
        sparseSize = 24_023;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseSortTestCase() {
        // --------------------- sub-case 1 ---------------------
        expIndices = new int[]{3, 4, 13, 36, 645, 3324, 11345};
        expEntries = new double[]{1.34, 5.15, -989.234, 1.345, 5126.234, 456.3, 617.4};
        exp = new CooVector(sparseSize, expEntries, expIndices);

        a.sortIndices();
        assertEquals(exp, a);
    }
}
