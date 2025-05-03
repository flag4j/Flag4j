package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorSetTests {
    Complex128[] aEntries, expEntries;
    CVector a, exp;
    int index;

    @Test
    void doubleSetTestCase() {
        double val;

        // ------------------ sub-case 1 ------------------
        val = 45.14;
        index = 0;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(val), new Complex128(9.254), new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        val = 45.14;
        index = 1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(val), new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        val = 45.14;
        index = 2;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(val)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ sub-case 4 ------------------
        val = 45.14;
        index = 3;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        double finalVal = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal, index));

        // ------------------ sub-case 5 ------------------
        val = 45.14;
        index = -1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        double finalVal2 = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal2, index));
    }

    @Test
    void Complex128SetTestCase() {
        Complex128 val;

        // ------------------ sub-case 1 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 0;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{val, new Complex128(9.254), new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ sub-case 2 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), val, new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ sub-case 3 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 2;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), val};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ sub-case 4 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 3;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        Complex128 finalVal = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal, index));

        // ------------------ sub-case 5 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = -1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        Complex128 finalVal2 = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal2, index));
    }


    @Test
    void setSliceVecTestCase() {
        CVector slice;
        int startIdx;

        // --------------------- sub-case 1 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new CVector(1, 2, 0.0095, 31.9);
        startIdx = 2;
        exp = new CVector(-241.2, 9562.5, 1, 2, 0.0095, 31.9);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 2 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new CVector(1, 2, 0.0095);
        startIdx = 0;
        exp = new CVector(1, 2, 0.0095, -99.345, 1345.255, 1.5);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 3 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new CVector(1, 2, 0.0095);
        startIdx = 1;
        exp = new CVector(-241.2, 1, 2, 0.0095, 1345.255, 1.5);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 4 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new CVector(999d);
        startIdx = 5;
        exp = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 999);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 5 ---------------------
        a = new CVector(-241.2);
        slice = new CVector(999d);
        startIdx = 0;
        exp = new CVector(999d);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 6 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new CVector(1, 2, 0.0095, 31.9, 5, 1, 5, 61.234, 112, 4);
        startIdx = 2;

        CVector finalSlice = slice;
        int finalStartIdx = startIdx;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setSlice(finalSlice, finalStartIdx));

        slice = new CVector(1, 2, 3);
        startIdx = -1;
        int finalStartIdx1 = startIdx;
        CVector finalSlice1 = slice;
        assertThrows(IllegalArgumentException.class, ()->a.setSlice(finalSlice1, finalStartIdx1));

        slice = new CVector(1, 2, 3);
        startIdx = 4;
        int finalStartIdx2 = startIdx;
        CVector finalSlice2 = slice;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setSlice(finalSlice2, finalStartIdx2));
    }


    @Test
    void setItemsVecTestCase() {
        CVector items;
        int[] itemIndices;
        int startIdx;

        // --------------------- sub-case 1 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new CVector(0.0095, 31.9);
        itemIndices = new int[]{0, 3};
        exp = new CVector(0.0095, 9562.5, 1.34, 31.9, 1345.255, 1.5);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 2 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new CVector(0.0095, 31.9, -0.345, 15.6);
        itemIndices = new int[]{0, 5, 2, 3};
        exp = new CVector(0.0095, 9562.5, -0.345, 15.6, 1345.255, 31.9);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 3 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new CVector(0.0095, 31.9);
        itemIndices = new int[]{-1, 3};

        CVector finalItems = items;
        int[] finalItemIndices = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems, finalItemIndices));

        // --------------------- sub-case 4 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new CVector(0.0095, 31.9);
        itemIndices = new int[]{1, 3, 4};

        CVector finalItems1 = items;
        int[] finalItemIndices1 = itemIndices;
        assertThrows(IllegalArgumentException.class, ()->a.setItems(finalItems1, finalItemIndices1));

        // --------------------- sub-case 4 ---------------------
        a = new CVector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new CVector(0.0095, 31.9);
        itemIndices = new int[]{1, 12};

        CVector finalItems2 = items;
        int[] finalItemIndices2 = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems2, finalItemIndices2));
    }


    @Test
    void setItemsDoubleTestCase() {
        Complex128[] items;
        int[] itemIndices;
        int startIdx;

        // --------------------- sub-case 1 ---------------------
        a = new CVector("-241.2", "9562.5", "1.34", "-99.345", "1345.255", "1.5");
        items = new Complex128[]{new Complex128(0.0095, 145.5), new Complex128(0, 31.9)};
        itemIndices = new int[]{0, 3};
        exp = new CVector("0.0095+145.5i", "9562.5", "1.34", "31.9i", "1345.255", "1.5");

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 2 ---------------------
        a = new CVector("-241.2", "9562.5", "1.34", "-99.345", "1345.255", "1.5");
        items = new Complex128[]{new Complex128(0.0095), new Complex128(31.9),
                new Complex128(-0.345), new Complex128(15.6)};
        itemIndices = new int[]{0, 5, 2, 3};
        exp = new CVector(0.0095, 9562.5, -0.345, 15.6, 1345.255, 31.9);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 3 ---------------------
        a = new CVector("-241.2", "9562.5", "1.34", "-99.345", "1345.255", "1.5");
        items = new Complex128[]{new Complex128(0.0095), new Complex128(31.9)};
        itemIndices = new int[]{-1, 3};

        Complex128[] finalItems = items;
        int[] finalItemIndices = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems, finalItemIndices));

        // --------------------- sub-case 4 ---------------------
        a = new CVector("-241.2", "9562.5", "1.34", "-99.345", "1345.255", "1.5");
        items = new Complex128[]{new Complex128(0.0095), new Complex128(31.9)};
        itemIndices = new int[]{1, 3, 4};

        Complex128[] finalItems1 = items;
        int[] finalItemIndices1 = itemIndices;
        assertThrows(IllegalArgumentException.class, ()->a.setItems(finalItems1, finalItemIndices1));

        // --------------------- sub-case 4 ---------------------
        a = new CVector("-241.2", "9562.5", "1.34", "-99.345", "1345.255", "1.5");
        items = new Complex128[]{new Complex128(0.0095), new Complex128(31.9)};
        itemIndices = new int[]{1, 12};

        Complex128[] finalItems2 = items;
        int[] finalItemIndices2 = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems2, finalItemIndices2));
    }
}
