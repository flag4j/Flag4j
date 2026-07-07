package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorSetTests {

    double[] entries, expEntries;
    Vector a, exp;

    @Test
    void setTestCase() {
        // --------------------- sub-case 1 ---------------------
        entries = new double[]{1.34, -99.345, 1345.255, 1.5};
        a = new Vector(entries);
        expEntries = new double[]{-0.0009843, -99.345, -14.5, 1.5};
        exp = new Vector(expEntries);

        a.set(-0.0009843, 0);
        a.set(-14.5, 2);

        assertEquals(exp, a);
    }


    @Test
    void setSliceVecTestCase() {
        Vector slice;
        int startIdx;

        // --------------------- sub-case 1 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new Vector(1, 2, 0.0095, 31.9);
        startIdx = 2;
        exp = new Vector(-241.2, 9562.5, 1, 2, 0.0095, 31.9);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 2 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new Vector(1, 2, 0.0095);
        startIdx = 0;
        exp = new Vector(1, 2, 0.0095, -99.345, 1345.255, 1.5);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 3 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new Vector(1, 2, 0.0095);
        startIdx = 1;
        exp = new Vector(-241.2, 1, 2, 0.0095, 1345.255, 1.5);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 4 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new Vector(999d);
        startIdx = 5;
        exp = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 999);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 5 ---------------------
        a = new Vector(-241.2);
        slice = new Vector(999d);
        startIdx = 0;
        exp = new Vector(999d);

        assertEquals(exp, a.setSlice(slice, startIdx));

        // --------------------- sub-case 6 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        slice = new Vector(1, 2, 0.0095, 31.9, 5, 1, 5, 61.234, 112, 4);
        startIdx = 2;

        Vector finalSlice = slice;
        int finalStartIdx = startIdx;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setSlice(finalSlice, finalStartIdx));

        slice = new Vector(1, 2, 3);
        startIdx = -1;
        int finalStartIdx1 = startIdx;
        Vector finalSlice1 = slice;
        assertThrows(IllegalArgumentException.class, ()->a.setSlice(finalSlice1, finalStartIdx1));

        slice = new Vector(1, 2, 3);
        startIdx = 4;
        int finalStartIdx2 = startIdx;
        Vector finalSlice2 = slice;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setSlice(finalSlice2, finalStartIdx2));
    }


    @Test
    void setItemsVecTestCase() {
        Vector items;
        int[] itemIndices;
        int startIdx;

        // --------------------- sub-case 1 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Vector(0.0095, 31.9);
        itemIndices = new int[]{0, 3};
        exp = new Vector(0.0095, 9562.5, 1.34, 31.9, 1345.255, 1.5);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 2 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Vector(0.0095, 31.9, -0.345, 15.6);
        itemIndices = new int[]{0, 5, 2, 3};
        exp = new Vector(0.0095, 9562.5, -0.345, 15.6, 1345.255, 31.9);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 3 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Vector(0.0095, 31.9);
        itemIndices = new int[]{-1, 3};

        Vector finalItems = items;
        int[] finalItemIndices = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems, finalItemIndices));

        // --------------------- sub-case 4 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Vector(0.0095, 31.9);
        itemIndices = new int[]{1, 3, 4};

        Vector finalItems1 = items;
        int[] finalItemIndices1 = itemIndices;
        assertThrows(IllegalArgumentException.class, ()->a.setItems(finalItems1, finalItemIndices1));

        // --------------------- sub-case 4 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Vector(0.0095, 31.9);
        itemIndices = new int[]{1, 12};

        Vector finalItems2 = items;
        int[] finalItemIndices2 = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems2, finalItemIndices2));
    }


    @Test
    void setItemsDoubleTestCase() {
        Double[] items;
        int[] itemIndices;
        int startIdx;

        // --------------------- sub-case 1 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Double[]{0.0095, 31.9};
        itemIndices = new int[]{0, 3};
        exp = new Vector(0.0095, 9562.5, 1.34, 31.9, 1345.255, 1.5);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 2 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Double[]{0.0095, 31.9, -0.345, 15.6};
        itemIndices = new int[]{0, 5, 2, 3};
        exp = new Vector(0.0095, 9562.5, -0.345, 15.6, 1345.255, 31.9);

        assertEquals(exp, a.setItems(items, itemIndices));

        // --------------------- sub-case 3 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Double[]{0.0095, 31.9};
        itemIndices = new int[]{-1, 3};

        Double[] finalItems = items;
        int[] finalItemIndices = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems, finalItemIndices));

        // --------------------- sub-case 4 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Double[]{0.0095, 31.9};
        itemIndices = new int[]{1, 3, 4};

        Double[] finalItems1 = items;
        int[] finalItemIndices1 = itemIndices;
        assertThrows(IllegalArgumentException.class, ()->a.setItems(finalItems1, finalItemIndices1));

        // --------------------- sub-case 4 ---------------------
        a = new Vector(-241.2, 9562.5, 1.34, -99.345, 1345.255, 1.5);
        items = new Double[]{0.0095, 31.9};
        itemIndices = new int[]{1, 12};

        Double[] finalItems2 = items;
        int[] finalItemIndices2 = itemIndices;
        assertThrows(IndexOutOfBoundsException.class, ()->a.setItems(finalItems2, finalItemIndices2));
    }
}
