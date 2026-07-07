package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CooVectorConstructorTests {
    int expSize;
    Shape expShape;
    double[] expEntries;
    int[] expEntriesI;
    int[] expIndices;
    CooVector a, b;


    @Test
    void sizeTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new double[0];
        expIndices = new int[0];
        a = new CooVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);
        assertArrayEquals(expIndices, a.indices);

        // ------------- sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize));
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooVector(expSize, expEntries, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);

        // ------------- sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntries, expIndices));

        // ------------- sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntries, expIndices));


        // ------------- sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntries, expIndices));
    }


    @Test
    void sizeEntriesIntIndicesTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooVector(expSize, expEntriesI, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);

        // ------------- sub-case 2 -------------
        expSize = -1;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntriesI, expIndices));

        // ------------- sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntriesI, expIndices));


        // ------------- sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntriesI, expIndices));
    }


    @Test
    void copyTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        b = new CooVector(expSize, expEntries, expIndices);
        a = new CooVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);
    }
}
