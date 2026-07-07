package org.flag4j.arrays.sparse.sparse_complex_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CooCVectorConstructorTests {
    int expSize;
    Shape expShape;
    Complex128[] expEntries;
    double[] expEntriesD;
    int[] expEntriesI;
    int[] expIndices;
    CooCVector a, b;


    @Test
    void sizeTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new Complex128[0];
        expIndices = new int[0];
        a = new CooCVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);
        assertArrayEquals(expIndices, a.indices);

        // ------------- sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize));
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new Complex128[]{new Complex128(-1233, 1.3314), new Complex128(9034, 10.23445),
            new Complex128(83.133, -334), new Complex128(-92.133, -9.4), new Complex128(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooCVector(expSize, expEntries, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);

        // ------------- sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntries, expIndices));

        // ------------- sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new Complex128[]{new Complex128(-1233, 1.3314), new Complex128(9034, 10.23445),
                new Complex128(83.133, -334), new Complex128(-92.133, -9.4), new Complex128(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntries, expIndices));


        // ------------- sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new Complex128[]{new Complex128(-1233, 1.3314), new Complex128(9034, 10.23445),
                new Complex128(83.133, -334), new Complex128(-92.133, -9.4), new Complex128(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntries, expIndices));
    }


    @Test
    void sizeEntriesDoubleIndicesTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new Complex128[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new Complex128(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooCVector(expSize, expEntriesD, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);

        // ------------- sub-case 2 -------------
        expSize = -1;
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesD, expIndices));

        // ------------- sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new Complex128[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new Complex128(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesD, expIndices));


        // ------------- sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new Complex128[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new Complex128(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesD, expIndices));
    }


    @Test
    void copyTestCase() {
        // ------------- sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new Complex128[]{new Complex128(-1233, 1.3314), new Complex128(9034, 10.23445),
                new Complex128(83.133, -334), new Complex128(-92.133, -9.4), new Complex128(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        b = new CooCVector(expSize, expEntries, expIndices);
        a = new CooCVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.data);
    }
}
