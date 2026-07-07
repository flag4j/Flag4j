package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorConstructorTests {
    int expSize;
    Complex128 fillValue;
    double fillValueD;
    Shape expShape;
    Complex128[] expEntries;
    double[] entriesD;
    int[] entriesI;
    CVector a, b;

    @Test
    void sizeTestCase() {
        // ----------- sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        a = new CVector(expSize);

        assertEquals(expSize, a.size);
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        a = new CVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 3 ------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new Vector(expSize));
    }

    @Test
    void sizeFillTestCase() {
        // ----------- sub-case 1 ------------
        expSize = 5;
        fillValue = new Complex128(-10.23423, 100.2);
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = fillValue;
        }

        a = new CVector(expSize, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        expSize = 0;
        fillValue = new Complex128(-10.23423, 100.2);
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = fillValue;
        }

        a = new CVector(expSize,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 3 ------------
        expSize = -1;
        fillValue = new Complex128(-10.23423, 100.2);
        assertThrows(IllegalArgumentException.class, () -> new CVector(expSize, fillValue));
    }


    @Test
    void sizeFillDTestCase() {
        // ----------- sub-case 1 ------------
        expSize = 5;
        fillValueD = 10.234;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(fillValueD);
        }

        a = new CVector(expSize, fillValueD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        expSize = 0;
        fillValueD = -10.234;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(fillValueD);
        }

        a = new CVector(expSize,fillValueD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 3 ------------
        expSize = -1;
        fillValueD = -10.234;
        assertThrows(IllegalArgumentException.class, () -> new CVector(expSize, fillValueD));
    }


    @Test
    void entriesTestCase() {
        // ----------- sub-case 1 ------------
        entriesD = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = entriesD.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesD[i]);
        }

        a = new CVector(entriesD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        entriesD = new double[]{-0.234974};
        expSize = entriesD.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesD[i]);
        }

        a = new CVector(entriesD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }
    }


    @Test
    void entriesITestCase() {
        // ----------- sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesI[i]);
        }

        a = new CVector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesI[i]);
        }

        a = new CVector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }
    }


    @Test
    void entriesCTestCase() {
        // ----------- sub-case 1 ------------
        expEntries = new Complex128[]{new Complex128(100, 234.13), new Complex128(-0.992, 113.3),
                new Complex128(-0.0000000000001), new Complex128(0, -342.13)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new CVector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        expEntries = new Complex128[]{new Complex128(-22, -0.92)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new CVector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }
    }


    @Test
    void copyTestCase() {
        // ----------- sub-case 1 ------------
        expEntries = new Complex128[]{new Complex128(100, 234.13), new Complex128(-0.992, 113.3),
                new Complex128(-0.0000000000001), new Complex128(0, -342.13)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        b = new CVector(expEntries);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesI[i]);
        }

        b = new CVector(entriesI);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 3 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesI[i]);
        }

        b = new CVector(entriesI);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
                for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }

        // ----------- sub-case 4 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new Complex128[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesI[i]);
        }

        b = new CVector(entriesI);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.data[i]);
        }
    }

}
