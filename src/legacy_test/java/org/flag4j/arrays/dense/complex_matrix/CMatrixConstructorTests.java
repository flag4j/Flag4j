package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixConstructorTests {
    Complex128[][] entries;
    String[][] entriesS;
    double[][] entriesD;
    int[][] entriesI;
    Complex128[] expEntries, entriesC1D;
    int expNumRows;
    int expNumCols;
    int size;
    Shape shape;
    Complex128 value;
    double valueD;

    CMatrix A, B;
    Matrix C;


    @Test
    void sizeConstructorTestCase() {
        // -------------- sub-case 1 --------------
        size = 5;
        expEntries = new Complex128[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        size = 2;
        expEntries = new Complex128[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- sub-case 3 --------------
        size = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(size));
    }


    @Test
    void valueSizeTestCase() {
        // -------------- sub-case 1 --------------
        size = 5;
        value = new Complex128(-23.13, 100.442);
        expEntries = new Complex128[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value;
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, value);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        size = 5;
        value = new Complex128(0, 5);
        expEntries = new Complex128[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value;
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, value);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 3 --------------
        value = new Complex128(0, 5);
        size = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(size, value));
    }


    @Test
    void valueSizeDoubleTestCase() {
        // -------------- sub-case 1 --------------
        size = 5;
        valueD = 9.1;
        expEntries = new Complex128[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(valueD);
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, valueD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        size = 5;
        valueD = -0.000013;
        expEntries = new Complex128[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(valueD);
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, valueD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- sub-case 3 --------------
        valueD = 5;
        size = -100;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(size, valueD));
    }


    @Test
    void rowsColsTestCase() {
        // -------------- sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        A = new CMatrix(expNumRows, expNumCols);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        A = new CMatrix(expNumRows, expNumCols);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 3 --------------
        expNumRows = -1;
        expNumCols = 2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols));

        // -------------- sub-case 4 --------------
        expNumRows = 1;
        expNumCols = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols));

        // -------------- sub-case 5 --------------
        expNumRows = -1;
        expNumCols = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols));
    }

    @Test
    void rowColValueTestCase() {
        // -------------- sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new Complex128[expNumRows*expNumCols];
        value = new Complex128(900.13);

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value;
        }

        A = new CMatrix(expNumRows, expNumCols, value);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        expEntries = new Complex128[expNumRows*expNumCols];
        value = new Complex128(-1, -13);

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value;
        }

        A = new CMatrix(expNumRows, expNumCols, value);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 3 --------------
        expNumRows = -1;
        expNumCols = 2;
        value = new Complex128(1, 1);

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, value));

        // -------------- sub-case 4 --------------
        expNumRows = 1;
        expNumCols = -2;
        value = new Complex128(1, 1);

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, value));

        // -------------- sub-case 5 --------------
        expNumRows = -1;
        expNumCols = -2;
        value = new Complex128(1, 1);

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, value));
    }

    @Test
    void rowColValueDoubleTestCase() {
        // -------------- sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new Complex128[expNumRows*expNumCols];
        valueD = 14;

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(valueD);
        }

        A = new CMatrix(expNumRows, expNumCols, valueD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        expEntries = new Complex128[expNumRows*expNumCols];
        valueD = Double.POSITIVE_INFINITY;

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(valueD);
        }

        A = new CMatrix(expNumRows, expNumCols, valueD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- sub-case 3 --------------
        expNumRows = -1;
        expNumCols = 2;
        valueD = 1;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, valueD));

        // -------------- sub-case 4 --------------
        expNumRows = 1;
        expNumCols = -2;
        valueD = 1;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, valueD));

        // -------------- sub-case 5 --------------
        expNumRows = -1;
        expNumCols = -2;
        valueD = 1;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, valueD));
    }


    @Test
    void shapeTestCase() {
        // -------------- sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        shape = new Shape(expNumRows, expNumCols);
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        A = new CMatrix(shape);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        shape = new Shape(expNumRows, expNumCols);
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }

        A = new CMatrix(shape);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void shapeValueTestCase() {
        // -------------- sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        shape = new Shape(expNumRows, expNumCols);
        value = new Complex128(19, -1345.0001);
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value;
        }

        A = new CMatrix(shape, value);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        shape = new Shape(expNumRows, expNumCols);
        value = new Complex128(1.3, 1.3566);
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value;
        }

        A = new CMatrix(shape, value);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void shapeValueDoubleTestCase() {
        // -------------- sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        shape = new Shape(expNumRows, expNumCols);
        valueD = 13.3456;
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(valueD);
        }

        A = new CMatrix(shape, valueD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        shape = new Shape(expNumRows, expNumCols);
        valueD = -99.3137;
        expEntries = new Complex128[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(valueD);
        }

        A = new CMatrix(shape, valueD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }

    @Test
    void arrTestCase() {
        entries = new Complex128[][]{{new Complex128("100.234-0.0103i"), new Complex128("134.5")},
                {new Complex128("i"), new Complex128("100.3465i")},
                {new Complex128("-0.9344-1.345i"), new Complex128("-103894.1334")},
                {new Complex128("0"), new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY)}};
        expNumRows = 4;
        expNumCols = 2;
        expEntries = new Complex128[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entries.length; i++) {
            for(int j=0; j<entries[0].length; j++) {
                expEntries[count++] = entries[i][j];
            }
        }
        A = new CMatrix(entries);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }

    @Test
    void arrStringTestCase() {
        entriesS = new String[][]{{"1", "-2+8.133i", "0.13334-i"}, {"133.4", "-29.13i", "8+9i"}};
        expNumRows = 2;
        expNumCols = 3;
        expEntries = new Complex128[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesS.length; i++) {
            for(int j=0; j<entriesS[0].length; j++) {
                expEntries[count++] = new Complex128(entriesS[i][j]);
            }
        }
        A = new CMatrix(entriesS);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }

    @Test
    void arrDoubleTestCase() {
        entriesD = new double[][]{{1, -2, 0.13334, 15.3, Double.NEGATIVE_INFINITY},
                {133.4, -29.13, 8, 0, Double.POSITIVE_INFINITY}};
        expNumRows = 2;
        expNumCols = 5;
        expEntries = new Complex128[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesD.length; i++) {
            for(int j=0; j<entriesD[0].length; j++) {
                expEntries[count++] = new Complex128(entriesD[i][j]);
            }
        }
        A = new CMatrix(entriesD);

        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void cmatTestCase() {
        entriesS = new String[][]{{"1", "-2+8.133i", "0.13334-i"}, {"133.4", "-29.13i", "8+9i"}};
        B = new CMatrix(entriesS);
        expNumRows = 2;
        expNumCols = 3;
        expEntries = new Complex128[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesS.length; i++) {
            for(int j=0; j<entriesS[0].length; j++) {
                expEntries[count++] = new Complex128(entriesS[i][j]);
            }
        }

        A = new CMatrix(B);
        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void rowsColsComplexTestCase() {
        entriesC1D = new Complex128[]{new Complex128(1, 345.6), new Complex128(-3441, 0.0094343431),
                new Complex128(-6, -9.2), new Complex128(Double.NEGATIVE_INFINITY, 1),
                new Complex128(9.234, -07643.2), new Complex128(6, 7),
                new Complex128(0.0000000002134), new Complex128(0, -92.2),
                new Complex128(5, 666666.4545), new Complex128(438905, 13)};
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new Complex128[]{new Complex128(1, 345.6), new Complex128(-3441, 0.0094343431),
                new Complex128(-6, -9.2), new Complex128(Double.NEGATIVE_INFINITY, 1),
                new Complex128(9.234, -07643.2), new Complex128(6, 7),
                new Complex128(0.0000000002134), new Complex128(0, -92.2),
                new Complex128(5, 666666.4545), new Complex128(438905, 13)};

        A = new CMatrix(expNumRows, expNumCols, entriesC1D);
        assertArrayEquals(expEntries, A.data);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }
}
