package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CTensorConstructorTests {

    double[] entriesD;
    int[] entriesI;
    double value;
    Complex128 valueC;
    Complex128[] expEntries;
    int expRank;
    Shape expShape;
    Tensor C;
    CTensor A, B;


    @Test
    void shapeConstructorTestCase() {
        // -------------- sub-case 1 --------------
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 2 --------------
        expShape = new Shape();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 3 --------------
        expShape = new Shape(1003);
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = Complex128.ZERO;
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }


    @Test
    void shapeValueConstructorTestCase() {
        // -------------- sub-case 1 --------------
        value = 3.1345;
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(value);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 2 --------------
        value = 11.4;
        expShape = new Shape();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(value);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 3 --------------
        value = 8;
        expShape = new Shape(1003);
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(value);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }


    @Test
    void shapeCValueConstructorTestCase() {
        // -------------- sub-case 1 --------------
        valueC = new Complex128(3.1345);
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = valueC;
        }

        expRank = expShape.getRank();
        A = new CTensor(expShape, valueC);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 2 --------------
        valueC = new Complex128(11.4, -0.3313);
        expShape = new Shape();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = valueC;
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, valueC);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 3 --------------
        valueC = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expShape = new Shape(1003);
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = valueC;
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, valueC);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }


    @Test
    void shapeEntriesDouble() {
        // ---------- sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new Complex128[expShape.numel().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new Complex128(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, };
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, entriesD));


        // ---------- sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 1, 4, 56, 113, 34, 5};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, entriesD));
    }


    @Test
    void shapeEntries() {
        // ---------- sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193), new Complex128(113334, -84334), new Complex128(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 2 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193), new Complex128(113334, -84334), new Complex128(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 3 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193), new Complex128(113334, -84334), new Complex128(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 4 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193), new Complex128(113334, -84334), new Complex128(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 5 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193)};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, expEntries));

        // ---------- sub-case 6 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193), new Complex128(113334, -84334), new Complex128(190, 334.55),
                new Complex128(0, 0)};
        expRank = expShape.getRank();
        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, expEntries));
    }


    @Test
    void cTensorConstructorTestCase() {
        expShape = new Shape(2, 3, 1, 2);
        expEntries = new Complex128[] {
                new Complex128(1, 0.32), new Complex128(11.2334, -94.45), new Complex128(94), new Complex128(0, 445.2),
                new Complex128(1, 3), new Complex128(-9, -13.4), new Complex128(34.5, 1), new Complex128(0,-1),
                new Complex128(0.0000043), new Complex128(0, 0.99810382193), new Complex128(113334, -84334), new Complex128(190, 334.55)};
        expRank = expShape.getRank();
        B = new CTensor(expShape, expEntries);
        A = new CTensor(B);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }
}
