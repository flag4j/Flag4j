package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class TensorConstructorTests {
    double[] entriesD;
    Double[] entriesDObject;
    int[] entriesI;
    Integer[] entriesInteger;
    double value;
    double[] expEntries;
    int expRank;
    Shape expShape;
    Tensor A, B;


    private double[] toDouble(Integer[] arr) {
        double[] result = new double[arr.length];

        for(int i=0; i<arr.length; i++) {
            result[i] = arr[i];
        }

        return result;
    }

    @Test
    void shapeConstructorTestCase() {
        // -------------- sub-case 1 --------------
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new double[expShape.numel().intValue()];
        expRank = expShape.getRank();
        A = new Tensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 2 --------------
        expShape = new Shape();
        expEntries = new double[expShape.numel().intValue()];
        expRank = expShape.getRank();
        A = new Tensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 3 --------------
        expShape = new Shape(1003);
        expEntries = new double[expShape.numel().intValue()];
        expRank = expShape.getRank();
        A = new Tensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }


    @Test
    void shapeValueConstructorTestCase() {
        // -------------- sub-case 1 --------------
        value = 3.1345;
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new double[expShape.numel().intValue()];
        expRank = expShape.getRank();
        Arrays.fill(expEntries, value);
        A = new Tensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 2 --------------
        value = 11.4;
        expShape = new Shape();
        expEntries = new double[expShape.numel().intValue()];
        expRank = expShape.getRank();
        Arrays.fill(expEntries, value);
        A = new Tensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // -------------- sub-case 3 --------------
        value = 8;
        expShape = new Shape(1003);
        expEntries = new double[expShape.numel().intValue()];
        expRank = expShape.getRank();
        Arrays.fill(expEntries, value);
        A = new Tensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }


    @Test
    void shapeEntriesDoubleTestCase() {
        // ---------- sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, };
        expRank = expShape.getRank();
        expEntries = entriesD.clone();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesD));


        // ---------- sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 1, 4, 56, 113, 34, 5};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesD));
    }


    @Test
    void shapeEntriesIntTestCase() {
        // ---------- sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesI));

        // ---------- sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1, 19, 2313, 112, 3};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesI));
    }


    @Test
    void tensorConstructorTestCase() {
        expShape = new Shape(2, 3, 1, 2);
        expEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        expRank = expShape.getRank();
        B = new Tensor(expShape, expEntries);
        A = new Tensor(B);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);
    }


    @Test
    void shapeEntriesDoubleObjectTestCase() {
        // ---------- sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesDObject = new Double[]{1d, 2d, 3d, 4d, 5d, 6d, 7d, -221.233, 11.33, 11d,
                2d, -11334d, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4d, 8d, 100d, 2343d,
                9.33, 3244d, 156.224, 3445d, 60.3, 44d, 13d, 1d, 4d, 56d};
        expRank = expShape.getRank();
        expEntries = Stream.of(entriesDObject).mapToDouble(Double::doubleValue).toArray();
        A = new Tensor(expShape, entriesDObject);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesDObject = new Double[]{1d, 2d, 3d, 4d, 5d, 6d, 7d, -221.233, 11.33, 11d,
                2d, -11334d, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4d, 8d, 100d, 2343d,
                9.33, 3244d, 156.224, 3445d, 60.3, 44d, 13d, 1d, 4d, 56d};
        expRank = expShape.getRank();
        expEntries = Stream.of(entriesDObject).mapToDouble(Double::doubleValue).toArray();
        A = new Tensor(expShape, entriesDObject);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesDObject = new Double[]{1d, 2d, 3d, 4d, 5d, 6d, 7d, -221.233, 11.33, 11d,
                2d, -11334d, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4d, 8d, 100d, 2343d,
                9.33, 3244d, 156.224, 3445d, 60.3, 44d, 13d, 1d, 4d, 56d};
        expRank = expShape.getRank();
        expEntries = Stream.of(entriesDObject).mapToDouble(Double::doubleValue).toArray();
        A = new Tensor(expShape, entriesDObject);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesDObject = new Double[]{1d, 2d, 3d, 4d, 5d, 6d, 7d, -221.233, 11.33, 11d,
                2d, -11334d, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4d, 8d, 100d, 2343d,
                9.33, 3244d, 156.224, 3445d, 60.3, 44d, 13d, 1d, 4d, 56d};
        expRank = expShape.getRank();
        expEntries = Stream.of(entriesDObject).mapToDouble(Double::doubleValue).toArray();
        A = new Tensor(expShape, entriesDObject);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesDObject = new Double[]{1d, 2d, 3d, 4d, 5d, 6d, 7d, -221.233, 11.33, 11d,
                2d, -11334d, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4d, 8d, 100d, 2343d,
                9.33, 3244d, 156.224, 3445d, 60.3, 44d, 13d, 1d, 4d, 56d, 113d, 34d, 5d};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesDObject));


        // ---------- sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesDObject = new Double[]{1d, 2d, 3d, 4d, 5d, 6d, 7d, -221.233, 11.33, 11d,
                2d, -11334d, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4d, 8d, 100d, 2343d,
                9.33, 3244d, 156.224, 3445d, 60.3, 44d, 13d, 1d, 4d, 56d, 113d, 34d, 5d};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesDObject));
    }


    @Test
    void shapeEntriesIntegerObjectTestCase() {
        // ---------- sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesInteger = new Integer[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = toDouble(entriesInteger);
        A = new Tensor(expShape, entriesInteger);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesInteger = new Integer[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = toDouble(entriesInteger);
        A = new Tensor(expShape, entriesInteger);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesInteger = new Integer[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = toDouble(entriesInteger);
        A = new Tensor(expShape, entriesInteger);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesInteger = new Integer[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = toDouble(entriesInteger);
        A = new Tensor(expShape, entriesInteger);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.data);

        // ---------- sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesInteger = new Integer[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesInteger));

        // ---------- sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesInteger = new Integer[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1, 19, 2313, 112, 3};
        expRank = expShape.getRank();
        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesInteger));
    }


    @Test
    void matrixTestCase() {
        Matrix mat;

        // ------------------------- sub-case 1 -------------------------
        expShape = new Shape(6, 2);
        entriesD = new double[]{1.3243, 4, -0.24, 0, 0.234235, -0.0, 234.567, 1.35, 1.45, 1.5, 88834.345, 9141.3335};
        mat = new Matrix(expShape, entriesD);
        A = new Tensor(mat.shape, mat.data.clone());

        assertEquals(expShape, A.shape);
        assertArrayEquals(mat.data, A.data);
    }


    @Test
    void vectorTestCase() {
        Vector vec;

        // ------------------------- sub-case 1 -------------------------
        entriesD = new double[]{1.3243, 4, -0.24, 0, 0.234235};
        vec = new Vector(entriesD);
        A = new Tensor(vec.shape, vec.data.clone());

        assertEquals(vec.shape, A.shape);
        assertArrayEquals(vec.data, A.data);
    }


    @Test
    void copyTestCase() {
        // -------------- sub-case 1 --------------
        expShape = new Shape(2, 3, 1, 2);
        expEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        A = new Tensor(expShape, expEntries);

        B = A.copy();

        assertEquals(A, B);
    }

}
