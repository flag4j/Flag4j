package org.flag4j.arrays.dense.tensor;


import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class TensorEqualsTests {

    static double[] aEntries;
    static Tensor A;
    static Shape aShape, bShape;

    int[][] sparseIndices;

    static void denseSetup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    static void sparseSetup() {
        aEntries = new double[]{
                1.23, 0, 0, 0, 0, -99.23425,
                0, 2.677, 0, -0.000194, 0, 0
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realDenseEqualsTestCase() {
        denseSetup();

        double[] bEntries;
        Tensor B;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);

        assertNotEquals(A, B);

        // ---------------------- sub-case 2 ----------------------
        bEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);

        assertEquals(A, B);

        // ---------------------- sub-case 3 ----------------------
        bEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        bShape = new Shape(4, 3);
        B = new Tensor(bShape, bEntries);

        assertNotEquals(A, B);

        // ---------------------- sub-case 4 ----------------------
        bEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        bShape = new Shape(2, 1, 3, 2);
        B = new Tensor(bShape, bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void sparseDenseEqualsTestCase() {
        sparseSetup();

        double[] bEntries;
        CooTensor B;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new double[]{1.23, -99.23425, 2.677, -0.000194};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getNdIndices(0),
                aShape.getNdIndices(5),
                aShape.getNdIndices(7),
                aShape.getNdIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertEquals(A.toCoo(), B);

        // ---------------------- sub-case 2 ----------------------
        bEntries = new double[]{1.23, -99.25, 2.677, -0.000194};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getNdIndices(0),
                aShape.getNdIndices(5),
                aShape.getNdIndices(7),
                aShape.getNdIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A.toCoo(), B);

        // ---------------------- sub-case 3 ----------------------
        bEntries = new double[]{1.23, -99.23425, 2.677, -0.000194};
        bShape = new Shape(21, 31, 2, 10005);
        sparseIndices = new int[][]{
                bShape.getNdIndices(0),
                bShape.getNdIndices(5),
                bShape.getNdIndices(7),
                bShape.getNdIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A.toCoo(), B);

        // ---------------------- sub-case 4 ----------------------
        bEntries = new double[]{1.23, -99.25, 2.677, -0.000194};
        bShape = new Shape(21, 3, 24);
        sparseIndices = new int[][]{
                aShape.getNdIndices(0),
                aShape.getNdIndices(5),
                aShape.getNdIndices(7),
                aShape.getNdIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A.toCoo(), B);
    }


    @Test
    void complexDenseEqualsTestCase() {
        denseSetup();

        Complex128[] bEntries;
        CTensor B;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(2.556), new Complex128(-121.5),
                new Complex128(15.61), new Complex128(14.15), new Complex128(-99.23425),
                new Complex128(0.001345), new Complex128(2.677), new Complex128(8.14),
                new Complex128(-0.000194), new Complex128(1), new Complex128(234)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);

        assertEquals(A.toComplex(), B);

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(2.556), new Complex128(-121.5),
                new Complex128(15.61, 1.4), new Complex128(14.15), new Complex128(-99.23425),
                new Complex128(0.001345), new Complex128(2.677), new Complex128(8.14),
                new Complex128(-0.000194), new Complex128(1), new Complex128(234)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);

        assertNotEquals(A.toComplex(), B);

        // ----------------------- sub-case 3 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(2.556), new Complex128(-121.5),
                new Complex128(15.61), new Complex128(14.15), new Complex128(-99.23425),
                new Complex128(0.001345), new Complex128(2.677), new Complex128(8.14),
                new Complex128(-0.000194), new Complex128(1), new Complex128(234)
        };
        bShape = new Shape(4, 3);
        B = new CTensor(bShape, bEntries);

        assertNotEquals(A.toComplex(), B);

        // ----------------------- sub-case 4 -----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(2.556), new Complex128(-121.5),
                new Complex128(15.61), new Complex128(14.15), new Complex128(-99.23425),
                new Complex128(9.245), new Complex128(2.677), new Complex128(8.14),
                new Complex128(-0.000194), new Complex128(1), new Complex128(234)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);

        assertNotEquals(A.toComplex(), B);
    }

    
    @Test
    void complexSparseEqualsTestCase() {
        sparseSetup();

        Complex128[] bEntries;
        CooCTensor B;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(-99.23425),
                new Complex128(2.677), new Complex128(-0.000194)};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getNdIndices(0),
                aShape.getNdIndices(5),
                aShape.getNdIndices(7),
                aShape.getNdIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertEquals(A.toCoo().toComplex(), B);

        // ---------------------- sub-case 2 ----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(-99.23425, 1.34235),
                new Complex128(2.677), new Complex128(-0.000194)};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getNdIndices(0),
                aShape.getNdIndices(5),
                aShape.getNdIndices(7),
                aShape.getNdIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A.toCoo().toComplex(), B);

        // ---------------------- sub-case 3 ----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(-99.23425),
                new Complex128(2.677), new Complex128(-0.000194)};
        bShape = new Shape(21, 31, 2, 10005);
        sparseIndices = new int[][]{
                bShape.getNdIndices(0),
                bShape.getNdIndices(5),
                bShape.getNdIndices(7),
                bShape.getNdIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A.toCoo().toComplex(), B);

        // ---------------------- sub-case 4 ----------------------
        bEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(-99.23425),
                new Complex128(2.677), new Complex128(-0.000194)};
        bShape = new Shape(21, 3, 24);
        sparseIndices = new int[][]{
                aShape.getNdIndices(0),
                aShape.getNdIndices(5),
                aShape.getNdIndices(7),
                aShape.getNdIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A.toCoo().toComplex(), B);
    }


    @Test
    void objectTestCase() {
        denseSetup();

        // ---------------------- sub-case 1 ----------------------
        assertNotEquals(A, new Shape(1, 14));

        // ---------------------- sub-case 2 ----------------------
        assertNotEquals(A, Double.valueOf(2.245));

        // ---------------------- sub-case 3 ----------------------
        assertNotEquals(A, "Hello Flag4j");
    }
}
