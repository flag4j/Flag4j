package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class CMatrixEqualsTests {
    double[][] bRealEntries;
    double[] bRealSparseEntries;

    Complex128[][] aEntries, bEntries;
    Complex128[] bComplexSparseEntries;

    int[] rowIndices, colIndices;
    Shape sparseShape;

    CooMatrix BRealSparse;
    CooCMatrix BComplexSparse;
    Matrix BReal;
    CMatrix A, B;

    @Test
    void complexTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        B = new CMatrix(bEntries);

        assertEquals(A, B);

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4), new Complex128(67.1, 0.0003443993)},
                {new Complex128(-723.234, 4), new Complex128(-9.431), new Complex128(8.4554, -98.2)}};
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4, 0.0000001)},
                {new Complex128(67.1), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void realTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56), new Complex128(4)},
                {new Complex128(67.1), new Complex128(8.4554)},
                {new Complex128(-723.234), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4},
                {67.1, 8.4554},
                {-723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertEquals(A, BReal.toComplex());

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56), new Complex128(4)},
                {new Complex128(67.1), new Complex128(8.4554)},
                {new Complex128(-723.234), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4, 67.1},
                {8.4554, -723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertNotEquals(A, BReal.toComplex());

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, 1.35), new Complex128(4)},
                {new Complex128(67.1), new Complex128(8.4554)},
                {new Complex128(-723.234), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4},
                {67.1, 8.4554},
                {-723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertNotEquals(A, BReal.toComplex());

        // ---------------------- sub-case 4 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56), new Complex128(4)},
                {new Complex128(67.1), new Complex128(8.454)},
                {new Complex128(-723.234), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4},
                {67.1, 8.4554},
                {-723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertNotEquals(A, BReal.toComplex());
    }


    @Test
    void complexSparseTestCase() {
        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO},
                {new Complex128(67.1, 1.556), new Complex128(1)},
                {Complex128.ZERO, new Complex128(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new Complex128[]{new Complex128(67.1, 1.556), new Complex128(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertNotEquals(A, BComplexSparse);

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO},
                {new Complex128(67.1, 1.556), Complex128.ZERO},
                {Complex128.ZERO, new Complex128(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new Complex128[]{new Complex128(67.1, 1.556), new Complex128(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+3);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertNotEquals(A, BComplexSparse);

        // ---------------------- sub-case 4 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO},
                {new Complex128(67.1), Complex128.ZERO},
                {Complex128.ZERO, new Complex128(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new Complex128[]{new Complex128(67.1, 1.556), new Complex128(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertNotEquals(A, BComplexSparse);
    }


    @Test
    void otherObjectTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO},
                {new Complex128(67.1, 1.556), Complex128.ZERO},
                {Complex128.ZERO, new Complex128(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        assertNotEquals(A, Double.valueOf(32.45));
        assertNotEquals(A, new Shape(4, 56));
        assertNotEquals("Hello World!", A);
    }
}
