package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.DirectSum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixDirectSumTests {

    double[][] aEntries, bEntries, expEntries;
    Complex128[][] bComplexEntries, expComplexEntries;
    double[] bSparseEntries;
    Complex128[] bSparseComplexEntries;

    int[] rowIndices, colIndices;
    Shape sparseShape;

    Matrix A, B, exp;
    CMatrix BComplex, expComplex;
    CooMatrix BSparse;
    CooCMatrix BSparseComplex;

    @Test
    void matrixTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{-9, 1.234}, {89.1, 0}, {0, -4.6}};
        B = new Matrix(bEntries);

        expEntries = new double[][]{
                {1, 2, 3, 0, 0},
                {4, 5, 6, 0, 0},
                {0, 0, 0, -9, 1.234},
                {0, 0, 0, 89.1, 0},
                {0, 0, 0, 0, -4.6}};
        exp = new Matrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void sparseMatrixTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseEntries = new double[]{9.32, 13.4, -80.1, 1001.0004};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparse = new CooMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        expEntries = new double[][]{
                {1, 2, 3, 0, 0, 0},
                {4, 5, 6, 0, 0, 0},
                {0, 0, 0, 9.32, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 13.4, -80.1, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1001.0004, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, BSparse));
    }


    @Test
    void complexMatrixTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bComplexEntries = new Complex128[][]{{new Complex128(0, 1), new Complex128(8.13)},
                {new Complex128(1.44, -9.436), new Complex128(6.71, 8.44)}};
        BComplex = new CMatrix(bComplexEntries);

        expComplexEntries = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3), new Complex128(0), new Complex128(0)},
                {new Complex128(4), new Complex128(5), new Complex128(6), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0, 1), new Complex128(8.13)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1.44, -9.436), new Complex128(6.71, 8.44)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, DirectSum.directSum(A, BComplex));
    }

    @Test
    void sparseComplexMatrixTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseComplexEntries = new Complex128[]{new Complex128(0, 1), new Complex128(8.13),
                new Complex128(1.44, -9.436), new Complex128(6.71, 8.44)};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparseComplex = new CooCMatrix(sparseShape, bSparseComplexEntries, rowIndices, colIndices);

        expComplexEntries = new Complex128[][]{
                {new Complex128(1), new Complex128(2), new Complex128(3), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(4), new Complex128(5), new Complex128(6), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0, 1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(8.13), new Complex128(1.44, -9.436), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(6.71, 8.44), new Complex128(0)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, DirectSum.directSum(A, BSparseComplex));
    }

    //--------------------------------------------------------------------------------------------------------------

    @Test
    void matrixInvTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{-9, 1.234}, {89.1, 0}, {0, -4.6}};
        B = new Matrix(bEntries);
        expEntries = new double[][]{
                {0, 0, 0, -9, 1.234},
                {0, 0, 0, 89.1, 0},
                {0, 0, 0, 0, -4.6},
                {1, 2, 3, 0, 0},
                {4, 5, 6, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void sparseMatrixInvTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseEntries = new double[]{9.32, 13.4, -80.1, 1001.0004};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparse = new CooMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        expEntries = new double[][]{
                {0, 0, 0, 9.32, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 13.4, -80.1, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1001.0004, 0},
                {1, 2, 3, 0, 0, 0},
                {4, 5, 6, 0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, BSparse));
    }


    @Test
    void complexMatrixInvTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bComplexEntries = new Complex128[][]{{new Complex128(0, 1), new Complex128(8.13)},
                {new Complex128(1.44, -9.436), new Complex128(6.71, 8.44)}};
        BComplex = new CMatrix(bComplexEntries);

        expComplexEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0, 1), new Complex128(8.13)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1.44, -9.436), new Complex128(6.71, 8.44)},
                {new Complex128(1), new Complex128(2), new Complex128(3), new Complex128(0), new Complex128(0)},
                {new Complex128(4), new Complex128(5), new Complex128(6), new Complex128(0), new Complex128(0)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, DirectSum.invDirectSum(A, BComplex));
    }

    @Test
    void sparseComplexMatrixInvTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseComplexEntries = new Complex128[]{new Complex128(0, 1), new Complex128(8.13),
                new Complex128(1.44, -9.436), new Complex128(6.71, 8.44)};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparseComplex = new CooCMatrix(sparseShape, bSparseComplexEntries, rowIndices, colIndices);

        expComplexEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0, 1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(8.13), new Complex128(1.44, -9.436), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(6.71, 8.44), new Complex128(0)},
                {new Complex128(1), new Complex128(2), new Complex128(3), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(4), new Complex128(5), new Complex128(6), new Complex128(0), new Complex128(0), new Complex128(0)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, DirectSum.invDirectSum(A, BSparseComplex));
    }
}
