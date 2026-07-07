package org.flag4j.linalg.ops.dense_sparse.complex;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

@SuppressWarnings("SpellCheckingInspection")
class ComplexDenseCooMatrixMultiplicationTests {

    static Shape sparseShape;
    static int[][] sparseIndices;
    static int sparseSize;
    static int[] sparseVecIndices;

    static Complex128[][] aEntries, expEntries;
    static Complex128[] bEntries, bVecEntries, bVecSparseEntries;

    static CMatrix A;
    static CooCMatrix B;
    static CVector bvec;
    static CooCVector bSparse;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};

        bEntries = new Complex128[]{new Complex128(1.334, -5.00024), new Complex128(-73.56, 234.56)};
    }

    static void createMatrices() {
        A = new CMatrix(aEntries);
        B = new CooCMatrix(sparseShape, bEntries, sparseIndices[0], sparseIndices[1]);
    }

    static void createDenseVector() {
        bvec = new CVector(bVecEntries);
    }

    static void createSparseVector() {
        bSparse = new CooCVector(sparseSize, bVecSparseEntries, sparseVecIndices);
    }


    @Test
    void matMatMultTestCase()  {
        // ----------------------- sub-case 1 -----------------------
        sparseShape = new Shape(2, 5);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 4}};
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("172.842232+40.75596i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("2172.4976+794.6704000000001i")},
                {new Complex128("0.0"), new Complex128("368.567656-357.60642975999997i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-220.68+703.6800000000001i")},
                {new Complex128("0.0"), new Complex128("837.5426799999999+82.9007024i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("8092.32+2537.82i")}};
        createMatrices();

        Complex128[] act = new Complex128[A.numRows*B.numCols];

        DenseCooFieldMatMult.standard(A.data, A.shape,
                B.data, B.rowIndices, B.colIndices, B.shape, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        DenseCooFieldMatMult.concurrentStandard(
                A.data, A.shape,
                B.data, B.rowIndices, B.colIndices, B.shape, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        // ----------------------- sub-case 2 -----------------------
        sparseShape = new Shape(2, 3);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 2}};
        expEntries = new Complex128[][]{
                {new Complex128("368.567656-357.60642975999997i"), new Complex128("4.002000000000001-15.00072i")},
                {new Complex128("-39577.094399999994-5651.525600000002i"), new Complex128("8092.32+2537.82i")}};
        createMatrices();

        act = new Complex128[B.numRows*A.numCols];
        DenseCooFieldMatMult.standard(
                B.data, B.rowIndices, B.colIndices, B.shape,
                A.data, A.shape, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        act = new Complex128[B.numRows*A.numCols];
        DenseCooFieldMatMult.concurrentStandard(
                B.data, B.rowIndices, B.colIndices, B.shape,
                A.data, A.shape, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);
    }


    @Test
    void matVecMultTestCase() {
        // ----------------------- sub-case 1 -----------------------
        sparseShape = new Shape(2, 4);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 3}};
        bVecEntries = new Complex128[]{new Complex128("1.334+0.00824i"), new Complex128("324.5+4.1i"),
                new Complex128("-24.5-45.1i"), new Complex128("0.0+6.1255i")};
        expEntries = new Complex128[][]{{new Complex128("453.38398400000005-1617.1084799999999i"),
                new Complex128("-1436.79728-450.59177999999997i")}};
        createMatrices();
        createDenseVector();
        Complex128[] act = new Complex128[B.numRows];

        DenseCooFieldMatMult.standardVector(
                B.data, B.rowIndices, B.colIndices, B.shape,
                bvec.data, bvec.shape, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        act = new Complex128[B.numRows];
        DenseCooFieldMatMult.concurrentStandardVector(
                B.data, B.rowIndices, B.colIndices, B.shape,
                bvec.data, bvec.shape, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        // ----------------------- sub-case 2 -----------------------
        sparseSize = 2;
        sparseVecIndices = new int[]{1};
        bVecSparseEntries = new Complex128[]{new Complex128("1.334+0.00824i")};
        expEntries = new Complex128[][]{
                {new Complex128("0.664416-12.535974400000002i"),
                new Complex128("4.002000000000001+0.024720000000000002i"),
                new Complex128("0.28428000000000003-46.023i")}};
        createMatrices();
        createSparseVector();

        act = new Complex128[A.numRows];
        DenseCooFieldMatMult.standardVector(
                A.data, A.shape, bSparse.data, bSparse.indices, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        act = new Complex128[A.numRows];
        DenseCooFieldMatMult.concurrentStandardVector(
                A.data, A.shape, bSparse.data, bSparse.indices, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);

        act = new Complex128[A.numRows];
        DenseCooFieldMatMult.concurrentBlockedVector(
                A.data, A.shape, bSparse.data, bSparse.indices, act);
        assertArrayEquals(ArrayUtils.flatten(expEntries), act);
    }
}
