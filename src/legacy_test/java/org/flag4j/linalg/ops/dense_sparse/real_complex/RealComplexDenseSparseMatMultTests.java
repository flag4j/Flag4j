package org.flag4j.linalg.ops.dense_sparse.real_complex;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatMult.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealComplexDenseSparseMatMultTests {

    double[][] aEntries;
    Complex128[][] expCEntries, bComplexEntries;
    Complex128[] act;

    Matrix A;
    CMatrix expC, BComplex;

    Complex128[] bEntries;
    CooCMatrix BSparseComplex;
    Shape shape;

    CooMatrix ASparse;
    CooVector ACooVector;
    double[] aSparseEntries;

    int[] rowIndices, colIndices;

    @Test
    void matMultTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        shape = new Shape(3, 2);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new Complex128[][]{{new Complex128("-92.7375568794+927.8378999999999i"), new Complex128("0.00143541-0.000246i")},
                {new Complex128("-515.255376035+5155.1225i"), new Complex128("-10.7763114+1.84684i")},
                {new Complex128("-0.00012148943299999999+0.0012154999999999998i"), new Complex128("0.0")},
                {new Complex128("-11.4330901794+114.3879i"), new Complex128("-115804.09409999999+19846.46i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[A.numRows*BSparseComplex.numCols];
        standard(A.data, A.shape,
                BSparseComplex.data, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*BSparseComplex.numCols];
        concurrentStandard(A.data, A.shape,
                BSparseComplex.data, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        shape = new Shape(3, 4);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("-1.04985560794+10.503789999999999i"), new Complex128("-92.7375568794+927.8378999999999i"), new Complex128("-0.00011494769430000002+0.0011500500000000001i")},
                {new Complex128("-10881.6915+1864.9i"), new Complex128("6434.2545-1102.7i"), new Complex128("-10.7763114+1.84684i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[BSparseComplex.numRows*A.numCols];
        standard(BSparseComplex.data, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape,
                A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[BSparseComplex.numRows*A.numCols];
        concurrentStandard(BSparseComplex.data, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape,
                A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 3 ----------------------
        aSparseEntries = new double[]{1, 9.43};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{2, 1};
        shape = new Shape(4, 3);
        ASparse = new CooMatrix(shape, aSparseEntries, rowIndices, colIndices);
        bComplexEntries = new Complex128[][]{
                {new Complex128(1.34, 13.4), new Complex128(234.6, 6)},
                {new Complex128(-9.55, 1.9414), new Complex128(9, 1)},
                {new Complex128(0.9923, -985.2), new Complex128(9234)}};
        BComplex = new CMatrix(bComplexEntries);
        expCEntries = new Complex128[][]{{new Complex128("0.9923-985.2i"), new Complex128("9234.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("-90.0565+18.307402i"), new Complex128("84.87+9.43i")},
                {new Complex128("0.0"), new Complex128("0.0")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[ASparse.numRows*BComplex.numCols];
        standard(ASparse.data, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.data, BComplex.shape, act);
        assertArrayEquals(expC.data, act);

        concurrentStandard(ASparse.data, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.data, BComplex.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 4 ----------------------
        aSparseEntries = new double[]{1, 9.43};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        shape = new Shape(2, 3);
        ASparse = new CooMatrix(shape, aSparseEntries, rowIndices, colIndices);
        bComplexEntries = new Complex128[][]{
                {new Complex128(1.34, 13.4), new Complex128(234.6, 6)},
                {new Complex128(-9.55, 1.9414), new Complex128(9, 1)},
                {new Complex128(0.9923, -985.2), new Complex128(9234)}};
        BComplex = new CMatrix(bComplexEntries);
        expCEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("2212.278+56.58i"), new Complex128("1.34+13.4i")},
                {new Complex128("0.0"), new Complex128("84.87+9.43i"), new Complex128("-9.55+1.9414i")},
                {new Complex128("0.0"), new Complex128("87076.62"), new Complex128("0.9923-985.2i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[BComplex.numRows*ASparse.numCols];
        standard(BComplex.data, BComplex.shape,
                ASparse.data, ASparse.rowIndices, ASparse.colIndices, ASparse.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[BComplex.numRows*ASparse.numCols];
        concurrentStandard(BComplex.data, BComplex.shape,
                ASparse.data, ASparse.rowIndices, ASparse.colIndices, ASparse.shape, act);
        assertArrayEquals(expC.data, act);
    }


    @Test
    void matVecMultTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 0};
        shape = new Shape(3, 1);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new Complex128[][]{{new Complex128("-92.7361214694+927.8376539999999i")},
                {new Complex128("-526.0316874350001+5156.969340000001i")},
                {new Complex128("-0.00012148943299999999+0.0012154999999999998i")},
                {new Complex128("-115815.52719017939+19960.8479i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[A.numRows*BSparseComplex.numCols];
        standardVector(A.data, A.shape, BSparseComplex.data, BSparseComplex.rowIndices, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*BSparseComplex.numCols];
        concurrentStandardVector(A.data, A.shape, BSparseComplex.data, BSparseComplex.rowIndices, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*BSparseComplex.numCols];
        blockedVector(A.data, A.shape, BSparseComplex.data, BSparseComplex.rowIndices, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*BSparseComplex.numCols];
        concurrentBlockedVector(A.data, A.shape, BSparseComplex.data, BSparseComplex.rowIndices, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45},
                {123.445},
                {78.234}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        shape = new Shape(3, 4);
        BSparseComplex = new CooCMatrix(shape, bEntries, rowIndices, colIndices);
        expCEntries = new Complex128[][]{{new Complex128("0.0")},
                {new Complex128("-1.04985560794+10.503789999999999i")},
                {new Complex128("-10881.6915+1864.9i")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[BSparseComplex.numRows*A.numCols];
        standardVector(BSparseComplex.data, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape, A.data,
                A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[BSparseComplex.numRows*A.numCols];
        concurrentStandardVector(BSparseComplex.data, BSparseComplex.rowIndices, BSparseComplex.colIndices, BSparseComplex.shape,
                A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 3 ----------------------
        aSparseEntries = new double[]{1, 9.43};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{2, 1};
        shape = new Shape(4, 3);
        ASparse = new CooMatrix(shape, aSparseEntries, rowIndices, colIndices);
        bComplexEntries = new Complex128[][]{
                {new Complex128(1.34, 13.4)},
                {new Complex128(-9.55, 1.9414)},
                {new Complex128(0.9923, -985.2)}};
        BComplex = new CMatrix(bComplexEntries);
        expCEntries = new Complex128[][]{{new Complex128("0.9923-985.2i")},
                {new Complex128("0.0")},
                {new Complex128("-90.0565+18.307402i")},
                {new Complex128("0.0")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[ASparse.numRows*BComplex.numCols];
        standardVector(ASparse.data, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.data, BComplex.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[ASparse.numRows*BComplex.numCols];
        concurrentStandardVector(ASparse.data, ASparse.rowIndices, ASparse.colIndices, ASparse.shape,
                BComplex.data, BComplex.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 4 ----------------------
        aSparseEntries = new double[]{9.43};
        rowIndices = new int[]{1};
        ACooVector = new CooVector(2, aSparseEntries, rowIndices);
        bComplexEntries = new Complex128[][]{
                {new Complex128(1.34, 13.4), new Complex128(234.6, 6)},
                {new Complex128(-9.55, 1.9414), new Complex128(9, 1)},
                {new Complex128(0.9923, -985.2), new Complex128(9234)}};
        BComplex = new CMatrix(bComplexEntries);
        expCEntries = new Complex128[][]{{new Complex128("2212.278+56.58i")},
                {new Complex128("84.87+9.43i")},
                {new Complex128("87076.62")}};
        expC = new CMatrix(expCEntries);

        act = new Complex128[BComplex.numRows];
        standardVector(BComplex.data, BComplex.shape,
                ACooVector.data, ACooVector.indices, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[BComplex.numRows];
        concurrentStandardVector(BComplex.data, BComplex.shape,
                ACooVector.data, ACooVector.indices, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[BComplex.numRows];
        blockedVector(BComplex.data, BComplex.shape,
                ACooVector.data, ACooVector.indices, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[BComplex.numRows];
        concurrentBlockedVector(BComplex.data, BComplex.shape,
                ACooVector.data, ACooVector.indices, act);
        assertArrayEquals(expC.data, act);
    }
}
