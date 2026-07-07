package org.flag4j.arrays.dense.matrix;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatrixMultTranspose;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixMultTests {
    double[][] aEntries, expEntries;
    Complex128[][] expCEntries;

    Matrix A, exp;
    CMatrix expC;


    @Test
    void matMultTestCase() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new Matrix(bEntries);
        expEntries = new double[][]{{-90.8659724794, 8768.731856002458},
                {-2068.717076035, 37924.640881531595},
                {205.65924851056695, 1419.6289704199999},
                {118.90475382059999, 1978.9472913999998}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("1.8715844-91.6141568794i"), new Complex128("8768.731856002458-10.238294909999999i")},
                {new Complex128("-1553.4617-1447.705376035i"), new Complex128("37924.640881531595+8428.0382634i")},
                {new Complex128("205.65936999999997+123.444878510567i"), new Complex128("1419.6289704199999-1126.188735i")},
                {new Complex128("130.337844+66.8009098206i"), new Complex128("1978.9472913999998-846470.621682i")}};
        expC = new CMatrix(expCEntries);

        assertEquals(expC, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTestCase() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        CooMatrix B;
        Shape bShape;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7375568794, 0.00143541},
                {-515.255376035, -10.7763114},
                {-0.00012148943299999999, 0.0},
                {-11.4330901794, -115804.09409999999}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTestCase() {
        Complex128[] bEntries;
        int[] rowIndices, colIndices;
        CooCMatrix B;
        Shape bShape;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrix(bShape, bEntries, rowIndices, colIndices);
        expCEntries = new Complex128[][]{{new Complex128("-92.7375568794+927.8378999999999i"), new Complex128("0.00143541-0.000246i")},
                {new Complex128("-515.255376035+5155.1225i"), new Complex128("-10.7763114+1.84684i")},
                {new Complex128("-0.00012148943299999999+0.0012154999999999998i"), new Complex128("0.0")},
                {new Complex128("-11.4330901794+114.3879i"), new Complex128("-115804.09409999999+19846.46i")}};
        expC = new CMatrix(expCEntries);

        assertEquals(expC, A.mult(B));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooCMatrix(bShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void powTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{
                {-5.1236033781278044E7, 2.1045223503086835E7, -50637.093447080224},
                {-1.9781393621644562E8, 6.545422293841543E7, -195326.15048506213},
                {-1.1422366926135931E7, 6767794.115183196, -11311.889782356784}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.pow(3));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.pow(0));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.pow(1));

        // ---------------------- sub-case 4 ----------------------
        aEntries = new double[][]{{1.1234, 99.234},
                {-932.45, 551.35},
                {123.445, 0.00013}};
        A = new Matrix(aEntries);

        assertThrows(LinearAlgebraException.class, ()->A.pow(2));

        // ---------------------- sub-case 5 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()->A.pow(-1));
    }


    @Test
    void multTransposeTestCase() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new Matrix(bEntries).T();

        exp = A.mult(B.T());
        assertEquals(exp, A.multTranspose(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new Matrix(bEntries).T();

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.multTranspose(finalB));
    }


    @Test
    void multTransposeSparseTestCase() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        CooMatrix B;
        Shape bShape;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 2};
        bShape = new Shape(2, 3);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7375568794, 0.00143541},
                {-515.255376035, -10.7763114},
                {-0.00012148943299999999, 0.0},
                {-11.4330901794, -115804.09409999999}};
        exp = new Matrix(expEntries);

        Matrix act = new Matrix(
                new Shape(A.numRows, B.numRows),
                RealDenseSparseMatrixMultTranspose.multTranspose(
                        A.data, A.shape, B.data, B.rowIndices, B.colIndices, B.shape
                )
        );

        assertEquals(exp, act);
    }
}
