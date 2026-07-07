package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixVectorTests {

    double[][] aEntries;
    double[] expEntries;
    Complex128[] expCEntries;

    Matrix A;
    Vector exp;
    CVector expC;


    @Test
    void matVecMultTestCase() {
        double[] bEntries;
        Vector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{1.666, -0.9345341, 0.0};
        B = new Vector(bEntries);
        expEntries = new double[]{-90.8659724794,
                -2068.717076035,
                205.65924851056695,
                118.90475382059999};
        exp = new Vector(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{1.666, -0.9345341, 0.0, 993.3};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matVecMultComplexTestCase() {
        Complex128[] bEntries;
        CVector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("1.666+1.0i"),
                new Complex128("-0.0-0.9345341i"),
                new Complex128("0.0")};
        B = new CVector(bEntries);
        expCEntries = new Complex128[]{new Complex128("1.8715844-91.6141568794i"),
                new Complex128("-1553.4617-1447.705376035i"),
                new Complex128("205.65936999999997+123.444878510567i"),
                new Complex128("130.337844+66.8009098206i")};
        expC = new CVector(expCEntries);

        assertEquals(expC, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("1.666+1.0i"),
                new Complex128("-0.0-0.9345341i")};
        B = new CVector(bEntries);

        CVector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matVecMultSparseTestCase() {
        double[] bEntries;
        int[] indices;
        CooVector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341};
        indices = new int[]{1};
        B = new CooVector(3, bEntries, indices);
        expEntries = new double[]{-92.7375568794,
                -515.255376035,
                -0.00012148943299999999,
                -11.4330901794};
        exp = new Vector(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341};
        indices = new int[]{1};
        B = new CooVector(14, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matVecMultSparseComplexTestCase() {
        Complex128[] bEntries;
        int[] indices;
        CooCVector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i")};
        indices = new int[]{2};
        B = new CooCVector(3, bEntries, indices);
        expCEntries = new Complex128[]{
                new Complex128("-0.00011494769430000002+0.0011500500000000001i"),
                new Complex128("0.8629674786220001-8.633977i"),
                new Complex128("0.0"),
                new Complex128("9273.596817143-92782.20049999999i")};
        expC = new CVector(expCEntries);

        assertEquals(expC, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i")};
        indices = new int[]{2};
        B = new CooCVector(9, bEntries, indices);

        CooCVector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }
}
