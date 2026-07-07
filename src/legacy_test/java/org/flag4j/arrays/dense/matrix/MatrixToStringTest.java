package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixToStringTest {

    double[][] aEntries;
    Matrix A;
    String exp;

    @BeforeAll
    static void setUp() {
        // Ensure print options are set to default before any test is run.
        PrintOptions.resetAll();
    }


    @AfterEach
    void tearDown() {
        // Ensure print options are reset after each test is run.
        PrintOptions.resetAll();
    }


    @Test
    void toStringTestCase() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(50);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 1.32239898234  2.46560324   1.45   -0.11234 ]\n" +
                " [    3.24E-4       1.456     -123.4  2341.56  ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 2 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(3);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 1.322  2.466   1.45   -0.112  ]\n" +
                " [   0    1.456  -123.4  2341.56 ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 3 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(0);
        PrintOptions.setPadding(4);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[  1    2     1       0    ]\n" +
                " [  0    1    -123    2342  ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 4 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 1.32  2.47   1.45    -0.11  ]\n" +
                " [  0    1.46  -123.4  2341.56 ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 5 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(1);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[           ...            ]\n" +
                " [ 0  1.46  -123.4  2341.56 ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 6 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(3);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 1.32  2.47  ...   -0.11  ]\n" +
                " [  0    1.46  ...  2341.56 ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 6 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(2);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 1.32  ...   -0.11  ]\n" +
                " [  0    ...  2341.56 ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 6 ------------------------
        aEntries = new double[][]{
                {1.32239898234, 2.46560324, 1.45, -0.11234},
                {0.000324, 1.456, -123.4, 2341.56}};
        A = new Matrix(aEntries);
        PrintOptions.setPrecision(3);
        PrintOptions.setPadding(3);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(false);
        exp = "shape: (2, 4)\n" +
                "[[1.322   2.466   1.45     -0.112    ]\n" +
                " [0       1.456   -123.4   2341.56   ]]";

        assertEquals(exp, A.toString());
    }
}
