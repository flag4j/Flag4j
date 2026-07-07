package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.io.PrintOptions;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixToStringTests {

    Complex128[][] aEntries;
    CMatrix A;
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
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(50);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[    2+4.25i     -2.345E-4    9.4i        1.2598      ]\n" +
                " [ 56.25-0.0024i      0      -1.4545i  -3.356-84.2525i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 2 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(3);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[   2+4.25i     0   9.4i         1.26      ]\n" +
                " [ 56.25-0.002i  0  -1.455i  -3.356-84.253i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 3 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(0);
        PrintOptions.setPadding(4);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[  2+4i    0    9i      1     ]\n" +
                " [   56     0    -i    -3-84i  ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 4 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 2+4.25i  0   9.4i       1.26     ]\n" +
                " [  56.25   0  -1.45i  -3.36-84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 5 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(1);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[              ...               ]\n" +
                " [ 56.25  0  -1.45i  -3.36-84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 6 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(3);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 2+4.25i  0  ...      1.26     ]\n" +
                " [  56.25   0  ...  -3.36-84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 6 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(2);
        PrintOptions.setCentering(true);
        exp = "shape: (2, 4)\n" +
                "[[ 2+4.25i  ...      1.26     ]\n" +
                " [  56.25   ...  -3.36-84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ sub-case 6 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(2, 4.25), new Complex128(-0.0002345),
                        new Complex128(0, 9.4), new Complex128(1.2598)},
                {new Complex128(56.25, -0.0024), Complex128.ZERO,
                        new Complex128(0, -1.4545), new Complex128(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(3);
        PrintOptions.setPadding(3);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(false);
        exp = "shape: (2, 4)\n" +
                "[[2+4.25i        0   9.4i      1.26             ]\n" +
                " [56.25-0.002i   0   -1.455i   -3.356-84.253i   ]]";

        assertEquals(exp, A.toString());
    }
}
