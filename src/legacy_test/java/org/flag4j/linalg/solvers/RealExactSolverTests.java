package org.flag4j.linalg.solvers;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.solvers.exact.RealExactSolver;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.SingularMatrixException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;


class RealExactSolverTests {
    static double[][] aEntries;
    static double[] bEntries;
    static double[] expEntries;

    static Matrix A;
    static Vector b;
    static Vector exp;

    private static void setMatrices() {
        A = new Matrix(aEntries);
        b = new Vector(bEntries);
        exp = new Vector(expEntries);
    }


    @Test
    void solveTestCase() {
        RealExactSolver solver = new RealExactSolver();

        // ----------------- sub-case 1 -----------------
        aEntries = new double[][]{
                {1.234, -0.024, 0.0},
                {100.4, 5.14, -1.444},
                {1.45, 985.1, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{1.3091068290163916, 8.976576125259498, 119.09551725559527};
        setMatrices();

        Assertions.assertEquals(exp, solver.solve(A, b));

        // ----------------- sub-case 2 -----------------
        aEntries = new double[][]{
                {1.234, -0.024, 0.0, 10.5},
                {100.4, 5.14, -1.444, 2.566},
                {1.45, 985.1, -75.1, 3.6},
                {2.45, 6.66, 0.0014, 51.6}};
        bEntries = new double[]{1.4, 5.6, -99.35, 0.0};
        expEntries = new double[]{-0.07323287523987436, -1.0515328249032092, -12.46496461132994, 0.13953643621412648};
        setMatrices();

        Assertions.assertEquals(exp, solver.solve(A, b));

        // ----------------- sub-case 3 -----------------
        aEntries = new double[][]{
                {1.234, -0.024, 0.0, 10.5},
                {100.4, 5.14, -1.444, 2.566},
                {1.45, 985.1, -75.1, 3.6}};
        bEntries = new double[]{1.4, 5.6, -99.35, 0.0};
        expEntries = new double[]{};
        setMatrices();

        assertThrows(LinearAlgebraException.class, ()->solver.solve(A, b));

        // ----------------- sub-case 4 -----------------
        aEntries = new double[][]{
                {1.234, -0.024, 0.0},
                {100.4, 5.14, -1.444},
                {1.45, 985.1, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35, 0.0};
        expEntries = new double[]{};
        setMatrices();

        assertThrows(IllegalArgumentException.class, ()->solver.solve(A, b));

        // ----------------- sub-case 5 -----------------
        aEntries = new double[][]{
                {1, 2, 0.0},
                {1, 2, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        setMatrices();

        assertThrows(SingularMatrixException.class, ()->solver.solve(A, b));
    }
}
