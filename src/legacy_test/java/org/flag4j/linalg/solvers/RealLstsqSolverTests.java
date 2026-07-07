package org.flag4j.linalg.solvers;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.solvers.lstsq.RealLstsqSolver;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealLstsqSolverTests {
    static final RealLstsqSolver solver = new RealLstsqSolver();

    static double[][] aEntries;
    static Matrix A;


    @Test
    void solveTestCase() {
        double[] bEntries;
        double[] expEntries;
        Vector b;
        Vector exp;

        // ----------------- sub-case 1 -----------------
        aEntries = new double[][]{
                {1, 2, 1.55},
                {1, 2.3, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-54.8730919765167, 27.117873450750196, 1.3144161774298762};
        A = new Matrix(aEntries);
        b = new Vector(bEntries);
        exp = new Vector(expEntries);

        assertEquals(exp, solver.solve(A, b));

        // ----------------- sub-case 2 -----------------
        aEntries = new double[][]{
                {1.415, 35.6, 111.56},
                {1.14, -2.145, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-0.5953239804326492, -3.7452078451800968, 1.215236489070668};
        A = new Matrix(aEntries);
        b = new Vector(bEntries);
        exp = new Vector(expEntries);

        assertEquals(exp, solver.solve(A, b));
    }


    @Test
    void solveTestMatrix() {
        double[][] bEntries, expEntries;
        Matrix B, exp;

        // ----------------- sub-case 1 -----------------
        aEntries = new double[][]{
                {1, 2, 1.55},
                {1, 2.3, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[][]{
                {1.4, 90.2},
                {5.6, -0.024},
                {-99.35, 2.5}};
        expEntries = new double[][]{
                {-54.8730919765167, 667.0824135681675},
                {27.117873450750196, -289.32793215916524},
                {1.3144161774298762, 1.1441617742987609}};
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);
        A = new Matrix(aEntries);

        assertEquals(exp, solver.solve(A, B));

        // ----------------- sub-case 2 -----------------
        aEntries = new double[][]{
                {1.415, 35.6, 111.56},
                {1.14, -2.145, -1.444},
                {1, 2, -75.1},
                {56.4, 0.0024, 1},
                {-1, 35, 6.4}};
        bEntries = new double[][]{
                {1.4, 5.6, -99.35},
                {90.2, -0.024, 2.5},
                {20024.5, -9.42, 0.024},
                {25.6, 2.0, -895.2},
                {1.34, 245.006, 0.0345115}};
        expEntries = new double[][]{
                {11.236812466218115, -0.025918271804248924, -15.869733971321153},
                {235.3971728868125, 5.076829738809295, -0.6842081525682754},
                {-135.5921485602008, -0.9734072210388273, -0.39345431490836763}
        };
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        assertEquals(exp, solver.solve(A, B));
    }
}
