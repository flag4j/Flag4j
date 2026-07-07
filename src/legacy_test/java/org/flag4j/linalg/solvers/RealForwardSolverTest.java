package org.flag4j.linalg.solvers;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.solvers.exact.triangular.RealForwardSolver;
import org.flag4j.util.exceptions.SingularMatrixException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class RealForwardSolverTest {

    RealForwardSolver solver;
    double[][] lEntries;
    Matrix L;

    @Test
    void solveVectorTestCase() {
        solver = new RealForwardSolver();
        double[] bEntries, expEntries;
        Vector b, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new double[][]{
                {1.4, 0, 0, 0, 0, 0},
                {-924.5561, 1.5, 0, 0, 0, 0},
                {105.63, -0.0, 14, 0, 0, 0},
                {14.5, -566.1, 0.00024, 4.1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 1, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 5.1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24, 795.1};
        b = new Vector(bEntries);
        expEntries = new double[]{0.9571428571428573, 593.3548447619049, -7.221642857142858, 81887.60994670248, -584095.4301186552, 5.156070982220032E8};
        exp = new Vector(expEntries);

        Assertions.assertEquals(exp, solver.solve(L, b));

        // ---------------------- sub-case 2 ----------------------
        lEntries = new double[][]{
                {1.4, 0, 0, 0, 0, 0},
                {-924.5561, 1.5, 0, 0, 0, 0},
                {105.63, -0.0, 14, 0, 0, 0},
                {14.5, -566.1, 0.00024, 4.1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 0, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 5.1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24, 795.1};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(SingularMatrixException.class, ()->solver.solve(L, finalB));

        // ---------------------- sub-case 3 ----------------------
        lEntries = new double[][]{
                {1.4, 0, 0, 0, 0, 0},
                {-924.5561, 1.5, 0, 0, 0, 0},
                {105.63, -0.0, 14, 0, 0, 0},
                {14.5, -566.1, 0.00024, 4.1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 1, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 5.1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24};
        b = new Vector(bEntries);

        Vector finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));

        // ---------------------- sub-case 4 ----------------------
        lEntries = new double[][]{
                {1.4, 0, 0, 0, 0},
                {-924.5561, 1.5, 0, 0, 0},
                {105.63, -0.0, 14, 0, 0},
                {14.5, -566.1, 0.00024, 4.1, 0},
                {3.45, 1.56, -99.356, 7.125, 1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24, 13.4};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB2));


        // ---------------------- sub-case 4 ----------------------
        lEntries = new double[][]{
                {1.4, 0, 0, 0, 0},
                {-924.5561, 1.5, 0, 0, 0},
                {105.63, -0.0, 14, 0, 0},
                {14.5, -566.1, 0.00024, 4.1, 0}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1};
        b = new Vector(bEntries);

        Vector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB3));
    }


    @Test
    void solveMatrixTestCase() {
        solver = new RealForwardSolver();
        double[][] bEntries, expEntries;
        Matrix B, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new double[][]{
                {1.4, 0, 0, 0, 0, 0},
                {-924.5561, 1.5, 0, 0, 0, 0},
                {105.63, -0.0, 14, 0, 0, 0},
                {14.5, -566.1, 0.00024, 4.1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 1, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 5.1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[][]{
                {1.324, -5.345, 1.55},
                {-0.35, 2.55, 1000345.2},
                {-8, 2.1, 56.2},
                {1, 2, 3},
                {100, 0, 2.5},
                {1.5, -693.5, 2.1}
        };
        B = new Matrix(bEntries);
        expEntries = new double[][]{
                {0.9457142857142858, -3.817857142857143, 1.1071428571428572},
                {582.6772744761905, -2351.5154069047617, 667579.2104547619},
                {-7.706842857142858, 28.955732142857148, -4.339107142857143},
                {80448.99855450512, -324667.19996819267, 9.217477511851482E7},
                {-574777.0750422318, 2319812.2611380727, -6.577871287236996E8},
                {5.0738121751018417E8, -2.0478013119683797E9, 5.806578804847218E11}
        };
        exp = new Matrix(expEntries);

        Assertions.assertEquals(exp, solver.solve(L, B));
    }


    @Test
    void solveUnitVectorTestCase() {
        solver = new RealForwardSolver(true);
        double[] bEntries, expEntries;
        Vector b, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new double[][]{
                {1, 0, 0, 0, 0, 0},
                {-924.5561, 1, 0, 0, 0, 0},
                {105.63, -0.0, 1, 0, 0, 0},
                {14.5, -566.1, 0.00024, 1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 1, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24, 795.1};
        b = new Vector(bEntries);
        expEntries = new double[]{1.34, 1244.005174, -141.5442, 704066.8329720079, -5031484.481532196, 2.265173786917746E10};
        exp = new Vector(expEntries);

        Assertions.assertEquals(exp, solver.solve(L, b));

        // ---------------------- sub-case 3 ----------------------
        lEntries = new double[][]{
                {1, 0, 0, 0, 0, 0},
                {-924.5561, 1, 0, 0, 0, 0},
                {105.63, -0.0, 1, 0, 0, 0},
                {14.5, -566.1, 0.00024, 1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 1, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24};
        b = new Vector(bEntries);

        Vector finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));

        // ---------------------- sub-case 4 ----------------------
        lEntries = new double[][]{
                {1, 0, 0, 0, 0},
                {-924.5561, 1, 0, 0, 0},
                {105.63, -0.0, 1, 0, 0},
                {14.5, -566.1, 0.00024, 1, 0},
                {3.45, 1.56, -99.356, 7.125, 1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1, 1000.24, 13.4};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB2));


        // ---------------------- sub-case 4 ----------------------
        lEntries = new double[][]{
                {1, 0, 0, 0, 0},
                {-924.5561, 1, 0, 0, 0},
                {105.63, -0.0, 1, 0, 0},
                {14.5, -566.1, 0.00024, 1, 0}
        };
        L = new Matrix(lEntries);
        bEntries = new double[]{1.34, 5.1, 0.0, -145.1};
        b = new Vector(bEntries);

        Vector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB3));
    }


    @Test
    void solveUnitMatrixTestCase() {
        solver = new RealForwardSolver(true);
        double[][] bEntries, expEntries;
        Matrix B, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new double[][]{
                {1, 0, 0, 0, 0, 0},
                {-924.5561, 1, 0, 0, 0, 0},
                {105.63, -0.0, 1, 0, 0, 0},
                {14.5, -566.1, 0.00024, 1, 0, 0},
                {3.45, 1.56, -99.356, 7.125, 1, 0},
                {-9.412, 3.51, -5.6, 0.0013, 4502, 1}
        };
        L = new Matrix(lEntries);
        bEntries = new double[][]{
                {1.324, -5.345, 1.55},
                {1223.7622764000002, -4939.202354499999, 1001778.261955},
                {-147.85412, 566.6923499999999, -107.52649999999998},
                {692753.662155029, -2796003.0863886136, 5.671066546435318E8},
                {-4952373.673752486, 1.9985549871568494E7, -4.0422083746742477E9},
                {2.2295580269226757E10, -8.997492212072707E10, 1.8198017848717656E13}
        };
        B = new Matrix(bEntries);
        expEntries = new double[][]{
                {1.324, -5.345, 1.55},
                {2447.8745528000004, -9880.954708999998, 1003211.32391},
                {-287.70824, 1131.2846999999997, -271.253},
                {2078476.317545087, -8389534.31616184, 1.1350245626990833E9},
                {-1.979392622825704E7, 7.988881452647084E7, -1.2130850349531082E10},
                {1.1140782325607643E11, -4.4963431324546594E11, 7.281110112399847E13}
        };
        exp = new Matrix(expEntries);

        Assertions.assertEquals(exp, solver.solve(L, B));
    }
}
