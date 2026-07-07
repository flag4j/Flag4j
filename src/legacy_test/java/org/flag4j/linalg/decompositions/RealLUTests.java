package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.decompositions.lu.RealLU;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealLUTests {

    double[][] aEntries, expLEntries, expUEntries, expPEntries, expQEntries;
    Matrix A, L, U, P, Q, expL, expU, expP, expQ;


    @Test
    void noPivotTestCase() {
        RealLU lu = new RealLU(LU.Pivoting.NONE);

        // -------------------------- sub-case 1 --------------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0},
                {0.0, 4.77, 71.578}};
        A = new Matrix(aEntries);
        expLEntries = new double[][]
                {{1.0, 0.0, 0.0},
                {642.3071428571429, 1.0, 0.0},
                {0.0, -0.0013123570799807352, 1.0}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {0.0, -3634.6815000000006, 5928.494928571429},
                {0.0, 0.0, 79.3583022931406}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();

        assertEquals(expL, L);
        assertEquals(expU, U);

        // -------------------------- sub-case 2 --------------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0},
                {0.0, 4.77, 71.578},
                {2.556, 1034, 0.00043}};
        A = new Matrix(aEntries);
        expLEntries = new double[][]
                {{1.0, 0.0, 0.0},
                {642.3071428571429, 1.0, 0.0},
                {0.0,-0.0013123570799807352, 1.0},
                {1.8257142857142858, -0.2816335351529425, 21.25190067067225}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {0.0, -3634.6815000000006, 5928.494928571429},
                {0.0, 0.0, 79.3583022931406}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();

        assertEquals(expL, L);
        assertEquals(expU, U);

        // -------------------------- sub-case 3 --------------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0}};
        A = new Matrix(aEntries);
        expLEntries = new double[][]
                {{1.0, 0.0},
                {642.3071428571429, 1.0}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {0.0, -3634.6815000000006, 5928.494928571429}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();

        assertEquals(expL, L);
        assertEquals(expU, U);
    }


    @Test
    void partialPivotTestCase() {
        RealLU lu = new RealLU(LU.Pivoting.PARTIAL);

        // -------------------------- sub-case 1 --------------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0},
                {0.0, 4.77, 71.578}};
        A = new Matrix(aEntries);
        expPEntries = new double[][]
                {{0.0, 1.0, 0.0},
                 {1.0, 0.0, 0.0},
                 {0.0, 0.0, 1.0}};
        expP = new Matrix(expPEntries);
        expLEntries = new double[][]
                {{1.0, 0.0, 0.0},
                {0.001556887559356338, 1.0, 0.0},
                {0.0, 0.8429363264507691, 1.0}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{899.23, 7.2, 0.0},
                {0.0, 5.658790409572634, -9.23},
                {0.0, 0.0, 79.3583022931406}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        P = lu.getP().toDense();

        assertEquals(expP, P.T());
        assertEquals(expL, L);
        assertEquals(expU, U);

        // -------------------------- sub-case 2 --------------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0},
                {0.0, 4.77, 71.578},
                {2.556, 1034, 0.00043}};
        A = new Matrix(aEntries);
        expPEntries = new double[][]
                {{0.0, 0.0, 0.0, 1.0},
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0, 0.0}};
        expP = new Matrix(expPEntries);
        expLEntries = new double[][]
                {{1.0, 0.0, 0.0},
                {0.0028424318583677144, 1.0, 0.0},
                {0.0, 0.0046132441125635, 1.0},
                {0.001556887559356338, 0.005472826318908125, -0.12895027255739647}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{899.23, 7.2, 0.0},
                {0.0, 1033.9795344906197, 0.00043},
                {0.0, 0.0, 71.57799801630503}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();

        assertEquals(expP, P.T());
        assertEquals(expL, L);
        assertEquals(expU, U);

        // -------------------------- sub-case 3 --------------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0}};
        A = new Matrix(aEntries);
        expPEntries = new double[][]
                {{0.0, 1.0},
                {1.0, 0.0}};
        expP = new Matrix(expPEntries);
        expLEntries = new double[][]
                {{1.0, 0.0},
                {0.001556887559356338, 1.0}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{899.23, 7.2, 0.0},
                {0.0, 5.658790409572634, -9.23}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();

        assertEquals(expP, P.T());
        assertEquals(expL, L);
        assertEquals(expU, U);
    }


    @Test
    void completePivotTestCase() {
        RealLU lu = new RealLU(LU.Pivoting.FULL);

        // -------------------------- sub-case 1 --------------------------
        aEntries = new double[][]
                {{1, 4, 7},
                {7, 8, 2},
                {9, 5, 1}};
        A = new Matrix(aEntries);
        expQEntries = new double[][]
                {{1, 0, 0},
                {0, 0, 1},
                {0, 1, 0}};
        expQ = new Matrix(expQEntries);
        expPEntries = new double[][]
                {{0, 0, 1},
                {1, 0, 0},
                {0, 1, 0}};
        expP = new Matrix(expPEntries);
        expLEntries = new double[][]
                {{1.0, 0.0, 0.0},
                {0.1111111111111111, 1.0, 0.0},
                {0.7777777777777778, 0.1774193548387097, 1.0}};
        expL = new Matrix(expLEntries);
        expUEntries = new double[][]
                {{9.0, 1.0, 5.0},
                {0, 6.888888888888889, 3.4444444444444446},
                {0, 0, 3.4999999999999996}};
        expU = new Matrix(expUEntries);

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();

        assertEquals(expP, P);
        assertEquals(expQ, Q);
        assertEquals(expL, L);
        assertEquals(expU, U);
    }
}
