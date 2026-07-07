package org.flag4j.linalg.decompositions;

import org.flag4j.CustomAssertions;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.hess.RealHess;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class RealHessenburgTests {

    double[][] aEntries;
    Matrix A, Q, H, A_hat;

    RealHess hess;

    @Test
    void hessDecompTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        A = new Matrix(aEntries);
        hess = new RealHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape).round(), A.sub(A_hat).round());

        // ----------------------- sub-case 2 -----------------------
        aEntries = new double[][]{
                {1.44, 5.26, -35, 1.9},
                {0.00024, 16.7, 0, 13.56},
                {1.35345, -2.0525, 18056.2, 1.5},
                {1.56, 1.6, 1.656, 0.1}};
        A = new Matrix(aEntries);
        hess = new RealHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape), A.sub(A_hat).roundToZero(1.0e-10));

        // ----------------------- sub-case 2.1 -----------------------
        hess = new RealHess();
        hess.decompose(A);

        Assertions.assertEquals(H, hess.getH());

        // ----------------------- sub-case 3 -----------------------
        aEntries = new double[][]{
                {0.0, 0.0, 1.0E-5, 0.0, 8.0E-5, 6400.0, 0.0, 0.0, 0.0, 200000.0, 0.0},
                {0.0, 100.0, -75000.0, 0.0, 0.0, 20000.0, 25.0, 0.0, 0.0, -2.5E-4, 1.125E-9},
                {0.0, 0.002, 1.0E-8, 0.0, 40000.0, 0.0, 100000.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0E8, -80.0, 0.0, 0.0, 0.00244140625, 0.0, 1.0E-9, 0.0},
                {0.0, 40000.0, 0.0, 0.125, 0.0, -0.0012, 1.25, 0.0, -50.0, 0.0, 0.0},
                {0.0, 0.005, 0.0, 1.5625E-6, 7500.0, 1.0E-8, 281250.0, 0.0, 2.1875, 0.0, -31.25},
                {0.0, 0.0, -800.0, 0.2, 0.0, 224000.0, 0.01, 0.0, 0.0, 0.0, 1000.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0E-4, 0.0016384, 1638400.0, 81.92},
                {0.0, 0.0, 0.0, 0.0, 2.4E-5, 320.0, 0.0, 0.0, 3.0E8, -90000.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0E9}};
        A = new Matrix(aEntries);
        hess = new RealHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).mult(Q.T());

        // This is a really difficult matrix numerically speaking so the delta is quite permissive.
        Assertions.assertTrue(Q.isOrthogonal());
        CustomAssertions.assertEquals(A, A_hat, 1.0e-6);

        // ----------------------- sub-case 3.1 -----------------------
        hess.decompose(A, 1, 9);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).mult(Q.T());

        // This is a really difficult matrix numerically speaking so the delta is quite permissive.
        Assertions.assertTrue(Q.isOrthogonal());
        CustomAssertions.assertEquals(A, A_hat, 1.0e-6);
    }
}
