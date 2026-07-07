package org.flag4j.linalg.decompositions;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.hess.SymmHess;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class RealSymmHessTests {

    double[][] aEntries;
    Matrix A, Q, H, A_hat;
    SymmHess hess;

    @Test
    void symmHessDecompTestCase() {
        // ----------------------- sub-case 2 -----------------------
        aEntries = new double[][]{
                {100.2345, -8.1445},
                {-8.1445, 100.2345}};
        A = new Matrix(aEntries);
        hess = new SymmHess(true).decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape), A.sub(A_hat).round());

        // ----------------------- sub-case 2 -----------------------
        aEntries = new double[][]{
                {1, 2, 3},
                {2, 5, 6},
                {3, 6, 9}};
        A = new Matrix(aEntries);
        hess = new SymmHess(true).decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape), A.sub(A_hat).round());

        // ----------------------- sub-case 3 -----------------------
        aEntries = new double[][]{
                {1,  -4,   2.5, 15, 0   },
                {-4,  2,   8.1, 4,  1   },
                {2.5, 8.1, 4,  -9,  8.25},
                {15,  4,  -9, 10.3, 6   },
                {0,   1,  8.25, 6, -18.5}};
        A = new Matrix(aEntries);
        hess = new SymmHess(true).decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape), A.sub(A_hat).round());

        // ----------------------- sub-case 4 -----------------------
        aEntries = new double[][]{
                {1.4, -0.002, 14.51},
                {-0.002, 4.501, -9.14},
                {14.51, -9.14, 16.5}};
        A = new Matrix(aEntries);
        hess = new SymmHess(true).decompose(A);
        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape), A.sub(A_hat).round());

        // ----------------------- sub-case 5 -----------------------
        aEntries = new double[][]{
                {1.4, 5, 14.51},
                {-0.002, 4.501, -9.14},
                {11, -9.14, 16.5}};
        A = new Matrix(aEntries);
        hess = new SymmHess(true, true);
        Assertions.assertThrows(LinearAlgebraException.class, ()->hess.decompose(A));
    }
}
