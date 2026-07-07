package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexLUTests {

    static Complex128[][] aEntries;
    static CMatrix A;

    CMatrix L;
    CMatrix U;
    Matrix P;
    Matrix Q;

    CMatrix A_hat;

    static ComplexLU lu;

    static void setMatrices() {
        A = new CMatrix(aEntries);
    }


    @Test

    void noPivotTestCase() {
        lu = new ComplexLU(LU.Pivoting.NONE);

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i"), new Complex128("9.5-8.6i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i"), new Complex128("0.0+34.5i")},
                {new Complex128("1.05-47.1i"), new Complex128("66.2+3.5i"), new Complex128("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i")},
                {new Complex128("1.05-47.1i"), new Complex128("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- sub-case 3 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i"), new Complex128("9.5-8.6i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i"), new Complex128("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        A_hat = L.mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));
    }


    @Test

    void partialPivotTestCase() {
        lu = new ComplexLU();

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i"), new Complex128("9.5-8.6i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i"), new Complex128("0.0+34.5i")},
                {new Complex128("1.05-47.1i"), new Complex128("66.2+3.5i"), new Complex128("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i")},
                {new Complex128("1.05-47.1i"), new Complex128("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- sub-case 3 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i"), new Complex128("9.5-8.6i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i"), new Complex128("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U);

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));
    }


    @Test
    void partialFullTestCase() {
        lu = new ComplexLU(LU.Pivoting.FULL);

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i"), new Complex128("9.5-8.6i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i"), new Complex128("0.0+34.5i")},
                {new Complex128("1.05-47.1i"), new Complex128("66.2+3.5i"), new Complex128("45.33-0.009i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i")},
                {new Complex128("1.05-47.1i"), new Complex128("66.2+3.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- sub-case 3 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("1.255+45.1i"), new Complex128("-99.24+0.024i"), new Complex128("9.5-8.6i")},
                {new Complex128("6.466"), new Complex128("8.4-2.45i"), new Complex128("0.0+34.5i")}};
        setMatrices();

        lu.decompose(A);
        P = lu.getP().toDense();
        Q = lu.getQ().toDense();
        L = lu.getL();
        U = lu.getU();
        A_hat = P.T().mult(L).mult(U).mult(Q.T());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));
    }
}
