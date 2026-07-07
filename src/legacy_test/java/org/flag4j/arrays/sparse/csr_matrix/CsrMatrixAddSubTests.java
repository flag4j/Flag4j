package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.ops.dense_sparse.csr.real.RealCsrDenseOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrOps;
import org.flag4j.linalg.ops.sparse.csr.real_complex.RealComplexCsrOps;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixAddSubTests {

    static CsrMatrix A;
    static CsrMatrix B;
    static Matrix denseA;
    static Matrix denseB;
    static CsrMatrix expAdd;
    static CsrMatrix expAsubB;
    static CsrMatrix expBsubA;
    static Matrix expAddDense;
    static Matrix expAsubBDense;
    static Matrix expBsubADense;
    static double[][] aEntries;
    static double[][] bEntries;

    static Complex128[][] bCmpEntries;
    static CsrCMatrix BCmp;
    static CsrCMatrix expAddCmp;
    static CsrCMatrix expAsubBCmp;
    static CMatrix expAddDenseCmp;
    static CMatrix expAsubBDenseCmp;
    static CMatrix denseBCmp;
    static double b;
    static Complex128 bCmp;

    private static void makeRealMatrices() {
        denseA = new Matrix(aEntries);
        denseB = new Matrix(bEntries);
        A = denseA.toCsr();
        B = denseB.toCsr();
        expAddDense = denseA.add(denseB);
        expAsubBDense = denseA.sub(denseB);
        expBsubADense = denseB.sub(denseA);
        expAdd = expAddDense.toCsr();
        expAsubB = expAsubBDense.toCsr();
        expBsubA = expBsubADense.toCsr();
    }

    private static void makeRealConstMatrices() {
        denseA = new Matrix(aEntries);
        A = denseA.toCsr();
        expAddDense = denseA.add(b);
        expAsubBDense = denseA.sub(b);
    }

    private static void makeCmpMatrices() {
        denseA = new Matrix(aEntries);
        denseBCmp = new CMatrix(bCmpEntries);
        A = denseA.toCsr();
        BCmp = denseBCmp.toCsr();
        expAddDenseCmp = denseA.add(denseBCmp);
        expAsubBDenseCmp = denseA.sub(denseBCmp);
        expAddCmp = expAddDenseCmp.toCsr();
        expAsubBCmp = expAsubBDenseCmp.toCsr();
    }

    private static void makeCmpConstMatrices() {
        denseA = new Matrix(aEntries);
        A = denseA.toCsr();
        expAddDenseCmp = denseA.add(bCmp);
        expAsubBDenseCmp = denseA.sub(bCmp);
    }


    @Test
    void addSubSpTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {0, 0, 1.34, 0, 1.3, 0},
                {0, 0, 0, 56.1, 0, 0},
                {36.1, 13.2, 0, 0, 0, 8},
                {0, 0, 0, 0, 0, 0},
        };
        makeRealMatrices();
        assertEquals(expAdd, A.add(B));
        assertEquals(expAsubB, A.sub(B));
        assertEquals(expBsubA, B.sub(A));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
        assertThrows(LinearAlgebraException.class, ()->B.add(A));
        assertThrows(LinearAlgebraException.class, ()->B.sub(A));
    }


    @Test
    void addSubDeTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bEntries = new double[][]{
                {1, 2, 4, 25.23, 1.5, -0.204},
                {-99.34, 15.23, 1.34, 0, 1.3, 0},
                {50.2, 3491.3, 2.42, 56.1, Double.POSITIVE_INFINITY, 0},
                {36.1, 13.2, 2.5621, 83458934.245, 0, 8},
                {1.345, 983.3, 0, -9.234, 4.52, 0},
        };
        makeRealMatrices();
        assertEquals(expAddDense, RealCsrDenseOps.add(A, denseB));
        assertEquals(expAsubBDense, RealCsrDenseOps.sub(A, denseB));
        assertEquals(expBsubADense, RealCsrDenseOps.sub(B, denseA));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
        assertThrows(LinearAlgebraException.class, ()->B.add(A));
        assertThrows(LinearAlgebraException.class, ()->B.sub(A));
    }


    @Test
    void addSubSpCmpTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmpEntries = new Complex128[aEntries.length][aEntries[0].length];
        ArrayBuilder.fill(bCmpEntries, Complex128.ZERO);
        bCmpEntries[0][0] = new Complex128(23, 1.34);
        bCmpEntries[1][0] = new Complex128(0.133, -41.4);
        bCmpEntries[1][3] = new Complex128(-4.1, -34.1);
        bCmpEntries[3][1] = new Complex128(922.1);
        bCmpEntries[3][5] = new Complex128(34.5, 135);
        bCmpEntries[4][4] = new Complex128(23.501, 100.23);
        makeCmpMatrices();
        assertEquals(expAddCmp, RealComplexCsrOps.add(BCmp, A));
        assertEquals(expAsubBCmp, RealComplexCsrOps.sub(A, BCmp));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()-> RealComplexCsrOps.add(BCmp, A));

        assertThrows(LinearAlgebraException.class, ()-> RealComplexCsrOps.sub(A, BCmp));
    }


    @Test
    void addSubDeCmpTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmpEntries = new Complex128[aEntries.length][aEntries[0].length];
        ArrayBuilder.fill(bCmpEntries, Complex128.ZERO);
        bCmpEntries[0][0] = new Complex128(23, 1.34);
        bCmpEntries[1][0] = new Complex128(0.133, -41.4);
        bCmpEntries[1][3] = new Complex128(-4.1, -34.1);
        bCmpEntries[3][1] = new Complex128(922.1);
        bCmpEntries[3][5] = new Complex128(34.5, 135);
        bCmpEntries[4][4] = new Complex128(23.501, 100.23);
        makeCmpMatrices();

        assertEquals(expAddDenseCmp, RealFieldDenseCsrOps.add(A, denseBCmp));
        assertEquals(expAsubBDenseCmp, RealFieldDenseCsrOps.sub(A, denseBCmp));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->RealFieldDenseCsrOps.add(A, denseBCmp));
        assertThrows(LinearAlgebraException.class, ()->RealFieldDenseCsrOps.sub(A, denseBCmp));
    }


    @Test
    void addSubRealConstTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        b = 1.2;
        CsrMatrix expAdd = new Matrix(new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324+b, 0, 0, 13.4+b, 0, 0},
                {0, 0, 0, -23.5+b, 0, 0},
                {0, 14.1+b, 0, 0, 0, 0},
                {0, 0, 0, 9.143+b, 1.4+b, -2.1+b}}).toCsr();
        CsrMatrix expSub = new Matrix(new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324-b, 0, 0, 13.4-b, 0, 0},
                {0, 0, 0, -23.5-b, 0, 0},
                {0, 14.1-b, 0, 0, 0, 0},
                {0, 0, 0, 9.143-b, 1.4-b, -2.1-b}}).toCsr();
        makeRealConstMatrices();

        assertEquals(expAdd, A.add(b));
        assertEquals(expSub, A.sub(b));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        b = -2341.200239867181;
        expAdd = new Matrix(new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324+b, 0, 0, 13.4+b, 0, 0},
                {0, 0, 0, -23.5+b, 0, 0},
                {0, 14.1+b, 0, 0, 0, 0},
                {0, 0, 0, 9.143+b, 1.4+b, -2.1+b}}).toCsr();
        expSub = new Matrix(new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324-b, 0, 0, 13.4-b, 0, 0},
                {0, 0, 0, -23.5-b, 0, 0},
                {0, 14.1-b, 0, 0, 0, 0},
                {0, 0, 0, 9.143-b, 1.4-b, -2.1-b}}).toCsr();
        makeRealConstMatrices();

        assertEquals(expAdd, A.add(b));
        assertEquals(expSub, A.sub(b));
    }


    @Test
    void addSubCmpConstTests() {
        CsrCMatrix expAdd;
        CsrCMatrix expSub;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmp = new Complex128(-0.234, 155.2);
        expAdd = new CooCMatrix(new Shape(5, 6),
                new Complex128[]{bCmp.add(1.2324), bCmp.add(13.4), bCmp.add(-23.5), bCmp.add(14.1),
                        bCmp.add(9.143), bCmp.add(1.4), bCmp.add(-2.1)},
                new int[]{1, 1, 2, 3, 4, 4, 4},
                new int[]{0, 3, 3, 1, 3, 4, 5}).toCsr();
        expSub = new CooCMatrix(new Shape(5, 6),
                new Complex128[]{new Complex128(1.2324).sub(bCmp), new Complex128(13.4).sub(bCmp),
                        new Complex128(-23.5).sub(bCmp), new Complex128(14.1).sub(bCmp), new Complex128(9.143).sub(bCmp),
                        new Complex128(1.4).sub(bCmp), new Complex128(-2.1).sub(bCmp)},
                new int[]{1, 1, 2, 3, 4, 4, 4},
                new int[]{0, 3, 3, 1, 3, 4, 5}).toCsr();
        makeCmpConstMatrices();

        assertEquals(expAdd, A.add(bCmp));
        assertEquals(expSub, A.sub(bCmp));

        // ---------------------- sub-case 2 ----------------------
        bCmp = new Complex128(-92.14, -7884.7761);
        expAdd = new CooCMatrix(new Shape(5, 6),
                new Complex128[]{bCmp.add(1.2324), bCmp.add(13.4), bCmp.add(-23.5), bCmp.add(14.1),
                        bCmp.add(9.143), bCmp.add(1.4), bCmp.add(-2.1)},
                new int[]{1, 1, 2, 3, 4, 4, 4},
                new int[]{0, 3, 3, 1, 3, 4, 5}).toCsr();
        expSub = new CooCMatrix(new Shape(5, 6),
                new Complex128[]{new Complex128(1.2324).sub(bCmp), new Complex128(13.4).sub(bCmp),
                        new Complex128(-23.5).sub(bCmp), new Complex128(14.1).sub(bCmp), new Complex128(9.143).sub(bCmp),
                        new Complex128(1.4).sub(bCmp), new Complex128(-2.1).sub(bCmp)},
                new int[]{1, 1, 2, 3, 4, 4, 4},
                new int[]{0, 3, 3, 1, 3, 4, 5}).toCsr();
        makeCmpConstMatrices();

        assertEquals(expAdd, A.add(bCmp));
        assertEquals(expSub, A.sub(bCmp));
    }
}
