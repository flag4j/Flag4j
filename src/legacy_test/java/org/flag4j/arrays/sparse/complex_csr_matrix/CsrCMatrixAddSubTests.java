package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.ops.dense_sparse.csr.field_ops.DenseCsrFieldOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real_complex.RealComplexCsrDenseOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrOps;
import org.flag4j.linalg.ops.sparse.csr.real_complex.RealComplexCsrOps;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class CsrCMatrixAddSubTests {
    static CsrCMatrix A;
    static CsrMatrix B;
    static CMatrix denseA;
    static Matrix denseB;
    static CsrCMatrix expAdd;
    static CsrCMatrix expAsubB;
    static CsrCMatrix expBsubA;
    static CMatrix expAddDense;
    static CMatrix expAsubBDense;
    static CMatrix expBsubADense;
    static Complex128[][] aEntries;
    static double[][] bEntries;

    static double b;
    static Complex128 bCmp;
    static Complex128[][] bEntriesCmp;
    static CsrCMatrix BCmp;
    static CMatrix denseBCmp;

    private static void makeMatrices() {
        denseA = new CMatrix(aEntries);
        A = denseA.toCsr();

        if(bEntries != null) {
            denseB = new Matrix(bEntries);
            B = denseB.toCsr();
            expAddDense = denseA.add(denseB);
            expAsubBDense = denseA.sub(denseB);
            expBsubADense = denseB.sub(denseA);
            expAdd = expAddDense.toCsr();
            expAsubB = expAsubBDense.toCsr();
            expBsubA = expBsubADense.toCsr();
        } else if(bEntriesCmp != null) {
            denseBCmp = new CMatrix(bEntriesCmp);
            BCmp = denseBCmp.toCsr();
            expAddDense = denseA.add(denseBCmp);
            expAsubBDense = denseA.sub(denseBCmp);
            expBsubADense = denseBCmp.sub(denseA);
            expAdd = expAddDense.toCsr();
            expAsubB = expAsubBDense.toCsr();
            expBsubA = expBsubADense.toCsr();
        }
    }

    private static void makeRealConstMatrices() {
        denseA = new CMatrix(aEntries);
        A = denseA.toCsr();
        expAddDense = denseA.add(b);
        expAsubBDense = denseA.sub(b);
    }

    private static void makeCmpConstMatrices() {
        denseA = new CMatrix(aEntries);
        A = denseA.toCsr();
        expAddDense = denseA.add(bCmp);
        expAsubBDense = denseA.sub(bCmp);
    }


    @AfterEach
    void resetAll() {
        A = null;
        B = null;
        denseA = null;
        denseB = null;
        expAdd = null;
        expAsubB = null;
        expBsubA = null;
        expAddDense = null;
        expAsubBDense = null;
        expBsubADense = null;
        aEntries = null;
        bEntries = null;

        bEntriesCmp = null;
        BCmp = null;
        denseBCmp = null;
    }


    @Test
    void addSubSpTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        bEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {0, 0, 1.34, 0, 1.3, 0},
                {0, 0, 0, 56.1, 0, 0},
                {36.1, 13.2, 0, 0, 0, 8},
                {0, 0, 0, 0, 0, 0},
        };
        makeMatrices();
        assertEquals(expAdd, RealComplexCsrOps.add(A, B));
        assertEquals(expAsubB, RealComplexCsrOps.sub(A, B));
        assertEquals(expBsubA, RealComplexCsrOps.sub(B, A));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()-> RealComplexCsrOps.add(A, B));
        assertThrows(LinearAlgebraException.class, ()-> RealComplexCsrOps.sub(A, B));
        assertThrows(LinearAlgebraException.class, ()-> RealComplexCsrOps.sub(B, A));
    }


    @Test
    void addSubDeTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        bEntries = new double[][]{
                {1, 2, 4, 25.23, 1.5, -0.204},
                {-99.34, 15.23, 1.34, 0, 1.3, 0},
                {50.2, 3491.3, 2.42, 56.1, Double.POSITIVE_INFINITY, 0},
                {36.1, 13.2, 2.5621, 83458934.245, 0, 8},
                {1.345, 983.3, 0, -9.234, 4.52, 0},
        };
        makeMatrices();
        assertEquals(expAddDense, RealComplexCsrDenseOps.add(A, denseB));
        assertEquals(expAsubBDense, RealComplexCsrDenseOps.sub(A, denseB));
        assertEquals(expBsubADense, RealFieldDenseCsrOps.sub(B, denseA));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrDenseOps.add(A, denseB));
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrDenseOps.sub(A, denseB));
        assertThrows(LinearAlgebraException.class, ()->RealFieldDenseCsrOps.add(B, denseA));
        assertThrows(LinearAlgebraException.class, ()->RealFieldDenseCsrOps.sub(B, denseA));
    }


    @Test
    void addSubSpCmpTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        bEntriesCmp = new Complex128[aEntries.length][aEntries[0].length];
        ArrayBuilder.fill(bEntriesCmp, Complex128.ZERO);
        bEntriesCmp[0][0] = new Complex128(23, 1.34);
        bEntriesCmp[1][0] = new Complex128(0.133, -41.4);
        bEntriesCmp[1][3] = new Complex128(-4.1, -34.1);
        bEntriesCmp[3][1] = new Complex128(922.1);
        bEntriesCmp[3][5] = new Complex128(34.5, 135);
        bEntriesCmp[4][4] = new Complex128(23.501, 100.23);
        makeMatrices();

        assertEquals(expAdd, A.add(BCmp));
        assertEquals(expAsubB, A.sub(BCmp));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(BCmp));
        assertThrows(LinearAlgebraException.class, ()->A.sub(BCmp));
    }


    @Test
    void addSubDeCmpTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        bEntriesCmp = new Complex128[aEntries.length][aEntries[0].length];
        ArrayBuilder.fill(bEntriesCmp, Complex128.ZERO);
        bEntriesCmp[0][0] = new Complex128(23, 1.34);
        bEntriesCmp[1][0] = new Complex128(0.133, -41.4);
        bEntriesCmp[1][3] = new Complex128(-4.1, -34.1);
        bEntriesCmp[3][1] = new Complex128(922.1);
        bEntriesCmp[3][5] = new Complex128(34.5, 135);
        bEntriesCmp[4][4] = new Complex128(23.501, 100.23);
        makeMatrices();
        assertEquals(expAddDense, DenseCsrFieldOps.add(A, denseBCmp));
        assertEquals(expAsubBDense, DenseCsrFieldOps.sub(A, denseBCmp));

        // ---------------------- sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->DenseCsrFieldOps.add(A, denseBCmp));
        assertThrows(LinearAlgebraException.class, ()->DenseCsrFieldOps.sub(A, denseBCmp));
    }


    @Test
    void addSubRealConstTests() {
        Complex128[][] expAddEntries;
        Complex128[][] expSubEntries;
        CsrCMatrix expAdd;
        CsrCMatrix expSub;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        b = 1.2;
        expAddEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").add(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").add(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").add(b), Complex128.ZERO, Complex128.ZERO,
                        new Complex128("0.8303067369813061+0.6672665799918752i").add(b), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").add(b),
                        new Complex128("0.4271149460271625+0.6885614097388274i").add(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO}};
        expAdd = new CMatrix(expAddEntries).toCsr();
        expSubEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        new Complex128("0.8303067369813061+0.6672665799918752i").sub(b), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").sub(b),
                        new Complex128("0.4271149460271625+0.6885614097388274i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO}};
        expSub = new CMatrix(expSubEntries).toCsr();
        makeRealConstMatrices();

        assertEquals(expAdd, A.add(b));
        assertEquals(expSub, A.sub(b));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        b = -2341.200239867181;
        expAddEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").add(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").add(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").add(b), Complex128.ZERO, Complex128.ZERO,
                        new Complex128("0.8303067369813061+0.6672665799918752i").add(b), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").add(b),
                        new Complex128("0.4271149460271625+0.6885614097388274i").add(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO}};
        expAdd = new CMatrix(expAddEntries).toCsr();
        expSubEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        new Complex128("0.8303067369813061+0.6672665799918752i").sub(b), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").sub(b),
                        new Complex128("0.4271149460271625+0.6885614097388274i").sub(b), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO}};
        expSub = new CMatrix(expSubEntries).toCsr();
        makeRealConstMatrices();

        assertEquals(expAdd, A.add(b));
        assertEquals(expSub, A.sub(b));
    }


    @Test
    void addSubCmpConstTests() {
        Complex128[][] expAddEntries;
        Complex128[][] expSubEntries;
        CsrCMatrix expAdd;
        CsrCMatrix expSub;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        bCmp = new Complex128(-0.234, 155.2);
        expAddEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").add(bCmp), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").add(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").add(bCmp), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i").add(bCmp), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").add(bCmp), new Complex128("0.4271149460271625+0.6885614097388274i").add(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        expAdd = new CMatrix(expAddEntries).toCsr();
        expSubEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").sub(bCmp), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").sub(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").sub(bCmp), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i").sub(bCmp), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").sub(bCmp), new Complex128("0.4271149460271625+0.6885614097388274i").sub(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        expSub = new CMatrix(expSubEntries).toCsr();
        makeCmpConstMatrices();

        assertEquals(expAdd, A.add(bCmp));
        assertEquals(expSub, A.sub(bCmp));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        bCmp = new Complex128(-92.14, -7884.7761);
        expAddEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").add(bCmp), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").add(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").add(bCmp), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i").add(bCmp), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").add(bCmp), new Complex128("0.4271149460271625+0.6885614097388274i").add(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        expAdd = new CMatrix(expAddEntries).toCsr();
        expSubEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6947844604184964+0.12617739377846993i").sub(bCmp), Complex128.ZERO, Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.5801548358148666+0.26523862055012315i").sub(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.030397833484514858+0.0339598081443111i").sub(bCmp), Complex128.ZERO, Complex128.ZERO, new Complex128("0.8303067369813061+0.6672665799918752i").sub(bCmp), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.08364434412041977+0.4996667151892932i").sub(bCmp), new Complex128("0.4271149460271625+0.6885614097388274i").sub(bCmp), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        expSub = new CMatrix(expSubEntries).toCsr();
        makeCmpConstMatrices();

        assertEquals(expAdd, A.add(bCmp));
        assertEquals(expSub, A.sub(bCmp));
    }
}
