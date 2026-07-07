package org.flag4j.linalg.solvers;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.solvers.exact.triangular.ComplexForwardSolver;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.SingularMatrixException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexForwardSolverTests {

    ComplexForwardSolver solver;
    Complex128[][] lEntries;
    CMatrix L;

    @Test
    void solveVectorTestCase() {
        solver = new ComplexForwardSolver();
        Complex128[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1.25, -9.25), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(2.45, 15.5), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[]{
                new Complex128(1.345, 2.45), new Complex128(-92.24, 7.24), new Complex128(0, 45.2)
        };
        b = new CVector(bEntries);

        expEntries = new Complex128[]{
                new Complex128("-0.2408177905308465 + 0.177948350071736i"),
                new Complex128("6.742734536376031 + 19.670300024162774i"),
                new Complex128("-6.458765618863107 + 2.169298473450951i")};
        exp = new CVector(expEntries);

        assertEquals(exp, solver.solve(L, b));

        // ---------------------- sub-case 2 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1.25, -9.25), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[]{
                new Complex128(1.345, 2.45), new Complex128(-92.24, 7.24), new Complex128(0, 45.2)
        };
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(SingularMatrixException.class, ()->solver.solve(L, finalB));

        // ---------------------- sub-case 3 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1.25, -9.25), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[]{
                new Complex128(1.345, 2.45), new Complex128(-92.24, 7.24), new Complex128(0, 45.2),
                new Complex128(2.45, -1300.13415)
        };
        b = new CVector(bEntries);

        CVector finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }


    @Test
    void solveMatrixTestCase() {
        solver = new ComplexForwardSolver();
        Complex128[][] bEntries, expEntries;
        CMatrix b, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1.25, -9.25), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(2.45, 15.5), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1.345, 2.45), new Complex128(2.4, -.36)},
                {new Complex128(-92.24, 7.24), new Complex128(0, 13.5)},
                {new Complex128(0, 45.2), new Complex128(-1.3567)}
        };
        b = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("-0.2408177905308465 + 0.177948350071736i"), new Complex128("0.0726542324246772 + 0.24964131994261118i")},
                {new Complex128("6.742734536376031 + 19.670300024162774i"), new Complex128("14.250401796793327 - 1.5933241423340474i")},
                {new Complex128("-6.458765618863107 + 2.169298473450951i"), new Complex128("0.3668372528597201 + 4.645626702643428i")}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, solver.solve(L, b));

        // ---------------------- sub-case 2 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1.25, -9.25), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(2.45, 15.5), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), Complex128.ZERO}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1.345, 2.45), new Complex128(2.4, -.36)},
                {new Complex128(-92.24, 7.24), new Complex128(0, 13.5)},
                {new Complex128(0, 45.2), new Complex128(-1.3567)}
        };
        b = new CMatrix(bEntries);

        CMatrix finalB = b;
        assertThrows(SingularMatrixException.class, ()->solver.solve(L, finalB));

        // ---------------------- sub-case 3 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1.25, -9.25), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(2.45, 15.5), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1.345, 2.45), new Complex128(2.4, -.36)},
                {new Complex128(-92.24, 7.24), new Complex128(0, 13.5)}
        };
        b = new CMatrix(bEntries);

        CMatrix finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }


    @Test
    void solveUnitVectorTestCase() {
        solver = new ComplexForwardSolver(true);
        Complex128[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(1), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[]{
                new Complex128(1.345, 2.45), new Complex128(-92.24, 7.24), new Complex128(0, 45.2)
        };
        b = new CVector(bEntries);

        expEntries = new Complex128[]{
                new Complex128("1.345 + 2.45i"),
                new Complex128("1008.1453000000001 + 2003.2728200000001i"),
                new Complex128("-169164.93180900003 + 79027.31987100001i")};
        exp = new CVector(expEntries);

        assertEquals(exp, solver.solve(L, b));

        // ---------------------- sub-case 2 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(1), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[]{
                new Complex128(1.345, 2.45), new Complex128(-92.24, 7.24),
                new Complex128(0, 45.2), new Complex128(2.45, -1300.13415)
        };
        b = new CVector(bEntries);

        CVector finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }


    @Test
    void solveUnitMatrixTestCase() {
        solver = new ComplexForwardSolver(true);
        Complex128[][] bEntries, expEntries;
        CMatrix b, exp;

        // ---------------------- sub-case 1 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(1), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1.345, 2.45), new Complex128(2.4, -.36)},
                {new Complex128(-92.24, 7.24), new Complex128(0, 13.5)},
                {new Complex128(0, 45.2), new Complex128(-1.3567)}
        };
        b = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("1.345 + 2.45i"), new Complex128("2.4 - 0.36i")},
                {new Complex128("1008.1453000000001 + 2003.2728200000001i"), new Complex128("1956.68016 - 283.5456i")},
                {new Complex128("-169164.93180900003 + 79027.31987100001i"), new Complex128("18799.102988000002 + 163512.690432i")}
        };
        exp = new CMatrix(expEntries);
        assertEquals(exp, solver.solve(L, b));

        // ---------------------- sub-case 2 ----------------------
        lEntries = new Complex128[][]{
                {new Complex128(1), Complex128.ZERO, Complex128.ZERO},
                {new Complex128(-815.5, 1.444), new Complex128(1), Complex128.ZERO},
                {new Complex128(0, -9.256), new Complex128(2.45, -83.2), new Complex128(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new Complex128[][]{
                {new Complex128(1.345, 2.45), new Complex128(2.4, -.36)},
                {new Complex128(-92.24, 7.24), new Complex128(0, 13.5)}
        };
        b = new CMatrix(bEntries);

        CMatrix finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }
}
