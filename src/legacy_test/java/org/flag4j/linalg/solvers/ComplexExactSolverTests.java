package org.flag4j.linalg.solvers;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.solvers.exact.ComplexExactSolver;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexExactSolverTests {

    static Complex128[][] aEntries;
    static Complex128[] bEntries;
    static Complex128[] expEntries;

    static CMatrix A;
    static CVector b;
    static CVector exp;

    private static void setMatrices() {
        A = new CMatrix(aEntries);
        b = new CVector(bEntries);
        exp = new CVector(expEntries);
    }


    @Test
    void solveTestCase() {
        ComplexExactSolver solver = new ComplexExactSolver();

        // ----------------- sub-case 1 -----------------
        aEntries = new Complex128[][]{
                {new Complex128(24.5, -0.35), new Complex128(25.6, -22.04), new Complex128(9.52, 2.3)},
                {new Complex128(-2.04), Complex128.ZERO, new Complex128(5.662)},
                {new Complex128(0.3456), new Complex128(0, -1.56), new Complex128(-23.5, -20.245)}
        };
        bEntries = new Complex128[]{new Complex128(35.6, -6.3), new Complex128(-0.0245, 0.024), new Complex128(0, 100.3)};
        expEntries = new Complex128[]{
                new Complex128("-5.259018068549946 - 7.623306153823768i"),
                new Complex128("-0.025597320154231645 + 8.146170435212893i"),
                new Complex128("-1.899134026817714 - 2.74241337933601i")};
        setMatrices();

        assertEquals(exp, solver.solve(A, b));

        // ----------------- sub-case 2 -----------------
        aEntries = new Complex128[][]{
                {new Complex128(24.5, -0.35), new Complex128(25.6, -22.04), new Complex128(9.52, 2.3)},
                {new Complex128(-2.04), Complex128.ZERO, new Complex128(5.662)},
                {new Complex128(0.3456), new Complex128(0, -1.56), new Complex128(-23.5, -20.245)}
        };
        bEntries = new Complex128[]{new Complex128(35.6, -6.3), new Complex128(-0.0245, 0.024),
                new Complex128(0, 100.3), new Complex128(94,52)};
        setMatrices();

        assertThrows(IllegalArgumentException.class, ()->solver.solve(A, b));

        // ----------------- sub-case 3 -----------------
        // TODO: This is not actually considered singular within the threshold. Its determinant is very small but not small enough.
//        aEntries = new Complex128[][]{
//                {new Complex128(24.5, -9.351), new Complex128(-2.56, 99.52)},
//                {new Complex128(-0.38037723083219255, 0.15548985842976704), new Complex128(0.00355, -1.56)}
//        }; // A singular matrix.
//        bEntries = new Complex128[]{new Complex128(35.6, -6.3), new Complex128(-0.0245, 0.024)};
//        expEntries = new Complex128[]{};
//        setMatrices();
//
//        assertThrows(SingularMatrixException.class, ()->solver.solve(A, b));
    }
}
