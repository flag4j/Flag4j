package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixScalMultTests {
    double[][] aEntries, expEntries;
    Complex128[][] expCEntries;
    Matrix A, exp;
    CMatrix expC;
    Complex128 aC;
    double a;


    @Test
    void realTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new Matrix(aEntries);
        a = -1.66623;
        expEntries = new double[][]{{1*-1.66623, 2*-1.66623, 3.3434*-1.66623},
                {-0.221*-1.66623, 81.346*-1.66623, 90.234*-1.66623}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.mult(a));

        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new Matrix(aEntries);
        a = 0;
        expEntries = new double[2][3];
        expEntries[1][0] = -0.0;
        exp = new Matrix(expEntries);

        assertEquals(exp, A.mult(a));
    }


    @Test
    void complexTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new Matrix(aEntries);
        aC = new Complex128(-9.123, 19.23);
        expCEntries = new Complex128[][]{
                {new Complex128(1).mult(aC), new Complex128(2).mult(aC), new Complex128(3.3434).mult(aC)},
                {new Complex128(-0.221).mult(aC), new Complex128(81.346).mult(aC), new Complex128(90.234).mult(aC)}};
        expC = new CMatrix(expCEntries);

        assertEquals(expC, A.mult(aC));

        // -------------------- sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new Matrix(aEntries);
        aC = new Complex128(0);
        expCEntries = new Complex128[][]{
                {new Complex128(1).mult(aC), new Complex128(2).mult(aC), new Complex128(3.3434).mult(aC)},
                {new Complex128(-0.221).mult(aC), new Complex128(81.346).mult(aC), new Complex128(90.234).mult(aC)}};
        expC = new CMatrix(expCEntries);

        assertEquals(expC, A.mult(aC));
    }
}
