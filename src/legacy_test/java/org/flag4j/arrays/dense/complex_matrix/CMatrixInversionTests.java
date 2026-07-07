package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.Invert;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixInversionTests {
    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void invTestCase() {
        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(2.4, 1), new Complex128(9), new Complex128(0, 4)},
                {new Complex128(34), new Complex128(6, 2), new Complex128(24, 7)},
                {new Complex128(6), new Complex128(25, -14.3), new Complex128(14, 43)}
        };
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(-0.020890310281788768, -0.08401682835191364), new Complex128(0.030417400447454772, 0.00290515089267757), new Complex128(-0.011345283148180455, 0.020625927995890685)},
                {new Complex128(0.1502148762820325, 0.01519693661063067), new Complex128(-0.008811227842584712, -0.004051902797816908), new Complex128(-0.007622836636394531, -0.008153804836961882)},
                {new Complex128(0.021439567207676365, 0.09645349255347833), new Complex128(-2.306593523821924E-4, -0.002301110100512132), new Complex128(0.008806824872674935, -0.029115034319768644)}
        };
        exp = new CMatrix(expEntries);
        assertEquals(exp, Invert.inv(A));

        // --------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(1), new Complex128(2)},
                {new Complex128(-2), new Complex128(-4)}
        }; // This matrix is singular.
        A = new CMatrix(aEntries);

        assertThrows(RuntimeException.class, ()-> Invert.inv(A));


        // --------------------- sub-case 3 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(2.4, 1), new Complex128(9), new Complex128(0, 4)},
                {new Complex128(34), new Complex128(6, 2), new Complex128(24, 7)}
        };
        A = new CMatrix(aEntries);

        assertThrows(LinearAlgebraException.class, ()-> Invert.inv(A));
        assertThrows(LinearAlgebraException.class, ()-> Invert.inv(A.T()));
    }
}
