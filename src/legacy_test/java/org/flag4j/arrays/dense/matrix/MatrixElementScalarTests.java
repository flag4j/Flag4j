package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixElementScalarTests {
    double[][] aEntries, expEntries;
    Complex128[][] expEntriesC;
    double scalar;
    Complex128 scalarC;
    Matrix A;
    Matrix expResult;
    CMatrix expResultC;


    @Test
    void scalDivTestCase() {
        // -------------- sub-case 1 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalar = 1.44;
        expEntries = new double[][]{{1.334*(1.0/1.44), -2.3112*(1.0/1.44), 334.3*(1.0/1.44)},
                {4.13*(1.0/1.44), -35.33*(1.0/1.44), 6*(1.0/1.44)}};
        expResult = new Matrix(expEntries);

        assertArrayEquals(expResult.data, A.div(scalar).data);
        assertEquals(expResult.shape, A.div(scalar).shape);

        // -------------- sub-case 2 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalarC = new Complex128(1.3245, -42.5);
        expEntriesC = new Complex128[][]{{new Complex128(1.334).div(scalarC),
                new Complex128(-2.3112).div(scalarC),
                new Complex128(334.3).div(scalarC)},
                {new Complex128(4.13).div(scalarC),
                        new Complex128(-35.33).div(scalarC),
                        new Complex128(6).div(scalarC)}};
        expResultC = new CMatrix(expEntriesC);

        assertArrayEquals(expResultC.data, A.div(scalarC).data);
        assertEquals(expResultC.shape, A.div(scalarC).shape);
    }


    @Test
    void recipTestCase() {
        // -------------- sub-case 1 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalar = 1.44;
        expEntries = new double[][]{{1.0/1.334, 1.0/-2.3112, 1.0/334.3},
                {1.0/4.13, 1.0/-35.33, 1.0/6}};
        expResult = new Matrix(expEntries);

        assertArrayEquals(expResult.data, A.recip().data);
        assertEquals(expResult.shape, A.recip().shape);
    }
}
