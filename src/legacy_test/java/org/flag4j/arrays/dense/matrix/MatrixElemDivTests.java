package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixElemDivTests {

    Matrix A, B, result, expResult;
    CMatrix BC, resultC, expResultC;
    double[][] entriesA, entriesB;
    Complex128[][] entriesBC;

    private double[] getExp(double[] src1, double[] src2) {
        double[] result = new double[src1.length];

        for(int i=0; i<result.length; i++)
            result[i] = src1[i]/src2[i];

        return result;
    }

    private Complex128[] getExp(double[] src1, Complex128[] src2) {
        Complex128[] result = new Complex128[src1.length];
        double divisor;

        for(int i=0; i<result.length; i++)
            result[i] = src2[i].invert().mult(src1[i]);

        return result;
    }


    @Test
    void elemDivTestCase() {
        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94, -0.4442}, {0.0000234, 1333.4, 44.5, 134.3}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);
        expResult = new Matrix(A.shape, getExp(A.data, B.data));

        result = A.div(B);

        assertArrayEquals(expResult.data, result.data);
        assertEquals(expResult.shape, result.shape);

        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94}, {0.0000234, 1333.4, 44.5}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));
    }


    @Test
    void elemDivComplexTestCase() {
        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new Complex128[][]{{new Complex128(1.4, 5), new Complex128(0, -1), new Complex128(1.3), Complex128.ZERO},
                {new Complex128(4.55, -93.2), new Complex128(-2, -13), new Complex128(8.9), new Complex128(0, 13)}};
        A = new Matrix(entriesA);
        BC = new CMatrix(entriesBC);
        expResultC = new CMatrix(A.shape, getExp(A.data, BC.data));

        resultC = A.div(BC);

        for(int i = 0; i<resultC.data.length; i++) {
            if(Double.isNaN((expResultC.data[i]).re)) {
                assertTrue(Double.isNaN((resultC.data[i]).re));
            } else {
                assertEquals((expResultC.data[i]).re, (resultC.data[i]).re);
            }

            if(Double.isNaN((expResultC.data[i]).im)) {
                assertTrue(Double.isNaN((resultC.data[i]).im));
            } else {
                assertEquals((expResultC.data[i]).im, (resultC.data[i]).im);
            }
        }

        assertEquals(expResultC.shape, resultC.shape);

        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new Complex128[][]{{new Complex128(1.4, 5), new Complex128(0, -1), new Complex128(1.3)},
                {new Complex128(4.55, -93.2), new Complex128(-2, -13), new Complex128(8.9)}};
        A = new Matrix(entriesA);
        BC = new CMatrix(entriesBC);

        assertThrows(LinearAlgebraException.class, ()->A.div(BC));
    }
}
