package org.flag4j;

import org.flag4j.arrays.backend.primitive_arrays.AbstractDenseDoubleNDArray;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Assertions;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public final class CustomAssertions {


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(Matrix exp, Matrix act) {
        Assertions.assertEquals(exp.shape, act.shape);
        for(int i = 0; i<exp.data.length; i++) {
            if(Double.isNaN(exp.data[i])) {
                assertTrue(Double.isNaN(act.data[i]));
            } else {
                Assertions.assertEquals(exp.data[i], act.data[i], 0);
            }
        }
    }


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(CMatrix exp, CMatrix act) {
        Assertions.assertEquals(exp.shape, act.shape);
        for(int i = 0; i<exp.data.length; i++) {
            if(exp.data[i].isNaN()) {
                if(Double.isNaN(exp.data[i].re)) {
                    assertTrue(Double.isNaN(act.data[i].re));
                } else {
                    Assertions.assertEquals(exp.data[i].re, act.data[i].re, 0);
                }

                if(Double.isNaN(exp.data[i].im)) {
                    assertTrue(Double.isNaN(act.data[i].im));
                } else {
                    Assertions.assertEquals(exp.data[i].im, act.data[i].im, 0);
                }
            } else {
                Assertions.assertEquals(exp.data[i], act.data[i]);
            }
        }
    }


    /**
     * Checks if sparse matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(CooMatrix exp, CooMatrix act) {
        Assertions.assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.rowIndices, act.colIndices);
        for(int i = 0; i<exp.data.length; i++) {
            if(Double.isNaN(exp.data[i])) {
                assertTrue(Double.isNaN(act.data[i]));
            } else {
                Assertions.assertEquals(exp.data[i], act.data[i], 0);
            }
        }
    }


    /**
     * Checks if sparse vectors are equal counting NaN equal to NaN.
     * @param exp Expected vector.
     * @param act Actual vector.
     */
    public static void assertEqualsNaN(CooVector exp, CooVector act) {
        Assertions.assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.indices, act.indices);
        for(int i = 0; i<exp.data.length; i++) {
            if(Double.isNaN(exp.data[i])) {
                assertTrue(Double.isNaN(act.data[i]));
            } else {
                Assertions.assertEquals(exp.data[i], act.data[i], 0);
            }
        }
    }


    public static void assertEquals(AbstractDenseDoubleNDArray<?> exp, AbstractDenseDoubleNDArray<?> act, double delta) {
        Assertions.assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.data, act.data, delta);
    }


    public static void assertEquals(CVector exp, CVector act, double delta) {
        Assertions.assertEquals(exp.shape, act.shape);
        assertEquals(exp.data, act.data, delta);
    }


    public static void assertEquals(CMatrix exp, CMatrix act, double delta) {
        Assertions.assertEquals(exp.shape, act.shape);
        assertEquals(exp.data, act.data, delta);
    }


    public static void assertEquals(CTensor exp, CTensor act, double delta) {
        Assertions.assertEquals(exp.shape, act.shape);
        assertEquals(exp.data, act.data, delta);
    }


    private static void assertEquals(Complex128[] exp, Complex128[] act, double delta) {
        for(int i = 0; i<exp.length; i++) {
            Assertions.assertEquals(exp[i].re, act[i].re,
                    delta, "real not equal at index " + i);
            Assertions.assertEquals(exp[i].im, act[i].im,
                    delta, "imaginary not equal at index " + i);
        }
    }
}
