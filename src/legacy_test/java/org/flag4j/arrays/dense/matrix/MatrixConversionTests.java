package org.flag4j.arrays.dense.matrix;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixConversionTests {

    double[][] aEntries;
    Matrix A;


    @Test
    void toComplexTestCase() {
        CMatrix exp;

        // --------------------- sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new CMatrix(aEntries);

        assertEquals(exp, A.toComplex());
    }


    @Test
    void toVectorTestCase() {
        Vector exp;

        // --------------------- sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new Vector(ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toVector());
    }


    @Test
    void toTensorTestCase() {
        Tensor exp;

        // --------------------- sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new Tensor(A.shape, ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toTensor());
    }
}
