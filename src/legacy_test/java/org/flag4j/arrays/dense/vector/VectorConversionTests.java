package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorConversionTests {

    double[] aEntries;
    Vector a;

    double[][] matExpEntries;
    Matrix matExp;

    double[] tensorExpEntries;
    Shape tensorExpShape;
    Tensor tensorExp;

    @Test
    void toMatrixTestCase() {
        // --------------------- sub-case 1  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new Vector(aEntries);
        matExpEntries = new double[][]{{1.34}, {-8.244}, {1.234}, {90031.3}};
        matExp = new Matrix(matExpEntries);

        assertEquals(matExp, a.toMatrix());

        // --------------------- sub-case 2  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new Vector(aEntries);
        matExpEntries = new double[][]{{1.34}, {-8.244}, {1.234}, {90031.3}};
        matExp = new Matrix(matExpEntries);

        assertEquals(matExp, a.toMatrix(true));

        // --------------------- sub-case 2  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new Vector(aEntries);
        matExpEntries = new double[][]{{1.34, -8.244, 1.234, 90031.3}};
        matExp = new Matrix(matExpEntries);

        assertEquals(matExp, a.toMatrix(false));
    }


    @Test
    void toTensorTestCase() {
        // --------------------- sub-case 1  ---------------------
        aEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        a = new Vector(aEntries);
        tensorExpEntries = new double[]{1.34, -8.244, 1.234, 90031.3};
        tensorExpShape = new Shape(aEntries.length);
        tensorExp = new Tensor(tensorExpShape, tensorExpEntries);

        assertEquals(tensorExp, a.toTensor());

        // --------------------- sub-case 2  ---------------------
        aEntries = new double[]{-34534, 6.2234};
        a = new Vector(aEntries);
        tensorExpEntries = new double[]{-34534, 6.2234};
        tensorExpShape = new Shape(aEntries.length);
        tensorExp = new Tensor(tensorExpShape, tensorExpEntries);

        assertEquals(tensorExp, a.toTensor());
    }
}
