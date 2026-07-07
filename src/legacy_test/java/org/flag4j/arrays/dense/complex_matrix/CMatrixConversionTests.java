package org.flag4j.arrays.dense.complex_matrix;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixConversionTests {

    Complex128[] expEntries;
    CTensor expTensor, tensor;
    CVector expVector, vector;
    Shape expTensorShape;

    Complex128[][] aEntries;
    double[][] expRealEntries;
    CMatrix A;
    Matrix expReal, real;

    @Test
    void toRealTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 7.233), new Complex128(-0.34436, 13)},
                {new Complex128(80.234, -9), new Complex128(Double.POSITIVE_INFINITY, 885.224)},
                {new Complex128(843.15), new Complex128(99.3434, 146)}};
        A = new CMatrix(aEntries);
        expRealEntries = new double[][]{
                {1, -0.34436},
                {80.234, Double.POSITIVE_INFINITY},
                {843.15, 99.3434}};
        expReal = new Matrix(expRealEntries);

        real = A.toReal();

        assertEquals(expReal, real);
    }


    @Test
    void toTensorTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 7.233), new Complex128(-0.34436, 13)},
                {new Complex128(80.234, -9), new Complex128(Double.POSITIVE_INFINITY, 885.224)},
                {new Complex128(843.15), new Complex128(99.3434, 146)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[]{
                new Complex128(1, 7.233), new Complex128(-0.34436, 13),
                new Complex128(80.234, -9), new Complex128(Double.POSITIVE_INFINITY, 885.224),
                new Complex128(843.15), new Complex128(99.3434, 146)};
        expTensorShape = new Shape(A.numRows, A.numCols);
        expTensor = new CTensor(expTensorShape, expEntries);

        tensor = A.toTensor();
        assertArrayEquals(A.data, tensor.data);
        assertEquals(A.shape, tensor.shape);
        assertEquals(2, tensor.getRank());
    }


    @Test
    void toVectorTestCase() {
        // -------------------- sub-case 1 --------------------
        aEntries = new Complex128[][]{
                {new Complex128(1, 7.233), new Complex128(-0.34436, 13)},
                {new Complex128(80.234, -9), new Complex128(Double.POSITIVE_INFINITY, 885.224)},
                {new Complex128(843.15), new Complex128(99.3434, 146)}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[]{
                new Complex128(1, 7.233), new Complex128(-0.34436, 13),
                new Complex128(80.234, -9), new Complex128(Double.POSITIVE_INFINITY, 885.224),
                new Complex128(843.15), new Complex128(99.3434, 146)};
        expVector = new CVector(expEntries);

        vector = A.toVector();
        assertArrayEquals(expVector.data, vector.data);
    }
}
