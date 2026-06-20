package org.flag4j.arrays.dense.matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.util.exceptions.NDArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class MatrixAddSubEqTests {

    int[] rowIndices, colIndices;
    Shape sparseShape;

    double[][] aEntries;
    Matrix A, exp;

    @Test
    void addEqRealTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634, 14.466}, {Double.POSITIVE_INFINITY, 345.6, 8.346}};
        B = new Matrix(bEntries);
        exp = A.add(B);

        A.addEq(B);

        assertEquals(exp, A);

        // --------------------- sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634}, {Double.POSITIVE_INFINITY, 345.6}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(NDArrayShapeException.class, ()->A.addEq(finalB));
    }


    @Test
    void addEqDoubleTestCase() {
        double b;

        // --------------------- sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        b = 316.455;
        exp = A.add(b);

        A.addEq(b);

        assertEquals(exp, A);
    }


    @Test
    void subEqRealTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634, 14.466}, {Double.POSITIVE_INFINITY, 345.6, 8.346}};
        B = new Matrix(bEntries);
        exp = A.sub(B);

        A.subEq(B);

        assertEquals(exp, A);

        // --------------------- sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634}, {Double.POSITIVE_INFINITY, 345.6}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(NDArrayShapeException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqDoubleTestCase() {
        double b;

        // --------------------- sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        b = 316.455;
        exp = A.sub(b);

        A.subEq(b);

        assertEquals(exp, A);
    }
}
