package org.flag4j.arrays.dense.tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorConversionTests {

    static double[] aEntries;
    static Shape shape;
    static Tensor A;
    static CTensor exp;
    static Complex128[] expEntries;

    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234};
        shape = new Shape(1, 3, 2, 1, 2);
        A = new Tensor(shape, aEntries);
    }


    @Test
    void toComplexTestCase() {
        // --------------------- sub-case 1 ---------------------
        expEntries = new Complex128[]{
                new Complex128(1.23), new Complex128(2.556), new Complex128(-121.5),
                new Complex128(15.61), new Complex128(14.15), new Complex128(-99.23425),
                new Complex128(0.001345), new Complex128(2.677), new Complex128(8.14),
                new Complex128(-0.000194), new Complex128(1), new Complex128(234)};
        exp = new CTensor(shape, expEntries);

        assertEquals(exp, A.toComplex());
    }


    @Test
    void toMatrixTestCase() {
        Tensor B;

        double[] expEntries;
        Shape expShape;
        Matrix exp;

        // ----------------------- sub-case 1 -----------------------
        expEntries = aEntries.clone();
        expShape = new Shape(1, aEntries.length);
        exp = new Matrix(expShape, expEntries);

        assertEquals(exp, A.toMatrix());

        // ----------------------- sub-case 2 -----------------------
        expEntries = aEntries.clone();
        expShape = new Shape(4, 3);
        B = A.reshape(expShape);
        exp = new Matrix(expShape, expEntries);

        assertEquals(exp, B.toMatrix());
    }


    @Test
    void toVectorTestCase() {
        double[] expEntries;
        Vector exp;

        // ----------------------- sub-case 1 -----------------------
        expEntries = aEntries.clone();
        exp = new Vector(expEntries);

        assertEquals(exp, A.toVector());
    }
}
