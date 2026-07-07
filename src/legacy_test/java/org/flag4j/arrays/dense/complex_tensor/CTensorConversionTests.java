package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CTensorConversionTests {
    static Complex128[] aEntries;
    static Shape shape;
    static CTensor A;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(235.61, 1.45), new Complex128(0, -0.00024),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(1.345, -781.2),
                new Complex128(0.014, -2.45),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(51.0, 24.56),  new Complex128(6.1, -0.03),  new Complex128(-0.00014, 1.34),};
        shape = new Shape(1, 3, 2, 1, 2);
        A = new CTensor(shape, aEntries);
    }


    @Test
    void toRealTestCase() {
        double[] expEntries;
        Tensor exp;

        // --------------------- sub-case 1 ---------------------
        expEntries = new double[]{1.4415, 235.61, 0, 1.0, -85.1, 1.345, 0.014, -140.0, 0, 51.0, 6.1, -0.00014};
        exp = new Tensor(shape, expEntries);

        assertEquals(exp, A.toReal());
    }


    @Test
    void toMatrixTestCase() {
        CTensor B;

        Complex128[] expEntries;
        Shape expShape;
        CMatrix exp;

        // ----------------------- sub-case 1 -----------------------
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        expShape = new Shape(1, aEntries.length);
        exp = new CMatrix(expShape, expEntries);

        assertEquals(exp, A.toMatrix());

        // ----------------------- sub-case 2 -----------------------
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        expShape = new Shape(4, 3);
        B = A.reshape(expShape);
        exp = new CMatrix(expShape, expEntries);

        assertEquals(exp, B.toMatrix());
    }


    @Test
    void toVectorTestCase() {
        Complex128[] expEntries;
        CVector exp;

        // ----------------------- sub-case 1 -----------------------
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        exp = new CVector(expEntries);

        assertEquals(exp, A.toVector());
    }
}
