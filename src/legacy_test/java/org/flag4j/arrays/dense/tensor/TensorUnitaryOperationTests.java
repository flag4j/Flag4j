package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorUnitaryOperationTests {

    static double[] aEntries, expEntries;
    static Shape aShape, expShape;
    static Tensor A, exp;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{
                0.000123, 5.23523, -834513.36, 235.6,
                934, 13.5, -0.0, 0.1,
                345, 8345.6, 1.00015, Double.POSITIVE_INFINITY};
        aShape = new Shape(3, 2, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void sqrtTestCase() {
        // --------------------- sub-case 1 ---------------------
        expEntries = new double[]{
                Math.sqrt(0.000123), Math.sqrt(5.23523), Math.sqrt(-834513.36), Math.sqrt(235.6),
                Math.sqrt(934), Math.sqrt(13.5), Math.sqrt(-0.0), Math.sqrt(0.1),
                Math.sqrt(345), Math.sqrt(8345.6), Math.sqrt(1.00015), Math.sqrt(Double.POSITIVE_INFINITY)};
        expShape = aShape;
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.sqrt());
    }


    @Test
    void absTestCase() {
        // --------------------- sub-case 1 ---------------------
        expEntries = new double[]{
                Math.abs(0.000123), Math.abs(5.23523), Math.abs(-834513.36), Math.abs(235.6),
                Math.abs(934), Math.abs(13.5), Math.abs(-0.0), Math.abs(0.1),
                Math.abs(345), Math.abs(8345.6), Math.abs(1.00015), Math.abs(Double.POSITIVE_INFINITY)};
        expShape = aShape;
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.abs());
    }


    @Test
    void recipTestCase() {
        // --------------------- sub-case 1 ---------------------
        expEntries = new double[]{
                1/(0.000123), 1/(5.23523), 1/(-834513.36), 1/(235.6),
                1/(934.0), 1/(13.5), 1/(-0.0), 1/(0.1),
                1/(345.0), 1/(8345.6), 1/(1.00015), 1/(Double.POSITIVE_INFINITY)};
        expShape = aShape;
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.recip());
    }
}
