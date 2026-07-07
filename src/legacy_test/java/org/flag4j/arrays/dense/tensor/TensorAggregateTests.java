package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorAggregateTests {

    static Shape shape;
    static double[] entries;
    static Tensor A;
    static Double exp;
    static int[] expIndices;

    @BeforeAll
    static void setup() {
        entries = new double[]{1.234, 25.236, 6466.6, -0.0013, -8983.56, -0.01, 1.5, -99.3556, 12.56, 14.5, 11.6, -0.456};
        shape = new Shape(1, 2, 3, 1, 2);
        A = new Tensor(shape, entries);
    }


//    @Test
//    void minMaxTestCase() {
//        Tensor empty = new Tensor(new Shape(0), new double[]{});
//
//        // -------------------------- Minimum Tests --------------------------
//        exp = -8983.56;
//        assertEquals(exp, A.min());
//        exp = 0.0013;
//        assertEquals(exp, A.minAbs());
//        expIndices = shape.getIndices(4);
//        assertArrayEquals(expIndices, A.argmin());
//
//        assertArrayEquals(new int[]{}, empty.argmin());
//
//        // -------------------------- Maximum Tests --------------------------
//        exp = 6466.6;
//        assertEquals(exp, A.max());
//        exp = 8983.56;
//        assertEquals(exp, A.maxAbs());
//        expIndices = shape.getIndices(2);
//        assertArrayEquals(expIndices, A.argmax());
//
//        assertArrayEquals(new int[]{}, empty.argmax());
//    }


    @Test
    void sumTestCase() {
        // -------------------------- sub-case 1 --------------------------
        exp = 1.234+25.236+6466.6-0.0013-8983.56-0.01+1.5-99.3556+12.56+14.5+11.6-0.456;
        assertEquals(exp, A.sum());
    }
}
