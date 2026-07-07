package org.flag4j.linalg.ops.dense.real_complex;


import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseVectorOps;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps.add;
import static org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps.sub;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RealComplexDenseOpsTests {
    Complex128[] src1, expResult, act;
    double[] src2;
    double a;
    Complex128 aC;
    Shape shape1, shape2;

    @Test
    void addTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1.1345, 2, -3, 4.13};
        expResult = new Complex128[]{new Complex128(9+1.1345, -1), new Complex128(-0.99+2, 13.445),
                new Complex128(0.9133-3), new Complex128(0+4.13, 10.3)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[4];
        add(shape1, src1, shape2, src2, act);
        assertArrayEquals(expResult, act);

        // ---------- sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1, 2, 3, 4, 5};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[5];
        assertThrows(LinearAlgebraException.class, () -> add(shape1, src1, shape2, src2, act));

        // ---------- sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1, 2};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> add(shape1, src1, shape2, src2, act));
    }


    @Test
    void addScaleComplex() {
        // ---------- sub-case 1 -----------------
        src2 = new double[]{1.1345, 2, -3, 4.13};
        aC = new Complex128(10.34, -9.331);
        expResult = new Complex128[]{new Complex128(10.34+1.1345, -9.331), new Complex128(10.34+2, -9.331),
                new Complex128(10.34-3, -9.331), new Complex128(10.34+4.13, -9.331)};

        Complex128[] act = new Complex128[4];
        RealFieldDenseVectorOps.add(src2, aC, act);
        assertArrayEquals(expResult, act);
    }


    @Test
    void subTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1.1345, 2, -3, 4.13};
        expResult = new Complex128[]{new Complex128(9-1.1345, -1), new Complex128(-0.99-2, 13.445),
                new Complex128(0.9133+3), new Complex128(0-4.13, 10.3)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[4];
        sub(shape1, src1, shape2, src2, act);
        assertArrayEquals(expResult, act);

        // ---------- sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1, 2, 3, 4, 5};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[4];
        assertThrows(LinearAlgebraException.class, () -> sub(shape1, src1, shape2, src2, act));

        // ---------- sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1, 2};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[4];
        assertThrows(LinearAlgebraException.class, () -> sub(shape1, src1, shape2, src2, act));
    }

    @Test
    void subReverseTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1.1345, 2, -3, 4.13};
        expResult = new Complex128[]{new Complex128(1.1345-9, 1), new Complex128(2+0.99, -13.445),
                new Complex128(-3-0.9133), new Complex128(4.13, -10.3)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[src1.length];
        sub(shape2, src2, shape1, src1, act);
        assertArrayEquals(expResult, act);

        // ---------- sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1, 2, 3, 4, 5};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[4];
        assertThrows(LinearAlgebraException.class, () -> sub(shape2, src2, shape1, src1, act));

        // ---------- sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new double[]{1, 2};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        act = new Complex128[4];
        assertThrows(LinearAlgebraException.class, () -> sub(shape2, src2, shape1, src1, act));
    }


    @Test
    void subScaleComplex() {
        // ---------- sub-case 1 -----------------
        src2 = new double[]{1.1345, 2, -3, 4.13};
        aC = new Complex128(10.34, -9.331);
        expResult = new Complex128[]{new Complex128(1.1345-10.34, 9.331), new Complex128(2-10.34, 9.331),
                new Complex128(-3-10.34, 9.331), new Complex128(4.13-10.34, 9.331)};

        act = new Complex128[4];
        sub(src2, aC, act);
        assertArrayEquals(expResult, act);
    }


    @Test
    void scaleMultComplex() {
        // ---------- sub-case 1 -----------------
        src2 = new double[]{1.1345, 2, -3, 4.13};
        aC = new Complex128(10.34, -9.331);
        expResult = new Complex128[]{aC.mult(1.1345), aC.mult(2),
                aC.mult(-3), aC.mult(4.13)};
//        assertArrayEquals(expResult, FieldOps.scalMult(src2, aC));
    }
}
