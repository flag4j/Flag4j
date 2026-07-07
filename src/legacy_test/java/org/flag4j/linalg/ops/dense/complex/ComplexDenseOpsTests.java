package org.flag4j.linalg.ops.dense.complex;

import org.flag4j.arrays.Shape;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.common.semiring_ops.AggregateSemiring;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringOps;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ComplexDenseOpsTests {

    Complex128[] src1, src2;
    Complex128[] expResult;
    Complex128 expResultC;
    Shape shape1, shape2;
    double a;
    Complex128 aC;


    @Test
    void addTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(-9234.23), new Complex128(109.2234, 1.435),
                new Complex128(0, -1943.134), new Complex128(-9234.1, -3245)};
        expResult = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435),
                new Complex128(0.9133, -1943.134), new Complex128(-9234.1, 10.3-3245)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, DenseSemiringOps.add(src1, shape1, src2, shape2, null));

        // ---------- sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435),
                new Complex128(0.9133, -1943.134), new Complex128(-9234.1, 10.3-3245), new Complex128(0, 1)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> DenseSemiringOps.add(src1, shape1, src2, shape2, null));

        // ---------- sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> DenseSemiringOps.add(src1, shape1, src2, shape2, null));
    }


    @Test
    void addDoubleTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        a = 933.1334;
        expResult = new Complex128[]{new Complex128(9+a, -1), new Complex128(-0.99+a, 13.445),
                new Complex128(0.9133+a), new Complex128(0+a, 10.3)};

        Complex128[] act = new Complex128[4];
        RealFieldDenseOps.add(src1, a, act);
        assertArrayEquals(expResult, act);
    }


    @Test
    void addComplex128TestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        aC = new Complex128(10.34, -1.334);
        expResult = new Complex128[]{new Complex128(9, -1).add(aC), new Complex128(-0.99, 13.445).add(aC),
                new Complex128(0.9133).add(aC), new Complex128(0, 10.3).add(aC)};
        Complex128[] act = new Complex128[4];
        SemiringOps.add(src1, aC, act);
        assertArrayEquals(expResult, act);
    }


    @Test
    void subTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(-9234.23), new Complex128(109.2234, 1.435),
                new Complex128(0, -1943.134), new Complex128(-9234.1, -3245)};

        expResult = new Complex128[]{new Complex128(9+9234.23, -1), new Complex128(-0.99-109.2234, 13.445-1.435),
                new Complex128(0.9133, 1943.134), new Complex128(9234.1, 10.3+3245)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
//        assertArrayEquals(expResult, DenseFieldOps.sub(src1, shape1, src2, shape2));

        // ---------- sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435),
                new Complex128(0.9133, -1943.134), new Complex128(-9234.1, 10.3-3245), new Complex128(0, 1)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
//        assertThrows(LinearAlgebraException.class, () -> DenseFieldOps.sub(src1, shape1, src2, shape2));

        // ---------- sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
//        assertThrows(LinearAlgebraException.class, () -> DenseFieldOps.sub(src1, shape1, src2, shape2));
    }


    @Test
    void subDoubleTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        a = 933.1334;
        expResult = new Complex128[]{new Complex128(9-a, -1), new Complex128(-0.99-a, 13.445),
                new Complex128(0.9133-a), new Complex128(0-a, 10.3)};

        Complex128[] act = new Complex128[4];
        RealFieldDenseOps.sub(src1, a, act);
        assertArrayEquals(expResult, act);
    }


    @Test
    void subComplex128TestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        aC = new Complex128(10.34, -1.334);
        expResult = new Complex128[]{new Complex128(9, -1).sub(aC), new Complex128(-0.99, 13.445).sub(aC),
                new Complex128(0.9133).sub(aC), new Complex128(0, 10.3).sub(aC)};
        Complex128[] act = new Complex128[4];
        RingOps.sub(src1, aC, act);
        assertArrayEquals(expResult, act);
    }


    @Test
    void sumTestCase() {
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        expResultC = new Complex128(9-0.99+0.9133, -1+13.445+10.3);
        assertEquals(expResultC, AggregateSemiring.sum(src1));
    }


    @Test
    void prodTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};

        expResultC = new Complex128(-1147.60574505, 42.660699650000005);
        assertEquals(expResultC, AggregateSemiring.prod(src1));

        // ---------- sub-case 2 -----------------
        src1 = new Complex128[]{};
        assertNull(AggregateSemiring.prod(src1));
    }


    @Test
    void scalMultDoubleTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        a = 933.1334;
        expResult = new Complex128[]{new Complex128(9*a, -1*a), new Complex128(-0.99*a, 13.445*a),
                new Complex128(0.9133*a), new Complex128(0*a, 10.3*a)};
//        assertArrayEquals(expResult, DenseFieldOps.scalMult(src1, a));
    }


    @Test
    void scalMultComplex128TestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        aC = new Complex128(10.34, -1.334);
        expResult = new Complex128[]{new Complex128(9, -1).mult(aC), new Complex128(-0.99, 13.445).mult(aC),
                new Complex128(0.9133).mult(aC), new Complex128(0, 10.3).mult(aC)};
        Complex128[] act = new Complex128[4];
        SemiringOps.scalMult(src1, aC, act);
        assertArrayEquals(expResult, act);
    }
}
