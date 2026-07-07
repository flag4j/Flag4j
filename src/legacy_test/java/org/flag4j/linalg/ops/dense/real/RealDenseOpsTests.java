package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense.real.RealDenseOps.*;
import static org.junit.jupiter.api.Assertions.*;

class RealDenseOpsTests {
    double[] src1, src2;
    double[] expResult;
    double[] act;
    double expResultD;
    Shape shape1;
    Shape shape2;
    double a;

    @Test
    void addTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4};
        expResult = new double[]{1+9.1233, 0.98332+22, 134.556+0.00009234,
                -9.13+11.234, -100.234-88.1, 0.0000000004+13.4};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        act = new double[6];

        add(src1, src2, act);
        assertArrayEquals(expResult, act);

        // ---------- sub-case 2 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4, 1, 1};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertThrows(IllegalArgumentException.class, () -> add(src1, src2, null));

        // ---------- sub-case 3 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertThrows(IllegalArgumentException.class, () -> add(src1, src2, null));
    }


    @Test
    void addScalarTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1+a, 0.98332+a, 134.556+a,
                -9.13+a, -100.234+a, 0.0000000004+a};

        assertArrayEquals(expResult, add(src1, a, null));
    }


    @Test
    void subTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4};
        expResult = new double[]{1-9.1233, 0.98332-22, 134.556-0.00009234,
                -9.13-11.234, -100.234+88.1, 0.0000000004-13.4};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, sub(src1, src2, null));

        // ---------- sub-case 2 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4, 1, 1};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> sub(src1, src2, act));

        // ---------- sub-case 3 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(IllegalArgumentException.class, () -> sub(src1, src2, null));
    }


    @Test
    void subScalarTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1-a, 0.98332-a, 134.556-a,
                -9.13-a, -100.234-a, 0.0000000004-a};

        assertArrayEquals(expResult, sub(src1, a, null));
    }


    @Test
    void prodTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.8000000004};
        expResultD = 1*0.98332*134.556*-9.13*-100.234*0.8000000004;
        assertEquals(expResultD, prod(src1));

        // ---------- sub-case 2 -----------------
        src1 = new double[]{};
        expResultD = 0;
        assertEquals(expResultD, prod(src1));
    }


    @Test
    void scaleMultTestCase() {
        // ---------- sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1*a, 0.98332*a, 134.556*a,
                -9.13*a, -100.234*a, 0.0000000004*a};

        assertArrayEquals(expResult, RealOps.scalMult(src1, a, null));
    }
}
