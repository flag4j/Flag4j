package org.flag4j.arrays.dense.vector;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.exceptions.NDArrayShapeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorStackJoinTests {

    double[] aEntries = {1.5, 6.2546, -0.24};
    Vector a = new Vector(aEntries);
    int[] indices;
    int sparseSize;

    @Test
    void realDenseJoinTestCase() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new Vector(bEntries);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 1.5};
        exp = new Vector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void realDenseStackTestCase() {
        double[] bEntries;
        Vector b;
        double[][] expEntries;
        Matrix exp;

        // ---------------------- sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- sub-case 2 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(NDArrayShapeException.class, ()->a.stack(finalB));

        // ---------------------- sub-case 3 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- sub-case 4 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(NDArrayShapeException.class, ()->a.stack(finalB2, 0));

        // ---------------------- sub-case 5 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 0.9345}, {6.2546, 1.5}, {-0.24, -9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- sub-case 6 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new Vector(bEntries);

        Vector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }
}
