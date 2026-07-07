package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorElemDivTests {

    CooVector a;
    int size;

    @Test
    void denseDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 4};
        size = 6;
        a = new CooVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices;
        Vector b;
        double[] expValues;
        CooVector exp;

        // -------------------- sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        expValues = new double[]{1.34/1.223, 51.6/3.4, -0.00245/14.5};
        expIndices = new int[]{0, 2, 4};
        exp = new CooVector(size, expValues, expIndices);
        assertEquals(exp, a.div(b));

        // -------------------- sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 123, 25.2, 155};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.div(finalB));
    }


    @Test
    void denseComplexDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 4};
        size = 6;
        a = new CooVector(size, aValues, aIndices);

        Complex128[] bValues, expValues;
        int[] expIndices;
        CVector b;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        bValues = new Complex128[]{new Complex128(24.3, -0.013), new Complex128(0, 13.6),
                new Complex128(2.4), new Complex128(-994.1 ,1.45), new Complex128(1495, 13.4),
                new Complex128(9924.515, 51.5)};
        b = new CVector(bValues);

        expValues = new Complex128[]{
                new Complex128(1.34).div(new Complex128(24.3, -0.013)),
                new Complex128(51.6).div(new Complex128(2.4)),
                new Complex128(-0.00245).div(new Complex128(1495, 13.4))
        };
        expIndices = new int[]{0, 2, 4};
        exp = new CooCVector(size, expValues, expIndices);
        assertEquals(exp, a.div(b));

        // -------------------- sub-case 2 --------------------
        bValues = new Complex128[]{new Complex128(24.3, -0.013), new Complex128(0, 13.6)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.div(finalB));
    }


    @Test
    void doubleScalarDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVector(size, aValues, aIndices);

        double b;

        double[] expValues;
        int[] expIndices;
        CooVector exp;

        // -------------------- sub-case 1 --------------------
        b = 24.56;

        expValues = new double[]{1.34*(1.0/b), 51.6*(1.0/b), -0.00245*(1.0/b), 99.2456*(1.0/b), -1005.6*(1.0/b)};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new CooVector(151, expValues, expIndices);
        assertEquals(exp, a.div(b));
    }


    @Test
    void complexScalarDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVector(size, aValues, aIndices);

        Complex128 b;

        Complex128[] expValues;
        int[] expIndices;
        CooCVector exp;

        // -------------------- sub-case 1 --------------------
        b = new Complex128(234.6677, -9.35);

        expValues = new Complex128[]{
                new Complex128(1.34).div(b), new Complex128(51.6).div(b),
                new Complex128(-0.00245).div(b), new Complex128(99.2456).div(b),
                new Complex128(-1005.6).div(b)};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new CooCVector(151, expValues, expIndices);

        assertEquals(exp, a.div(b));
    }
}
