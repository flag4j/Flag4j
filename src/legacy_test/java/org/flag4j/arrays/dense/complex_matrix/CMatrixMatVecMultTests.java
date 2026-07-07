package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixMatVecMultTests {
    Complex128[][] aEntries;
    Complex128[] expEntries;
    CMatrix A;
    CVector exp;


    @Test
    void matMultTestCase() {
        double[][] bEntries;
        Vector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{1.666},
                {-0.9345341},
                {0.0}};
        B = new Vector(ArrayUtils.flatten(bEntries));
        expEntries = new Complex128[]{new Complex128("163.51005868-15.462680014470001i"),
                new Complex128("1.666+694.4522897100001i"),
                new Complex128("12690.663369999998"),
                new Complex128("-3.377522800415388-8.877571549119406i")};
        exp = new CVector(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new Vector(ArrayUtils.flatten(bEntries));

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTestCase() {
        Complex128[][] bEntries;
        CVector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i")},
                {new Complex128("-0.0-0.9345341i")},
                {new Complex128("0.0")}};
        B = new CVector(ArrayUtils.flatten(bEntries));
        expEntries = new Complex128[]{
                new Complex128("215.01988001447+65.76525868i"),
                new Complex128("-692.78628971+1.0i"),
                new Complex128("12690.663369999998+7617.445i"),
                new Complex128("16.203765617290802-0.235930146825595i")};
        exp = new CVector(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")}};
        B = new CVector(ArrayUtils.flatten(bEntries));

        CVector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTestCase() {
        double[] bEntries;
        int[] rowIndices;
        CooVector B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-0.9345341};
        rowIndices = new int[]{1};
        B = new CooVector(3, bEntries, rowIndices);
        expEntries = new Complex128[]{
                new Complex128("-42.240941320000005+0.031119985530000005i"),
                new Complex128("0.0+694.4522897100001i"),
                new Complex128("0.0"),
                new Complex128("-8.611416161295983-14.11146491i")};
        exp = new CVector(expEntries);

        assertEquals(exp, A.mult(B));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-0.9345341};
        rowIndices = new int[]{1};
        B = new CooVector(14, bEntries, rowIndices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTestCase() {
        Complex128[] bEntries;
        int[] rowIndices, colIndices;
        CooCVector B;
        Shape bShape;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i")};
        rowIndices = new int[]{1};
        B = new CooCVector(3, bEntries, rowIndices);
        expEntries = new Complex128[]{
                new Complex128("-41.929586320000006+422.65111998553i"),
                new Complex128("6947.985+694.4522897100001i"),
                new Complex128("0.0"),
                new Complex128("-149.79641616129598+72.04562781472501i")};
        exp = new CVector(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1687, 2569070};
        B = new CooCVector(3450941, bEntries, rowIndices);

        CooCVector finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }

}
