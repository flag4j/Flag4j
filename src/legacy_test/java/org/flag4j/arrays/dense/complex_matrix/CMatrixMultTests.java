package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixMultTests {
    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void matMultTestCase() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new Matrix(bEntries);
        expEntries = new Complex128[][]{{new Complex128("163.51005868-15.462680014470001i"), new Complex128("5408.426908-109.8881922i")},
                {new Complex128("1.666+694.4522897100001i"), new Complex128("11.49931-65566.68726i")},
                {new Complex128("12690.663369999998"), new Complex128("87600.6175")},
                {new Complex128("-3.377522800415388-8.877571549119406i"), new Complex128("849.1747509679817+1368.4617155162825i")}};
        exp = new CMatrix(expEntries);

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
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("215.01988001447+65.76525868i"), new Complex128("5323.5830080000005-776.3366921999999i")},
                {new Complex128("-692.78628971+1.0i"), new Complex128("7937.8893100000005-68516.24526000001i")},
                {new Complex128("12690.663369999998+7617.445i"), new Complex128("87600.6175-69493.95073499999i")},
                {new Complex128("16.203765617290802-0.235930146825595i"), new Complex128("877.8355007466813+998.8809657375828i")}};
        exp = new CMatrix(expEntries);

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
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTestCase() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        CooMatrix B;
        Shape bShape;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{{new Complex128("-42.240941320000005+0.031119985530000005i"), new Complex128("63.018")},
                {new Complex128("0.0+694.4522897100001i"), new Complex128("-402.615-1085.31i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("-8.611416161295983-14.11146491i"), new Complex128("-46.68")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTestCase() {
        Complex128[] bEntries;
        int[] rowIndices, colIndices;
        CooCMatrix B;
        Shape bShape;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{{new Complex128("-41.929586320000006+422.65111998553i"), new Complex128("63.018-10.8i")},
                {new Complex128("6947.985+694.4522897100001i"), new Complex128("-588.615-1016.31i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("-149.79641616129598+72.04562781472501i"), new Complex128("-46.68+8.0i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO},
                {new Complex128(Math.PI, Math.PI), new Complex128(9.2146623235, 15.1), new Complex128(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128("-0.9345341+9.35i"), new Complex128("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooCMatrix(bShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void powTestCase() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(120661.30796950031, -3.323620715921176E7), new Complex128(-2.2728032650843892E7, -4235221.228035901), new Complex128(-3050628.7322100005, 644215.430715)},
                {new Complex128(-5.659757945347501E8, 1.1014787929019998E8), new Complex128(-1.18967017985705E7, 3.7825850277813834E8), new Complex128(1.76307919446E7, 4.752063143985001E7)},
                {new Complex128(4.29206180168535E8, -1.7498286570418503E7), new Complex128(4.23312470039206E7, -2.5908905305703476E8), new Complex128(-6822160.1279205, -3.2394488588211756E7)}
        };
        exp = new CMatrix(expEntries);
        assertEquals(exp, A.pow(3));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{{new Complex128(1), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128(1), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, new Complex128(1)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.pow(0));


        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        A = new CMatrix(aEntries);
        expEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.pow(1));

        // ---------------------- sub-case 4 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)}};
        A = new CMatrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()->A.pow(2));

        // ---------------------- sub-case 5 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(123.5, -9.3), new Complex128(45.2, -0.0333), new Complex128(5.4)},
                {new Complex128(1), new Complex128(0, -743.1), new Complex128(-34.5, -93.)},
                {new Complex128(7617.445), new Complex128(0), Complex128.ZERO}};

        assertThrows(IllegalArgumentException.class, ()->A.pow(-1));
    }
}
