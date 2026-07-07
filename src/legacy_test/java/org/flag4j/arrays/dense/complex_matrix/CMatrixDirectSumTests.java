package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.DirectSum;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixDirectSumTests {

    Shape sparseShape;
    int[] rowIndices, colIndices;

    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realDirectSumTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {1, -3.23},
                {324, 5.234},
                {-74.13, 44.5}};
        B = new Matrix(bEntries);
        expEntries = new Complex128[][]{{new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("1.0"), new Complex128("-3.23")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("324.0"), new Complex128("5.234")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-74.13"), new Complex128("44.5")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void realSparseDirectSumTestCase() {
        double[] bEntries;
        CooMatrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[]{2.456, -7.41};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{{new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("2.456")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-7.41"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void complexDirectSumTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.0+9.435i"), new Complex128("-3.23-8.234i")},
                {new Complex128("0.0+324.0i"), new Complex128("5.234-8.4i")},
                {new Complex128("-74.13+475.145i"), new Complex128("44.5+8.345i")}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{{new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("1.0+9.435i"), new Complex128("-3.23-8.234i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0+324.0i"), new Complex128("5.234-8.4i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-74.13+475.145i"), new Complex128("44.5+8.345i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void complexSparseDirectSumTestCase() {
        Complex128[] bEntries;
        CooCMatrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(234.6567, -6344.256), new Complex128(Double.NEGATIVE_INFINITY, 234.56)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{{new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128(234.6567, -6344.256)},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128(Double.NEGATIVE_INFINITY, 234.56), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void realInvDirectSumTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {1, -3.23},
                {324, 5.234},
                {-74.13, 44.5}};
        B = new Matrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("1.0"), new Complex128("-3.23")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("324.0"), new Complex128("5.234")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-74.13"), new Complex128("44.5")},
                {new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void realInvSparseDirectSumTestCase() {
        double[] bEntries;
        CooMatrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[]{2.456, -7.41};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("2.456")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-7.41"), new Complex128("0.0")},
                {new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void complexInvDirectSumTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128("1.0+9.435i"), new Complex128("-3.23-8.234i")},
                {new Complex128("0.0+324.0i"), new Complex128("5.234-8.4i")},
                {new Complex128("-74.13+475.145i"), new Complex128("44.5+8.345i")}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("1.0+9.435i"), new Complex128("-3.23-8.234i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0+324.0i"), new Complex128("5.234-8.4i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("-74.13+475.145i"), new Complex128("44.5+8.345i")},
                {new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void complexInvSparseDirectSumTestCase() {
        Complex128[] bEntries;
        CooCMatrix B;

        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(234.6567, -6344.256), new Complex128(Double.NEGATIVE_INFINITY, 234.56)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128(234.6567, -6344.256)},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128(Double.NEGATIVE_INFINITY, 234.56), new Complex128("0.0")},
                {new Complex128("9.234-0.864i"), new Complex128("58.1+3.0i"), new Complex128("-984.0-72.3i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0"), new Complex128("0.0"), new Complex128("0.0+87.3i"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }
}
