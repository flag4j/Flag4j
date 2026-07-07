package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixStackTests {
    Shape sparseShape;
    int[] rowIndices, colIndices;
    Complex128[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void complexMatrixTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ----------------------- sub-case 1 -----------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128(234.5, -87.234)}, {new Complex128(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3), new Complex128(234.5, -87.234)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3), new Complex128(-1867.4, 77.51)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- sub-case 2 -----------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128(43.566920234, 234.5)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- sub-case 3 -----------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128(234.5, -87.234), new Complex128(-1867.4, 77.51), new Complex128(9, -987.43)}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)},
                {new Complex128(234.5, -87.234), new Complex128(-1867.4, 77.51), new Complex128(9, -987.43)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- sub-case 4 -----------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128(234.5, -87.234), new Complex128(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);

        CMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- sub-case 5 -----------------------
        aEntries = new Complex128[][]{
                {new Complex128(9.234, -0.864), new Complex128(58.1, 3), new Complex128(-984, -72.3)},
                {new Complex128(1), Complex128.ZERO, new Complex128(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128(234.5, -87.234), new Complex128(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);

        CMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }
}
