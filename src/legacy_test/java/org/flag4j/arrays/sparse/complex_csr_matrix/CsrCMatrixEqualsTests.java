package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.util.ArrayBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class CsrCMatrixEqualsTests {
    CsrCMatrix A;
    CsrCMatrix B;

    Complex128[][] aEntries;
    Complex128[][] bEntries;

    Complex128[] aNnz;
    Complex128[] bNnz;
    int[][] aIndices;
    int[][] bIndices;
    Shape aShape;
    Shape bShape;

    @Test
    void realCsrEqualsTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[150][235];
        ArrayBuilder.fill(aEntries, Complex128.ZERO);
        bEntries = new Complex128[150][235];
        ArrayBuilder.fill(bEntries, Complex128.ZERO);
        aEntries[0][1] = new Complex128(134.4, -0.0234);
        aEntries[15][234] = new Complex128(-0.00024, 1.45);
        aEntries[49][1] = new Complex128(234.25000024, 234.5);
        aEntries[59][0] = new Complex128(0, 23.342);
        aEntries[59][2] = new Complex128(-0.0000005, 235.03);
        aEntries[149][1] = new Complex128(15);

        bEntries[0][1] = new Complex128(134.4, -0.0234);
        bEntries[15][234] = new Complex128(-0.00024, 1.45);
        bEntries[49][1] = new Complex128(234.25000024, 234.5);
        bEntries[59][0] = new Complex128(0, 23.342);
        bEntries[59][2] = new Complex128(-0.0000005, 235.03);
        bEntries[149][1] = new Complex128(15);

        A = new CMatrix(aEntries).toCsr();
        B = new CMatrix(bEntries).toCsr();

        assertEquals(A, B);

        // ---------------------- sub-case 2 ----------------------
        aNnz = new Complex128[]{new Complex128(1.34507518, -352), new Complex128(0.235, 2), new Complex128(0, 72735),
                new Complex128(94.1, -1), new Complex128(1, 3), new Complex128(-1, 0.002345), new Complex128(15)};
        bNnz = new Complex128[]{new Complex128(1.34507518, -352), new Complex128(0.235, 2), new Complex128(0, 72735),
                new Complex128(94.1, -1), new Complex128(1, 3), new Complex128(-1, 0.002345), new Complex128(15)};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertEquals(A, B);

        // ---------------------- sub-case 3 ----------------------
        aNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, new Complex128(345.1, 2.5), new Complex128(9.4, -1),
                new Complex128(235.1, 94.2), new Complex128(3.12, 4), Complex128.ZERO,
                new Complex128(0, 1), new Complex128(2,9733)};
        bNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(345.1, 2.5),
                new Complex128(9.4, -1),
                Complex128.ZERO, new Complex128(235.1, 94.2), new Complex128(3.12, 4),
                new Complex128(0, 1), new Complex128(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        CooCMatrix ta = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]);
        CooCMatrix tb = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]);

        assertEquals(A, B);

        // ---------------------- sub-case 4 ----------------------
        aNnz = new Complex128[]{new Complex128(234.5, -0.2), new Complex128(4.23, 9), new Complex128(345.1, 2.5), new Complex128(9.4, -1),
                new Complex128(235.1, 94.2), new Complex128(3.12, 4), Complex128.ZERO,
                new Complex128(0, 1), new Complex128(2,9733)};
        bNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(345.1, 2.5),
                new Complex128(9.4, -1),
                Complex128.ZERO, new Complex128(235.1, 94.2), new Complex128(3.12, 4),
                new Complex128(0, 1), new Complex128(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 5 ----------------------
        aNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, new Complex128(345.1, 2.5), new Complex128(9.4, -1),
                new Complex128(235.1, 94.2), new Complex128(3.12, 4), Complex128.ZERO,
                new Complex128(0, 1), new Complex128(2,9733)};
        bNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(345.1, 2.5),
                new Complex128(9.4, -1),
                Complex128.ZERO, new Complex128(235.1, 94.2), new Complex128(3.12, 4),
                new Complex128(0, 1), new Complex128(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 3, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 6 ----------------------
        aNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, new Complex128(345.1, 2.5), new Complex128(9.4, -1),
                new Complex128(235.1, 94.2), new Complex128(3.12, 4), Complex128.ZERO,
                new Complex128(0, 1), new Complex128(2,9733)};
        bNnz = new Complex128[]{new Complex128(234.5, -0.2), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(345.1, 2.5),
                new Complex128(9.4, -1),
                Complex128.ZERO, new Complex128(235.1, 94.2), new Complex128(3.12, 4),
                new Complex128(0, 1), new Complex128(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 22, 5008, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 7 ----------------------
        aNnz = new Complex128[]{new Complex128(1.34507518, -352), new Complex128(0.235, 2), new Complex128(0, 72735),
                new Complex128(94.1, -1), new Complex128(1, 3), new Complex128(-1, 0.002345), new Complex128(15)};
        bNnz = new Complex128[]{new Complex128(1.34507518, -352), new Complex128(0.235, 2), new Complex128(0, 72735),
                new Complex128(94.1, -1), new Complex128(1, 3), new Complex128(-1, 0.002345), new Complex128(15)};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 60002);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);
    }
}
