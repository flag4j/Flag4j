/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class RealCsrEqualsTests {
    CsrMatrix A;
    CsrMatrix B;

    double[][] aEntries;
    double[][] bEntries;

    double[] aNnz;
    double[] bNnz;
    int[][] aIndices;
    int[][] bIndices;
    Shape aShape;
    Shape bShape;

    @Test
    void realCsrEqualsTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[150][235];
        bEntries = new double[150][235];

        aEntries[0][1] = 134.4;
        aEntries[15][234] = -0.00024;
        aEntries[49][1] = 234.25000024;
        aEntries[59][0] = 23.342;
        aEntries[59][2] = -0.0000005;
        aEntries[149][1] = 15;

        bEntries[0][1] = 134.4;
        bEntries[15][234] = -0.00024;
        bEntries[49][1] = 234.25000024;
        bEntries[59][0] = 23.342;
        bEntries[59][2] = -0.0000005;
        bEntries[149][1] = 15;

        A = new Matrix(aEntries).toCsr();
        B = new Matrix(bEntries).toCsr();

        assertEquals(A, B);

        // ---------------------- sub-case 2 ----------------------
        aNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        bNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertEquals(A, B);

        // ---------------------- sub-case 3 ----------------------
        aNnz = new double[]{1, 0, 1.334, -0.0014, 23592.1, -992934.1, 0, 235.235, 5};
        bNnz = new double[]{1, 0, 0, 0, 1.334, -0.0014, 0, 23592.1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertEquals(A, B);

        // ---------------------- sub-case 4 ----------------------
        aNnz = new double[]{1, 0, 1.334, -0.0014, 23592.1, -992934.1, 0, 235.235, 5};
        bNnz = new double[]{1, -1.4, 0, 0, 1.334, -0.0014, 0, 23592.1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 5 ----------------------
        aNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        bNnz = new double[]{1, 2, -0.0014, 23592.1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 6 ----------------------
        aNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        bNnz = new double[]{1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5},
                {4, 5, 14, 5002}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 7 ----------------------
        aNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        bNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 1, 5, 13, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- sub-case 7 ----------------------
        aNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        bNnz = new double[]{1, 1.334, -0.0014, 23592.1, -992934.1, 235.235, 5};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 305, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);
    }
}
