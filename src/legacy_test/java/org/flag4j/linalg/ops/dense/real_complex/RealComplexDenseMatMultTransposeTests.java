/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package org.flag4j.linalg.ops.dense.real_complex;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseMatMultTranspose.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealComplexDenseMatMultTransposeTests {
    double[][] aEntries;
    Complex128[] act;

    Matrix A;
    CMatrix expC;

    @Test
    void matMultTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();
        expC = A.mult(B.T());

        act = new Complex128[A.numRows*B.numRows];
        multTranspose(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeBlocked(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeBlockedConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();
        expC = A.mult(B.T());


        act = new Complex128[A.numRows*B.numRows];
        multTranspose(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeBlocked(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeBlockedConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(expC.data, act);

        // ---------------------- sub-case 3 ----------------------
        aEntries = new double[][]{
                {1.1234, 99.234},
                {-932.45, 551.35},
                {0.000123, -0.92342}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i"), new Complex128("0.0")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();
        expC = B.mult(A.T());

        act = new Complex128[A.numRows*B.numRows];
        multTranspose(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeBlocked(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeConcurrent(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);

        act = new Complex128[A.numRows*B.numRows];
        multTransposeBlockedConcurrent(B.data, B.shape, A.data, A.shape, act);
        assertArrayEquals(expC.data, act);
    }
}
