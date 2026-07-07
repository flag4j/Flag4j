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

package org.flag4j.linalg.ops.dense.complex;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMultTranspose;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexDenseMatMultTransposeTests {
    Complex128[][] entriesA, entriesB;
    CMatrix A, B;
    Complex128[] exp, act;

    @Test
    void squareTestCase() {
        entriesA = new Complex128[][]{{new Complex128("1.34+14.3i"), new Complex128("1.51-9.51i"), new Complex128("71.5i")},
                {new Complex128("13.55+0i"), new Complex128("-0.00014+14.661i"), new Complex128("7.398+0.98134i")},
                {new Complex128("0.0014+9.55i"), new Complex128("-45.6i"), new Complex128("-94.51+0i")}};
        entriesB = new Complex128[][]{{new Complex128("0"), new Complex128("-94.1-65.1123i"), new Complex128("1.44")},
                {new Complex128("-0.000013+1i"), new Complex128("-1i"), new Complex128("80.441-9.331i")},
                {new Complex128("-8.314-1i"), new Complex128("814.4i"), new Complex128("1.4556+9.4414i")}};
        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);
        exp = A.mult(B.T()).data;

        // ------------ sub-case 1 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTranspose(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTransposeBlocked(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTransposeConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTransposeBlockedConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new Complex128[][]{{new Complex128("1.34+14.3i"), new Complex128("1.51-9.51i"), new Complex128("71.5i")},
                {new Complex128("13.55+0i"), new Complex128("-0.00014+14.661i"), new Complex128("7.398+0.98134i")},
                {new Complex128("0.0014+9.55i"), new Complex128("-45.6i"), new Complex128("-94.51+0i")}};
        entriesB = new Complex128[][]{{new Complex128("0"), new Complex128("-94.1-65.1123i")},
                {new Complex128("-0.000013+1i"), new Complex128("-1i")},
                {new Complex128("-8.314-1i"), new Complex128("814.4i")}};
        exp = new Complex128[]{new Complex128("81.00998037 - 592.9408763700001i"), new Complex128("-57434.09811-1434.3904819999998i"),
                new Complex128("-75.18663199817999 - 15.557191352999999i"), new Complex128("-2059.597296+5142.659675i"),
                new Complex128("831.3561400000001+94.51059280000001i"), new Complex128("576.090725-77867.69015722i")};
        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB).T();
        exp = A.mult(B.T()).data;

        // ------------ sub-case 1 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTranspose(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);

        // ------------ sub-case 2 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTransposeBlocked(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);

        // ------------ sub-case 3 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTransposeConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);

        // ------------ sub-case 4 ------------
        act = new Complex128[A.numRows*B.numRows];
        DenseSemiringMatMultTranspose.multTransposeBlockedConcurrent(A.data, A.shape, B.data, B.shape, act);
        assertArrayEquals(exp, act);
    }
}
