/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.ops.dispatch;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMult;
import org.flag4j.numbers.Complex128;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;

/**
 * Static utility class containing complex dense matrix multiplication kernels. All Kernels are provided as a {@link BinaryOperator}.
 */
final class Cm128DeMatMultKernels {

    private Cm128DeMatMultKernels() {
        // Hide default constructor for utility class.
    }

    // --------------- matrix-matrix kernels ---------------
    static final BinaryOperator<CMatrix> STRD = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.standard(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> REORD = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.reordered(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> BLK = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.blocked(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> BLK_REORD = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.blockedReordered(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> MT_STRD = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.concurrentStandard(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> MT_REORD = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.concurrentReordered(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> MT_BLK_REORD = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows*b.numRows];
        DenseSemiringMatMult.concurrentBlockedReordered(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, b.numCols), dest);
    };
    static final BinaryOperator<CMatrix> STRD_VEC_AS_MAT = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.standardVector(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, 1), dest);
    };
    static final BinaryOperator<CMatrix> BLK_VEC_AS_MAT = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.blockedVector(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, 1), dest);
    };
    static final BinaryOperator<CMatrix> MT_STRD_VEC_AS_MAT = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.concurrentStandardVector(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, 1), dest);
    };
    static final BinaryOperator<CMatrix> MT_BLK_VEC_AS_MAT = (CMatrix a, CMatrix b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.concurrentBlockedVector(a.data, a.shape, b.data, b.shape, dest);
        return new CMatrix(new Shape(a.numRows, 1), dest);
    };

    // --------------- matrix-vector kernels ---------------
    static final BiFunction<CMatrix, CVector, CVector> STRD_VEC = (CMatrix a, CVector b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.standardVector(a.data, a.shape, b.data, b.shape, dest);
        return new CVector(dest);
    };
    static final BiFunction<CMatrix, CVector, CVector> BLK_VEC = (CMatrix a, CVector b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.blockedVector(a.data, a.shape, b.data, b.shape, dest);
        return new CVector(dest);
    };
    static final BiFunction<CMatrix, CVector, CVector> MT_STRD_VEC = (CMatrix a, CVector b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.concurrentStandardVector(a.data, a.shape, b.data, b.shape, dest);
        return new CVector(dest);
    };
    static final BiFunction<CMatrix, CVector, CVector> MT_BLK_VEC = (CMatrix a, CVector b) -> {
        Complex128[] dest = new Complex128[a.numRows];
        DenseSemiringMatMult.concurrentBlockedVector(a.data, a.shape, b.data, b.shape, dest);
        return new CVector(dest);
    };
}
