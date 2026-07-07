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
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.ops.dense.real.RealDenseMatMult;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;

/**
 * Static utility class containing real dense matrix multiplication kernels. All Kernels are provided as a {@link BinaryOperator}.
 */
final class ReDeMatMultKernels {

    private ReDeMatMultKernels() {
        // Hide default constructor for utility class.
    }

    // --------------- Matrix-matrix kernels ---------------
    static final BinaryOperator<Matrix> STRD = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.standard(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> REORD = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.reordered(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> BLK = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.blocked(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> BLK_REORD = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.blockedReordered(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> MT_STRD = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.concurrentStandard(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> MT_REORD = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.concurrentReordered(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> MT_BLK_REORD = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, b.numCols),
                RealDenseMatMult.concurrentBlockedReordered(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> STRD_VEC_AS_MAT = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, 1), RealDenseMatMult.standardVector(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> BLK_VEC_AS_MAT = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, 1), RealDenseMatMult.blockedVector(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> MT_STRD_VEC_AS_MAT = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, 1), RealDenseMatMult.concurrentStandardVector(a.data, a.shape, b.data, b.shape));
    };
    static final BinaryOperator<Matrix> MT_BLK_VEC_AS_MAT = (Matrix a, Matrix b) -> {
        return new Matrix(new Shape(a.numRows, 1), RealDenseMatMult.concurrentBlockedVector(a.data, a.shape, b.data, b.shape));
    };

    // --------------- Matrix-vector kernels ---------------
    static final BiFunction<Matrix, Vector, Vector> STRD_VEC = (Matrix a, Vector b) -> {
        return new Vector(RealDenseMatMult.standardVector(a.data, a.shape, b.data, b.shape));
    };
    static final BiFunction<Matrix, Vector, Vector> BLK_VEC = (Matrix a, Vector b) -> {
        return new Vector(RealDenseMatMult.blockedVector(a.data, a.shape, b.data, b.shape));
    };
    static final BiFunction<Matrix, Vector, Vector> MT_STRD_VEC = (Matrix a, Vector b) -> {
        return new Vector(RealDenseMatMult.concurrentStandardVector(a.data, a.shape, b.data, b.shape));
    };
    static final BiFunction<Matrix, Vector, Vector> MT_BLK_VEC = (Matrix a, Vector b) -> {
        return new Vector(RealDenseMatMult.concurrentBlockedVector(a.data, a.shape, b.data, b.shape));
    };
}
