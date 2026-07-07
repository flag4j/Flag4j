/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.linalg.ops.RealDenseMatrixMultiplyDispatcher;
import org.flag4j.linalg.ops.TensorDot;
import org.flag4j.linalg.ops.TransposeDispatcher;

/**
 * Instances of this class can be used to compute the tensor dot product between two real dense primitive double tensors.
 */
public class RealDenseTensorDot extends TensorDot<double[]> {

    /**
     * Constructs a tensor dot product problem for computing the tensor contraction of two tensors over the
     * specified set of axes. That is, computes the sum of products between the two tensors along the specified set of axes.
     * @param shape1 Shape of the first tensor in the contraction.
     * @param src1 Entries of the first tensor in the contraction.
     * @param shape2 Shape of the second tensor in the contraction.
     * @param src2 Entries of the second tensor in the contraction.
     * @param src1Axes Axes along which to compute products for {@code src1} tensor.
     * @param src2Axes Axes along which to compute products for {@code src2} tensor.
     * @throws IllegalArgumentException If {@code src1Axes} and {@code src2Axes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor. Or, If the two tensor's shapes do not match along the specified axes pairwise
     * in {@code src1Axes} and {@code src2Axes}.
     */
    public RealDenseTensorDot(Shape shape1, double[] src1,
                              Shape shape2, double[] src2,
                              int[] src1Axes, int[] src2Axes) {
        super(shape1, src1, shape2, src2, src1Axes, src2Axes);
    }


    /**
     * <p>Computes this tensor dot product as specified in the constructor.
     * <p>It is recommended to use {@link #compute()} over this method as it will reduce excess copying.
     * @param dest The array to store the items of the tensor resulting from this tensor dot product. The size of this array
     * should be computed using {@link #getOutputSize()}.
     */
    @Override
    public void compute(double[] dest) {
        if(dest.length != destLength) {
            throw new IllegalArgumentException("dest array is not properly sized to store the result of the tensor dot product." +
                    " Expecting length " + destLength + " but got " + dest.length +  ". Try using calling" +
                    "getOutputSize() to determine the required size of the dest array.");
        }

        // Reform tensor dot product problem as a matrix multiplication problem.
        double[] at = TransposeDispatcher.dispatchTensor(src1, shape1, src1NewAxes);
        double[] bt = TransposeDispatcher.dispatchTensor(src2, shape2, src2NewAxes);

        double[] destEntries = RealDenseMatrixMultiplyDispatcher.dispatch(at, newShape1, bt, newShape2);
        // TODO: The RealDenseMatrixMultiplyDispatcher should be refactored to avoid needing to copy the result.
        System.arraycopy(destEntries, 0, dest, 0, destLength);
    }


    /**
     * Computes this tensor dot product as specified in the constructor.
     * @return The result of the tensor dot product problem specified in the constructor.
     */
    public double[] compute() {
        // Reform tensor dot product problem as a matrix multiplication problem.
        double[] at = TransposeDispatcher.dispatchTensor(src1, shape1, src1NewAxes);
        double[] bt = TransposeDispatcher.dispatchTensor(src2, shape2, src2NewAxes);

        return RealDenseMatrixMultiplyDispatcher.dispatch(at, newShape1, bt, newShape2);
    }
}
