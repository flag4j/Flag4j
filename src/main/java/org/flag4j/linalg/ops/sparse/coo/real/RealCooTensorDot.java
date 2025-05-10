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

package org.flag4j.linalg.ops.sparse.coo.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.util.ArrayJoiner;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;


/**
 * <p>Utility class for computing tensor dot products between two {@link CooTensor real sparse COO tensors}.
 */
public final class RealCooTensorDot {

    private RealCooTensorDot() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     * @param src1 First tensor in the contraction.
     * @param src2 Second tensor in the contraction.
     * @param src1Axes Axes along which to compute products for {@code src1} tensor.
     * @param src2Axes Axes along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     * {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor.
     */
    public static Tensor tensorDot(CooTensor src1, CooTensor src2,
                                   int[] src1Axes, int[] src2Axes) {
        // Each array must specify the same number of axes.
        ValidateParameters.ensureAllEqual(src1Axes.length, src2Axes.length);

        // Axis values must be less than the rank of the tensor and non-negative
        ValidateParameters.ensureLessEq(src1.getRank()-1, src1Axes);
        ValidateParameters.ensureAllGreaterEq(0, src1Axes);
        ValidateParameters.ensureLessEq(src2.getRank()-1, src2Axes);
        ValidateParameters.ensureAllGreaterEq(0, src2Axes);

        int[] notin;
        int n1;
        int n2;
        int pos;

        // ---- Compute new axes and shapes for first tensor. ----
        notin = ArrayUtils.notInAxes(src1Axes, src1.getRank());
        int[] src1NewAxes = ArrayJoiner.join(notin, src1Axes);

        n2 = 1;
        for(int axis : src1Axes) {
            n2 *= src1.shape.getSize(axis);
        }

        n1 = 1;
        int[] src1Dims = new int[notin.length];
        pos = 0;
        for(int axis : notin) {
            int a = src1.shape.getSize(axis);
            n1 *= a;
            src1Dims[pos++] = a;
        }

        Shape src1NewShape = new Shape(n1, n2);
        // -----------------------------------------------------

        // ---- Compute new axes and shapes for second tensor. ----
        notin = ArrayUtils.notInAxes(src2Axes, src2.getRank());
        int[] src2NewAxes = ArrayJoiner.join(src2Axes, notin);

        n2 = 1;
        for(int axis : src2Axes) {
            n2 *= src2.shape.getSize(axis);
        }

        n1 = 1;
        pos = 0;
        int[] src2Dims = new int[notin.length];
        for(int axis : notin) {
            int a = src2.shape.getSize(axis);
            n1 *= a;
            src2Dims[pos++] = a;
        }

        Shape src2NewShape = new Shape(n2, n1);
        // -----------------------------------------------------

        // Reform tensor dot product problem as a matrix multiplication problem.
        CooTensor at = src1.T(src1NewAxes).reshape(src1NewShape);
        CooTensor bt = src2.T(src2NewAxes).reshape(src2NewShape);

        // Get row and column indices for COO matrices.
        int[][] atMatIndices = RealDenseTranspose.blockedIntMatrix(at.indices);
        int[][] btMatIndices = RealDenseTranspose.blockedIntMatrix(bt.indices);

        // Compute equivalent matrix multiplication problem.
        double[] productEntries = RealSparseMatMult.standard(
                at.data, atMatIndices[0], atMatIndices[1], at.shape,
                bt.data, btMatIndices[0], btMatIndices[1], bt.shape
        );

        // Reshape to proper N-dimensional shape.
        Shape productShape = new Shape(ArrayJoiner.join(src1Dims, src2Dims));
        return at.makeDenseTensor(productShape, productEntries);
    }
}
