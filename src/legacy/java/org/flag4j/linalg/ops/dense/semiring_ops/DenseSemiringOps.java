/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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

package org.flag4j.linalg.ops.dense.semiring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ValidateParameters;

/**
 * This class provides low level methods for computing ops on dense semiring tensors.
 */
public final class DenseSemiringOps {

    private DenseSemiringOps() {
        // Hide constructor for utility class.
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @param dest Array to store result in. May be {@code null} or the same array as {@code src1} or {@code src2}.
     * @return The {@code dest} array if {@code dest != null}. If {@code dest != null}
     * @throws IllegalArgumentException If entry ndarrays are not the same size.
     */
    public static <T extends SemiringElement<T>> T[] add(T[] src1, Shape shape1,
                                                         T[] src2, Shape shape2,
                                                         T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        dest = ArrayBuilder.getOrCreateArray(dest, () -> (T[]) new SemiringElement[src1.length]);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = src1[i].add(src2[i]);

        return dest;
    }



    /**
     * Computes the element-wise product of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @param dest Array to store result in. May be {@code null}.
     * @return The {@code dest} array if {@code dest != null}. If {@code dest != null}
     * @throws IllegalArgumentException If {@code shape1.equals(shape2)}.
     */
    public static <T extends SemiringElement<T>> T[] elemMult(T[] src1, Shape shape1,
                                                              T[] src2, Shape shape2,
                                                              T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        dest = ArrayBuilder.getOrCreateArray(dest, () -> (T[]) new SemiringElement[src1.length]);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = src1[i].add(src2[i]);

        return dest;
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values in the 2D subarrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param shape Shape of the tensor to compute the trace of.
     * @param src Entries of the tensor to compute the trace of.
     * @param axis1 First axis for 2D subarray.
     * @param axis2 Second axis for 2D subarray.
     * @param destShape The resulting shape of the tensor trace. Use {@link #getTrShape(Shape, int, int)} to compute this.
     * @param dest Array to store the result of the generalized tensor trace of. Must satisfy
     * {@code dest.length == destShape.totalEntriesIntValueExact()}.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensor's rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e., the axes are equal or the tensor does not have the same length along the two axes.)
     * @throws IllegalArgumentException If {@code dest.length == destShape.totalEntriesIntValueExact()}.
     */
    public static <T extends SemiringElement<T>> void tensorTr(Shape shape, T[] src,
                                                               int axis1, int axis2,
                                                               Shape destShape, T[] dest) {
        ValidateParameters.ensureArrayLengthsEq(destShape.numelIntValueExact(), dest.length);
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.validateArrayIndices(shape.getRank(), axis1, axis2);
        ValidateParameters.ensureAllEqual(shape.getSize(axis1), shape.getSize(axis2));

        int[] strides = shape.getStrides();
        int rank = strides.length;

        // Calculate the offset increment for the diagonal.
        int traceLength = shape.getSize(axis1);
        int diagonalStride = strides[axis1] + strides[axis2];

        int[] destIndices = new int[rank - 2];
        for(int i=0; i<dest.length; i++) {
            destIndices = destShape.getNdIndices(i);

            int baseOffset = 0;
            int idx = 0;

            // Compute offset for mapping destination indices to indices in this tensor.
            for(int j=0; j<rank; j++) {
                if(j != axis1 && j != axis2) {
                    baseOffset += destIndices[idx++]*strides[j];
                }
            }

            // Sum over diagonal elements of the 2D subarray.
            T sum = src[baseOffset];
            int offset = baseOffset + diagonalStride;
            for(int diag=1; diag<traceLength; diag++) {
                sum = sum.add(src[offset]);
                offset += diagonalStride;
            }

            dest[i] = sum;
        }
    }


    /**
     * Computes the shape of the tensor resulting from the generalized tensor trace along the specified axes for a tensor with the
     * specified shape.
     * @param shape Shape of the tensor to compute the generalized tensor trace of.
     * @param axis1 First axis to compute the tensor trace along.
     * @param axis2 Second axis to compute the tensor trace along.
     * @return
     */
    public static Shape getTrShape(Shape shape, int axis1, int axis2) {
        final int rank = shape.getRank();
        int idx = 0;
        int[] newDims = new int[rank - 2];

        // Compute shape for resulting tensor.
        for(int i=0; i<rank; i++)
            if(i != axis1 && i != axis2) newDims[idx++] = shape.getSize(i);

        return new Shape(newDims);
    }
}
