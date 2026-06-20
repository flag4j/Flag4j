/*
 * MIT License
 *
 * Copyright (c) 2022-2026. Jacob Watters
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
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ValidateParameters;

/**
 * This class provides low level methods for computing ops on real dense tensors.
 */
public final class RealDenseOps {

    private RealDenseOps() {
        // Hide constructor for utility class.
    }


    /**
     * Computes the element-wise addition of two tensors.
     *
     * @param src1 Entries of first Tensor of the addition.
     * @param src2 Entries of second Tensor of the addition.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src1} or {@code src2}.
     *
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array will be created and
     * returned.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     */
    public static double[] add(double[] src1, double[] src2, double[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        int length = src1.length;
        dest = ArrayBuilder.getOrCreateArray(dest, length);

        for(int i=0; i<length; i++)
            dest[i] = src1[i] + src2[i];

        return dest;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param src1 Entries of first tensor.
     * @param src2 Entries of second tensor.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src1} or {@code src2}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array will be created and
     * returned.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     */
    public static double[] sub(double[] src1, double[] src2, double[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);

        int length = src1.length;
        dest = ArrayBuilder.getOrCreateArray(dest, length);

        for(int i=0; i<length; i++)
            dest[i] = src1[i] - src2[i];

        return dest;
    }


    /**
     * Subtracts a scalar from every element of a tensor.
     * @param src Entries of tensor to add scalar to.
     * @param b Scalar to subtract from tensor.
     * @param Array to store the result in. May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned.
     * Otherwise, a new array will be created and returned.
     */
    public static double[] sub(double[] src, double b, double[] dest) {
        int length = src.length;
        dest = ArrayBuilder.getOrCreateArray(dest, length);

        for(int i=0; i<length; i++)
            dest[i] = src[i] - b;

        return dest;
    }


    /**
     * Computes element-wise subtraction between tensors and stores the result in the first tensor.
     * @param src1 First tensor in subtraction. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the subtraction.
     * @param shape2 Shape of the second tensor.
     * @throws IllegalArgumentException If tensors are not the same shape.
     */
    public static void subEq(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, length = src1.length; i<length; i++)
            src1[i] -= src2[i];
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static void subEq(double[] src, double b) {
        for(int i=0, length=src.length; i<length; i++)
            src[i] -= b;
    }


    /**
     * Computes element-wise addition between tensors and stores the result in the first tensor.
     * @param src1 First tensor in addition. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the addition.
     * @param shape2 Shape of the second tensor.
     * @throws IllegalArgumentException If tensors are not the same shape.
     */
    public static void addEq(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, length = src1.length; i<length; i++)
            src1[i] += src2[i];
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static void addEq(double[] src, double b) {
        for(int i=0, length = src.length; i<length; i++)
            src[i] += b;
    }


    /**
     * Multiplies all data in a tensor.
     * @param src The data of the tensor.
     * @return The product of all data in the tensor.
     */
    public static double prod(double[] src) {
        if(src == null || src.length == 0) return 0;
        double product=1;

        for(double value : src)
            product *= value;

        return product;
    }


    /**
     * Multiplies all data in a tensor.
     * @param src The data of the tensor.
     * @return The product of all data in the tensor.
     */
    public static int prod(int[] src) {
        if(src == null || src.length == 0) return 0;
        int product=1;

        for(int value : src)
            product *= value;

        return product;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise reciprocals of the tensor.
     */
    public static double[] recip(double[] src) {
        double[] receps = new double[src.length];

        for(int i=0; i<receps.length; i++)
            receps[i] = 1.0/src[i];

        return receps;
    }


    /**
     * Adds a scalar to every element of a tensor.
     * @param src src of tensor to add scalar to.
     * @param b Scalar to add to tensor.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. If {@code dest == null} then a new array
     * will be created and returned.
     */
    public static double[] add(double[] src, double b, double[] dest) {
        int length = src.length;
        dest = ArrayBuilder.getOrCreateArray(dest, length);

        for(int i=0; i<length; i++)
            dest[i] = src[i] + b;

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
     * @param destShape The resulting shape of the tensor trace.
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
    public static void tensorTr(Shape shape, double[] src,
                                int axis1, int axis2,
                                Shape destShape, double[] dest) {
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
            double sum = src[baseOffset];
            int offset = baseOffset + diagonalStride;
            for(int diag=1; diag<traceLength; diag++) {
                sum += src[offset];
                offset += diagonalStride;
            }

            dest[i] = sum;
        }
    }


    /**
     * <p>Swaps two rows, over a specified range of columns, within a matrix. Specifically, all elements in the matrix within rows
     * {@code rowIdx1}
     * and {@code rowIdx2} and between columns {@code start} (inclusive) and {@code stop} (exclusive). This operation is done in-place.
     * <p>No bounds checking is done within this method to ensure that the indices provided are valid.
     *
     * @param shape Shape of the matrix.
     * @param data Data of the matrix (modified).
     * @param rowIdx1 Index of the first row to swap.
     * @param rowIdx2 Index of the second row to swap.
     * @param start Index of the column specifying the start of the range for the row swap (inclusive).
     * @param stop Index of the column specifying the end of the range for the row swap (exclusive).
     */
    public static void swapRowsUnsafe(Shape shape,  double[] data, int rowIdx1, int rowIdx2, int start, int stop) {
        if(rowIdx1 == rowIdx2) return;

        final int cols = shape.getSize(1);
        final int rowOffset1 = rowIdx1*cols;
        final int rowOffset2 = rowIdx2*cols;
        double temp;

        for(int j=start; j<stop; j++) {
            temp = data[rowOffset1 + j];
            data[rowOffset1 + j] = data[rowOffset2 + j];
            data[rowOffset2 + j] = temp;
        }
    }


    /**
     * <p>Swaps two columns, over a specified range of rows, within a matrix. Specifically, all elements in the matrix within columns
     * {@code colIdx1} and {@code colIdx2} and between rows {@code start} (inclusive) and {@code stop} (exclusive). This operation
     * is done in-place.
     * <p>No bounds checking is done within this method to ensure that the indices provided are valid.
     *
     * @param shape Shape of the matrix.
     * @param data Data of the matrix (modified).
     * @param colIdx1 Index of the first column to swap.
     * @param colIdx2 Index of the second column to swap.
     * @param start Index of the row specifying the start of the range for the row swap (inclusive).
     * @param stop Index of the row specifying the end of the range for the row swap (exclusive).
     */
    public static void swapColsUnsafe(Shape shape,  double[] data, int colIdx1, int colIdx2, int start, int stop) {
        if(colIdx1 == colIdx2) return;
        
        final int cols = shape.getSize(1);
        int rowOffset = start*cols;
        double temp;

        for(int i=start; i<stop; i++) {
            temp = data[rowOffset + colIdx1];
            data[rowOffset + colIdx1] = data[rowOffset + colIdx2];
            data[rowOffset + colIdx2] = temp;
            rowOffset += cols;
        }
    }
}
