/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
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

package org.flag4jv3.linalg.internal.kernels.dense;

import org.flag4jv3.ndarrays.ContiguousOrder;
import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.Shape;
import org.flag4jv3.util.arrays.ArrayBuilder;

import java.util.Objects;

/// Low-level utility class for operations working on the strides of dense nD arrays.
///
/// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
///     <strong>Warning:</strong> This class preforms minimal error checking and is <em>not</em> exhaustive.
///     This class is mostly intended for internal use only. Use this class *only* if you absolutely know what you are doing
///     and exercise caution.
/// </blockquote>
public final class DenseStrideOps {

    private DenseStrideOps() {
    } // Hide constructor for utility class.

    // TODO NOW: Should we be using Dense<TYPE>Data for this class instead? Probably.
    // TODO NOW: The order here is now important. These methods are not WRONG unless order is C ordering.


    /// Ensures that an nD-array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD-array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD-array specifying the offset and strides.
    /// @param dest The array to store the 1D buffer data of the contiguous result in. May be `null`.
    ///                 If `src` is already contiguous, then `dest` will be ignored (see return).
    ///                 If `dest` is `null`, then a new array will be created and returned. If `dest` *is not* `null`, then
    ///                 it must have the same length as `layout.shape().numelIntValueExact()` and a reference to `dest` will be returned.
    /// @param order The [order][ContiguousOrder] of the contiguous buffer to return.
    /// Must be either [C][ContiguousOrder#C] or [F][ContiguousOrder#F].
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD-array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. If `src` is *not* already contiguous and `dest` is *not* `null`, then a reference to `dest` will be returned.
    ///                 Otherwise, if `src` is *not* and `dest` is `null`, a new array will be created and returned.
    ///
    /// @throws IllegalArgumentException If `order` is not [C][ContiguousOrder#C] or [F][ContiguousOrder#F].
    /// @throws IllegalArgumentException If `dest != null && dest.length != layout.shape().numelIntValueExact()`
    public static double[] makeContiguous(double[] src, Shape shape, Layout layout, double[] dest, ContiguousOrder order) {
        ContiguousOrder.ensureCorFExact(order);
        Objects.requireNonNull(src, "src must not be null.");


        if (layout.isContiguous(order)) {
            return src;
        }

        int rank = shape.rank();
        int total = layout.shape().numelIntValueExact();

        dest = ArrayBuilder.getOrCreateArray(dest, total);

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = shape.dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dest[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dest;
    }


    /// Ensures that an nD-array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD-array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD-array specifying the offset and strides.
    /// @param dest The array to store the 1D buffer data of the contiguous result in.
    ///                 Must have the same length as `layout.shape().numelIntValueExact()` Must *not* be `null`
    ///                 if `src` is not already contiguous. If `src` is already contiguous, then `dest` will be ignored (see return).
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD-array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. Otherwise, this will be a reference to `dest`.
    ///
    /// @throws IllegalArgumentException If `dest.length != layout.shape().numelIntValueExact()`
    public static Object[] makeContiguous(Object[] src, Layout layout, Object[] dest, ContiguousOrder order) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous(order)) {
            return src;
        }

        Objects.requireNonNull(dest, "dest must not be null for non-contiguous Object arrays.");

        int rank = layout.rank();
        int total = layout.shape().numelIntValueExact();

        if (dest.length != total) {
            throw new IllegalArgumentException(
                    "dest must have the same length as total elements in shape but got "
                            + dest.length + " and " + total + "."
            );
        }

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = layout.shape().dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dest[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dest;
    }


    /// Helper for incrementing odometer.
    private static int incrementOuter(int[] idx, int[] strides, int[] dims, int srcPos, int rank) {
        for (int axis = rank - 1; axis >= 0; axis--) {
            idx[axis]++;
            srcPos += strides[axis];
            if (idx[axis] < dims[axis]) break;      // no carry
            idx[axis] = 0;                          // carry to next axis
            srcPos -= dims[axis]*strides[axis];   // rewind this axis
        }

        return srcPos;
    }
}
