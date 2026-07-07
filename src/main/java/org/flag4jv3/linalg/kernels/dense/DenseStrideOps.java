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

package org.flag4jv3.linalg.kernels.dense;

import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.Shape;
import org.flag4jv3.util.arrays.ArrayBuilder;

import java.util.Objects;

/// Low-level utility class for operations working on the strides of dense nD arrays.
///
/// **WARNING**: This class does perform *some* error checking, but it is not exhaustive.
/// This class is mostly intended for internal use only. Use this class *only* if you absolutely know what you are doing
/// and exercise caution.
public final class DenseStrideOps {

    private DenseStrideOps() {
    } // Hide constructor for utility class.


    /// Ensures that an nD array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD array specifying the offset and strides.
    /// @param dst The array to store the 1D buffer data of the contiguous result in. May be `null`.
    ///                 If `src` is already contiguous, then `dst` will be ignored (see return).
    ///                 If `dst` is `null`, then a new array will be created and returned. If `dst` *is not* `null`, then
    ///                 it must have the same length as `layout.shape.numelIntValueExact()` and a reference to `dst` will be returned.
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. If `src` is *not* already contiguous and `dst` is *not* `null`, then a reference to `dst` will be returned.
    ///                 Otherwise, if `src` is *not* and `dst` is `null`, a new array will be created and returned.
    ///
    /// @throws IllegalArgumentException If `dst != null && dst.length != layout.shape.numelIntValueExact()`
    public static boolean[] makeContiguous(boolean[] src, Layout layout, boolean[] dst) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous) {
            return src;
        }

        int rank = layout.shape.rank;
        int total = layout.shape.numelIntValueExact();

        dst = ArrayBuilder.getOrCreateArray(dst, total);

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = layout.shape.dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dst[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dst;
    }


    /// Ensures that an nD array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD array specifying the offset and strides.
    /// @param dst The array to store the 1D buffer data of the contiguous result in. May be `null`.
    ///                 If `src` is already contiguous, then `dst` will be ignored (see return).
    ///                 If `dst` is `null`, then a new array will be created and returned. If `dst` *is not* `null`, then
    ///                 it must have the same length as `layout.shape.numelIntValueExact()` and a reference to `dst` will be returned.
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. If `src` is *not* already contiguous and `dst` is *not* `null`, then a reference to `dst` will be returned.
    ///                 Otherwise, if `src` is *not* and `dst` is `null`, a new array will be created and returned.
    ///
    /// @throws IllegalArgumentException If `dst != null && dst.length != layout.shape.numelIntValueExact()`
    public static int[] makeContiguous(int[] src, Layout layout, int[] dst) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous) {
            return src;
        }

        int rank = layout.shape.rank;
        int total = layout.shape.numelIntValueExact();

        dst = ArrayBuilder.getOrCreateArray(dst, total);

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = layout.shape.dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dst[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dst;
    }


    /// Ensures that an nD array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD array specifying the offset and strides.
    /// @param dst The array to store the 1D buffer data of the contiguous result in. May be `null`.
    ///                 If `src` is already contiguous, then `dst` will be ignored (see return).
    ///                 If `dst` is `null`, then a new array will be created and returned. If `dst` *is not* `null`, then
    ///                 it must have the same length as `layout.shape.numelIntValueExact()` and a reference to `dst` will be returned.
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. If `src` is *not* already contiguous and `dst` is *not* `null`, then a reference to `dst` will be returned.
    ///                 Otherwise, if `src` is *not* and `dst` is `null`, a new array will be created and returned.
    ///
    /// @throws IllegalArgumentException If `dst != null && dst.length != layout.shape.numelIntValueExact()`
    public static long[] makeContiguous(long[] src, Layout layout, long[] dst) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous) {
            return src;
        }

        int rank = layout.shape.rank;
        int total = layout.shape.numelIntValueExact();

        dst = ArrayBuilder.getOrCreateArray(dst, total);

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = layout.shape.dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dst[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dst;
    }


    /// Ensures that an nD array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD array specifying the offset and strides.
    /// @param dst The array to store the 1D buffer data of the contiguous result in. May be `null`.
    ///                 If `src` is already contiguous, then `dst` will be ignored (see return).
    ///                 If `dst` is `null`, then a new array will be created and returned. If `dst` *is not* `null`, then
    ///                 it must have the same length as `layout.shape.numelIntValueExact()` and a reference to `dst` will be returned.
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. If `src` is *not* already contiguous and `dst` is *not* `null`, then a reference to `dst` will be returned.
    ///                 Otherwise, if `src` is *not* and `dst` is `null`, a new array will be created and returned.
    ///
    /// @throws IllegalArgumentException If `dst != null && dst.length != layout.shape.numelIntValueExact()`
    public static float[] makeContiguous(float[] src, Layout layout, float[] dst) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous) {
            return src;
        }

        int rank = layout.shape.rank;
        int total = layout.shape.numelIntValueExact();

        dst = ArrayBuilder.getOrCreateArray(dst, total);

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = layout.shape.dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dst[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dst;
    }


    /// Ensures that an nD array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD array specifying the offset and strides.
    /// @param dst The array to store the 1D buffer data of the contiguous result in. May be `null`.
    ///                 If `src` is already contiguous, then `dst` will be ignored (see return).
    ///                 If `dst` is `null`, then a new array will be created and returned. If `dst` *is not* `null`, then
    ///                 it must have the same length as `layout.shape.numelIntValueExact()` and a reference to `dst` will be returned.
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. If `src` is *not* already contiguous and `dst` is *not* `null`, then a reference to `dst` will be returned.
    ///                 Otherwise, if `src` is *not* and `dst` is `null`, a new array will be created and returned.
    ///
    /// @throws IllegalArgumentException If `dst != null && dst.length != layout.shape.numelIntValueExact()`
    public static double[] makeContiguous(double[] src, Shape shape, Layout layout, double[] dst) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous) {
            return src;
        }

        int rank = shape.rank;
        int total = layout.shape.numelIntValueExact();

        dst = ArrayBuilder.getOrCreateArray(dst, total);

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
                dst[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dst;
    }


    /// Ensures that an nD array is contiguous with respect to a [Layout].
    ///
    /// @param src The 1D buffer of the nD array to make contiguous. Must *not* be null.
    /// @param layout The layout of the nD array specifying the offset and strides.
    /// @param dst The array to store the 1D buffer data of the contiguous result in.
    ///                 Must have the same length as `layout.shape.numelIntValueExact()` Must *not* be `null`
    ///                 if `src` is not already contiguous. If `src` is already contiguous, then `dst` will be ignored (see return).
    /// @return A 1D array with the same length and elements as `src` that represents a contiguous data buffer
    ///                 of an nD array with the given `layout`. If `src` is already contiguous, this will be a reference to
    ///                 `src`. Otherwise, this will be a reference to `dst`.
    ///
    /// @throws IllegalArgumentException If `dst.length != layout.shape.numelIntValueExact()`
    public static Object[] makeContiguous(Object[] src, Layout layout, Object[] dst) {
        Objects.requireNonNull(src, "src must not be null.");

        if (layout.isContiguous) {
            return src;
        }

        Objects.requireNonNull(dst, "dst must not be null for non-contiguous Object arrays.");

        int rank = layout.shape.rank;
        int total = layout.shape.numelIntValueExact();

        if (dst.length != total) {
            throw new IllegalArgumentException(
                    "dst must have the same length as total elements in shape but got "
                            + dst.length + " and " + total + "."
            );
        }

        int[] idx = new int[rank];
        int srcPos = layout.offset();
        int[] strides = layout.strides();
        int[] dims = layout.shape.dims();

        int lastDim = rank > 0 ? dims[rank - 1] : 1;
        int lastStride = rank > 0 ? strides[rank - 1] : 0;
        int outer = total/Math.max(lastDim, 1);

        for (int o = 0, i = 0; o < outer; o++) {
            int p = srcPos;
            for (int j = 0; j < lastDim; j++, i++, p += lastStride)
                dst[i] = src[p];
            srcPos = incrementOuter(idx, strides, dims, srcPos, rank - 1);
        }

        return dst;
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
