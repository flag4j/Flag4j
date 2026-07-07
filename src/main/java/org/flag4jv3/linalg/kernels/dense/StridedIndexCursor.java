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

/**
 * A reusable cursor which yields, one at a time, the flat buffer indices of a strided view in
 * row-major order. Returns primitive {@code int}s to avoid the boxing an {@code Iterator<Integer>}
 * would incur. Useful when the traversal must be generic over buffer type (e.g. {@code double[]},
 * {@code T[]}) or interleaved with other logic, but note that per-element method calls are
 * measurably slower than the blocked loops in {@link #applyInPlace} / {@link #apply}.
 */
public final class StridedIndexCursor {
    private final int[] dims;
    private final int[] strides;
    private final int[] counter;
    private final int rank;
    private long remaining;
    private int index;


    /**
     * @param offset Index in the buffer of the element at nD index (0, 0, ..., 0).
     * @param dims Dimensions of the view.
     * @param strides Buffer step size along each axis.
     */
    public StridedIndexCursor(Layout layout) {
        this.dims = layout.shape.dims();
        this.strides = layout.strides();
        this.rank = coalesce(this.dims, this.strides);
        this.counter = new int[rank];
        this.remaining = layout.shape.numelIntValueExact();
        this.index = layout.offset;
    }


    /**
     * @return {@code true} if there are remaining elements in the traversal.
     */
    public boolean hasNext() {
        return remaining > 0;
    }


    /**
     * @return The flat buffer index of the next element in row-major order.
     */
    public int next() {
        int current = index;

        if (--remaining > 0) {
            for (int axis = rank - 1; axis >= 0; axis--) {
                if (++counter[axis] < dims[axis]) {
                    index += strides[axis];
                    break;
                }
                counter[axis] = 0;
                index -= strides[axis]*(dims[axis] - 1);
            }
        }

        return current;
    }
}
