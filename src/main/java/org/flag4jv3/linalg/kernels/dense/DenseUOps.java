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
import org.flag4jv3.util.NewValidateParameters;

import java.util.Objects;
import java.util.function.UnaryOperator;

public final class DenseUOps {

    private DenseUOps() {
    } // Hide default constructor for utility class.


    /**
     * Applies an operation to each element of an nD array in order (c-ordering).
     *
     * @param buffer The data buffer of the nD array to apply the operation to.
     * @param shape The shape of the nD array.
     * @param layout The layout of the nD array.
     * @param op The operation to apply
     * @param out The buffer to store the result in. This may be the same object as `buffer`. May *not* be `null`.
     * @param <T>
     * @return
     */
    public <T> T[] applyUOp(T[] buffer, Layout layout, UnaryOperator<T> op, T[] out) {
        Objects.requireNonNull(buffer, "buffer must not be null");
        Objects.requireNonNull(out, "out must not be null");
        NewValidateParameters.validateBufferSize(buffer.length, out.length, "out must be at least as large as buffer.");

        if (layout.isContiguous) {
            // Simple contiguous case.
            return applyContiguous(buffer, op, out);
        }

        return (buffer == out)
                ? applyInPlace(buffer, layout, op)
                : applyOutOfPlace(buffer, layout, op, out);
    }


    private <T> T[] applyOutOfPlace(T[] buffer, Layout layout, UnaryOperator<T> op, T[] out) {


        return out;
    }


    private <T> T[] applyInPlace(T[] buffer, Layout layout, UnaryOperator<T> op) {
        long size = layout.shape.numelIntValueExact();
        if (size == 0) return buffer;

        int[] dims = layout.shape.dims();
        int[] strides = layout.strides();
        int offset = layout.offset;

        if (dims.length == 0) { // Rank-0 (scalar) view.
            buffer[offset] = op.apply(buffer[offset]);
            return buffer;
        }

        int[] cDims = dims.clone();
        int[] cStrides = strides.clone();
        int rank = coalesce(cDims, cStrides);

        final int innerDim = cDims[rank - 1];
        final int innerStride = cStrides[rank - 1];

        if (rank == 1) {
            applyRun(buffer, offset, innerDim, innerStride, op);
            return buffer;
        }

        int[] counter = new int[rank - 1];
        long outer = size/innerDim;
        int base = offset;

        for (long c = 0; c < outer; c++) {
            applyRun(buffer, base, innerDim, innerStride, op);
            base = advance(base, counter, cDims, cStrides, rank - 2);
        }

        return buffer;
    }


    private <T> T[] applyContiguous(T[] buffer, UnaryOperator<T> op, T[] out) {
        for (int i = 0; i < buffer.length; i++) {
            out[i] = op.apply(buffer[i]);
        }

        return out;
    }
}
