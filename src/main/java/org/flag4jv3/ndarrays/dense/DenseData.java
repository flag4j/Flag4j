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

package org.flag4jv3.ndarrays.dense;

import org.flag4jv3.linalg.internal.kernels.DenseKernelSupport;
import org.flag4jv3.ndarrays.Layout;

import java.util.Objects;

/// Represents the data of a [dense nD-array][DenseNDArrayBase] whose buffer is an array of type [T].
///
/// <blockquote style="color: #306091; background-color: #9da7c2; border-left: 5px solid #4b82bd; padding: 10px;">
///     <strong>Note:</strong> [#buffer()] returns a reference to the internal `buffer` array. This is intentional as it allows
///     kernels to modify it directly.
/// </blockquote>
///
/// @param layout The [Layout] of the [dense nD-array][DenseNDArrayBase].
/// @param buffer The backing data buffer of the [dense nD-array][DenseNDArrayBase]. It will be verified during construction that
/// this `buffer` has the capacity to store the `layout`.
/// @param <T> The type of an individual element of the dense [dense nD-array][DenseNDArrayBase].
/// @see DenseDoubleData
public record DenseData<T>(Layout layout, T[] buffer) {
    public DenseData {
        Objects.requireNonNull(buffer, "buffer must not be null.");
        Objects.requireNonNull(layout, "layout must not be null.");
        DenseKernelSupport.requireCapacity(buffer.length, layout);
    }
}
