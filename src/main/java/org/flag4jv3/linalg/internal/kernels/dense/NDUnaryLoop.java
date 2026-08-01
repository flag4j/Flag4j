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

/// A functional interface defining a unary operation on a contiguous run of `n` logical elements of an nD-array.
@FunctionalInterface
public interface NDUnaryLoop<T> {
    /// Applies a unary operation to a run of `n` logical elements of an nD-array.
    ///
    /// <blockquote style="color: #306091; background-color: #9da7c2; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> All offsets and strides are in terms raw buffer indices. This differs from the
    ///     offsets and strides in a [org.flag4jv3.ndarrays.Layout] object which are in terms of logical elements.
    ///     For a given logical element stride `s` and [layout][org.flag4jv3.ndarrays.Layout] `layout`
    ///     the buffer stride would be `s*layout.[itemSize()]` (similarly for offset).
    /// </blockquote>
    ///
    /// <blockquote style="color: #692929; background-color: #c29d9d; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> Implementations need not be responsible for input validation (e.g., if `srcB == outB`
    ///     ensure there is no data clobbering). It is *purley* the responsibility of the caller of this method to ensure
    ///     the inputs are valid. Invalid input may lead to undefined behaviour.
    /// </blockquote>
    ///
    /// @param srcB The source buffer providing input to the unary operation.
    /// @param srcOff The offset (in buffer indices) to the starting input to the unary operation within `srcB`.
    /// @param srcBufStride The stride (in buffer indices) between inputs to the unary operation within `srcB`.
    /// @param outB The output buffer to store the results of the unary operation in.
    /// @param outOff The offset (in buffer indices) to the starting position to store the first output of the unary operation within
    /// `outB`.
    /// @param outBufStride The stride (in buffer indices) between outputs of the unary operation to store within `outB`.
    /// @param n The length of the run (in buffre indices).
    /// @implSpec There should be *no* assumption that `srcB` and `outB` are distinct arrays. However, it may be assumed that if
    /// `srcB == outB`, it is safe to apply the unary operation in order for the full run.
    /// There will be no data clobering/overwriting.
    void apply(T[] srcB, int srcOff, int srcBufStride,
               T[] outB, int outOff, int outBufStride, int n);
}
