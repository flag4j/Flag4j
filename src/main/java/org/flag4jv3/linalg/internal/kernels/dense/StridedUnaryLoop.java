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


import org.flag4jv3.ndarrays.Layout;

/// Functional interface defining an element-wise unary operation over a single
/// one-dimensional strided run of elements within an [nD-array's][org.flag4jv3.ndarrays.dense.DenseNDArrayBase] data buffer.
///
/// Instances are consumed by the nD dispatch layer, which decomposes an nD traversal into a
/// sequence of such runs (see [StridedRunCursor2]). Implementations are therefore concerned only
/// with a flat, strided walk; all shape and traversal-order logic lives above them.
///
/// <blockquote style="color: #b09dc2; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
/// <strong>Example:</strong> Complex conjugate directly on {@link org.flag4jv3.algebra.elements.Complex128 Complex128} objects.
/// {@snippet lang = "java":
/// StridedUnaryLoop<Complex128> conj = (srcB, srcOff, srcStride,
///                                      outB, outOff, outStride, n) -> {
///     for (int i = 0, sp = srcOff, op = outOff; i < n; i++, sp += srcStride, op += outStride)
///         outB[op] = srcB[sp].conj();
///     };}
/// To then apply this unary operation to a {@link org.flag4jv3.ndarrays.dense.DenseData DenseData<Complex128>} object:
/// {@snippet lang = "java":
/// DenseData<Complex128> out = DenseUOps.applyUOp(src, null, conj);}
/// </blockquote>
///
/// @param <T> The component type of the source and output buffers.
/// @see StridedRunCursor2
/// @see StridedDoubleUnaryLoop
@FunctionalInterface
public interface StridedUnaryLoop<T> {
    /// Applies a unary operation to a run of `n` logical elements of an nD-array.
    ///
    /// <blockquote style="color: #9da7c2; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> All offsets and strides are in terms raw buffer indices. This differs from the
    ///     offsets and strides in a {@link org.flag4jv3.ndarrays.Layout layout} object which are in terms of logical elements.
    ///     For a given logical element stride {@code s} and {@link org.flag4jv3.ndarrays.Layout layout}
    ///     the buffer stride would be {@code s*layout.[itemSize()]} (similarly for offset).
    /// </blockquote>
    ///
    /// <blockquote style="color: #c29d9d; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> Implementations need not be responsible for input validation (e.g., if {@code srcB == outB}
    ///     ensure there is no data clobbering). It is <em>purley</em> the responsibility of the caller of this method to ensure
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
    /// @param n The length of the run (in terms of logical elements).
    /// @implSpec There should be *no* assumption that `srcB` and `outB` are distinct arrays. However, it may be assumed that if
    /// `srcB == outB`, it is safe to apply the unary operation in order for the full run.
    /// There will be no data clobering/overwriting.


    /// Applies a unary operation to a run of `n` logical elements.
    ///
    /// <blockquote style="color: #9da7c2; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> All offsets and strides are in terms raw buffer indices (slots). This differs from the
    ///     offsets and strides in a {@link Layout layout} object which are in terms of logical elements.
    ///     For a layout {@code l} with logical stride {@code s}, the corresponding buffer stride is
    ///     {@code s*l.}{@link Layout#itemSize() itemSize()}. A single logical element occupies {@link Layout#itemSize() itemSize}
    ///     consecutive slots beginning at its buffer position.
    /// </blockquote>
    ///
    /// <blockquote style="color: #c29d9d; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> No input validation is guaranteed. Ensuring the buffers, offsets,
    ///     strides, and run length are consistent and in bounds is <em>purely</em> the caller's responsibility.
    ///     Invalid input yields undefined behavior, including silent data corruption.
    /// </blockquote>
    ///
    /// The caller guarantees the following, and implementations may rely on them:
    /// - Every buffer slot touched by either run is in bounds for its buffer.
    /// - The two runs do not destructively interfere: for every element `i`, no slot read as input for
    ///   element `i` is written as output for any element `j != i`. Consequently, the run may be processed
    ///   in any order. This holds trivially when `srcB != outB`; when `srcB == outB` the dispatch layer
    ///   establishes it, either because the runs address disjoint regions of the buffer or because they
    ///   address the buffer identically (element `i`'s input slots are exactly element `i`'s output slots).
    ///   Note that the latter case is the reason for the read-before-write requirement below.
    ///
    /// `srcBufStride` may be zero (a broadcast source) or negative (a reversed view). `outBufStride` is
    /// never zero. `srcOff` and `outOff` always address the first slot of their respective elements.
    ///
    /// @param srcB The source buffer providing inputs to the unary operation.
    /// @param srcOff The buffer index of the first input element within `srcB`.
    /// @param srcBufStride The stride, in buffer indices, between consecutive input elements within `srcB`.
    /// @param outB The output buffer in which results are stored.
    /// @param outOff The buffer index at which the first result is stored within `outB`.
    /// @param outBufStride The stride, in buffer indices, between consecutive results within `outB`.
    /// @param n The number of logical elements in the run. May be zero.
    /// @implSpec - Treat `n == 0` as a no-op.
    /// - Read *all* input components of an element before writing *any* output component of that element.
    ///   For `itemSize > 1` this is required for in-place correctness: writing `outB[p]` before reading
    ///   `srcB[p + 1]` corrupts the input when `srcB == outB`.
    /// - Write exactly `n` results and touch no slot outside the two runs described above.
    /// - Be safe for concurrent invocation. A single instance is shared across all worker threads of a
    ///   parallel dispatch; implementations should be stateless or otherwise thread-safe.
    void apply(T[] srcB, int srcOff, int srcBufStride,
               T[] outB, int outOff, int outBufStride, int n);
}
