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

package org.flag4jv3.ndarrays;

/// Specifies possible contiguous orderings (including [NONE]) of an [nD-array][org.flag4jv3.ndarrays.dense.DenseNDArrayBase]
public enum ContiguousOrder {
    /// Is neither [C] nor [F] contiguous.
    NONE,
    /// Is C contiguous &mdash; last axis varies fastest. That is, is contiguous and has row-major ordering
    ///  like in the C programming language.
    C,
    /// Is F contiguous &mdash; last axis varies fastest. That is, is contiguous and has column-major ordering
    /// like in the Fortran programming language.
    F,
    /// Is both [F] and [C] contiguous. This means the array has at most one dimension with [size][Shape#getSize(int)]
    /// greater than one (e.g., at most [rank-1][Shape#rank()]) or is [squeezable][Shape#squeeze()] to rank-1.
    BOTH;


    /// Ensures that `order` is either [C] or [F] exactly (i.e., not [BOTH]).
    ///
    /// @param order The order of interest.
    /// @throws IllegalArgumentException If `order != C || order != F`.
    public static void ensureCorFExact(ContiguousOrder order) {
        if (order != C && order != F) {
            throw new IllegalArgumentException("Expecting order to be C or F but got " + order);
        }
    }
}