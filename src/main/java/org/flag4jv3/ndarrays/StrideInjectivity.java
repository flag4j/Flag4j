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

/// Utility class for checking if a [layout][Layout] is injective.
///
/// In this context, "injective" means that no two distinct nD indices point to the same element in the internal data buffer.
/// This can fail to be true in views: for example a view with stride(s) of size `0` would be non-injective.
public final class StrideInjectivity {

    private StrideInjectivity() {
        // Hide default constructor for utility class.
    }


    /// Stride injectivity classification.
    public enum Result {
        /// The layout is known to be injective.
        INJECTIVE,
        /// The layout is known to be non-overlapping.
        NON_INJECTIVE,
        /// It could not be determined if the layout is or is not injective.
        UNKNOWN
    }


    /// Simple record for storing axis information.
    ///
    /// @param maxDelta The maximum gap between strides.
    /// @param absStride The absolute value of a stride.
    private record Axis(int maxDelta, int absStride) {
    }


    /// Checks if a layout *might be* injective.
    ///
    /// <blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> This method does not perform an exhaustive check for all possible cases as that would have
    ///      worst-case exponential time. As such, this method <em>may</em> erroneously return {@code false} even if the
    ///      layout is truly injective. However, it will <em>never</em> mistakenly return {@code true}.
    /// </blockquote>
    ///
    /// To see the nature of the injectivity classification, use [#classifyInjectivity(Layout)].
    ///
    /// @param layout The layout of interest.
    /// @return `true` if it can be guaranteed that the layout is injective; otherwise, `false`.
    ///
    /// @see #classifyInjectivity(Layout)
    public static boolean mayBeInjective(Layout layout) {
        return classifyInjectivity(layout) == Result.INJECTIVE;
    }


    /// Classifies a `layout` to be [injective][Result#INJECTIVE], [non-injective][Result#NON_INJECTIVE], or
    /// [unknown][Result#UNKNOWN].
    ///
    /// <blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> This method does not perform an exhaustive check for all possible cases as that would
    ///     have worst-case exponential time. As such, this method may not be able to guarantee if the layout is or is not
    ///     injective. In such cases, {@link Result#UNKNOWN UNKNOWN} is returned. See return section for details.
    /// </blockquote>
    ///
    /// @param layout The layout of interest.
    /// @return - [INJECTIVE][Result#INJECTIVE] if it could be guaranteed the `layout` is injective.
    /// - [NON_INJECTIVE][Result#NON_INJECTIVE] if it could be guaranteed the `layout` is non-injective.
    /// - [UNKNOWN][Result#UNKNOWN] if it could not be guaranteed either way.
    public static Result classifyInjectivity(Layout layout) {
        int rank = layout.shape.rank;
        int[] dims = layout.shape.dims;

        for (int i = 0; i < rank; i++) {
            // Empty domain: injectivity holds vacuously.
            if (dims[i] == 0) return Result.INJECTIVE;
        }

        // Parallel arrays of (maxDelta, |stride|) for axes that can distinguish indices.
        int n = 0;
        int[] maxDelta = new int[rank];
        long[] absStride = new long[rank];

        for (int k = 0; k < rank; k++) {
            if (dims[k] <= 1) continue;             // Cannot distinguish two indices.
            if (layout.strides[k] == 0) return Result.NON_INJECTIVE;   // All indices alias.

            maxDelta[n] = dims[k] - 1;
            absStride[n] = Math.abs((long) layout.strides[k]);
            n++;
        }

        sortAxesByAbsStrideAsc(absStride, maxDelta, n);

        /*
         * Loop invariant: on entry to iteration i, `span` is the maximum displacement
         * magnitude reachable using only axes[0...i), which — by the ascending sort —
         * are exactly the axes with |stride| no greater than this one's.
         */
        long span = 0;
        boolean uncertain = false;

        for (int i = 0; i < n; i++) {
            long stride = absStride[i];

            // Witness: delta_i = 1, delta_j = -maxDelta[j] for all j < i. Sums to zero.
            if (stride == span) return Result.NON_INJECTIVE;

            // Cancellation reduces to subset-sum over the preceding axes; don't solve it.
            if (stride < span) uncertain = true;

            span += (long) maxDelta[i]*stride;
        }

        return uncertain ? Result.UNKNOWN : Result.INJECTIVE;
    }


    /// Computes absolute value on an integer without overflowing on [Integer#MIN_VALUE].
    ///
    /// Note that if `value` is [Integer#MIN_VALUE], then [Integer#MAX_VALUE] will be returned.
    ///
    /// @param a The argument to compute the absolute value of.
    /// @return The absolute value of `a`.
    private static int absWithMinSupport(int a) {
        return (a == Integer.MIN_VALUE) ? Integer.MAX_VALUE : Math.abs(a);
    }


    /// Sorts a
    /// @param absStride
    /// @param maxDelta
    /// @param n
    private static void sortAxesByAbsStrideAsc(long[] absStride, int[] maxDelta, int n) {
        for (int i = 1; i < n; i++) {
            long key = absStride[i];
            int delta = maxDelta[i];
            int j = i - 1;

            while (j >= 0 && absStride[j] > key) {
                absStride[j + 1] = absStride[j];
                maxDelta[j + 1] = maxDelta[j];
                j--;
            }

            absStride[j + 1] = key;
            maxDelta[j + 1] = delta;
        }
    }
}
