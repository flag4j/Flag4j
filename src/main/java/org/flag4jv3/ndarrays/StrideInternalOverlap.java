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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/// Utility class for checking if a [layout][Layout] may have internal overlapping elements.
/// In this context, "overlapping elements" means that two separate indices point to the same element in the internal data buffer.
/// This can happen in views: for example a view with stride(s) of size `0` would be overlapping. A layout with *no* overlap
/// could be called "injective".
public final class StrideInternalOverlap {

    private StrideInternalOverlap() {
        // Hide default constructor for utility class.
    }


    /// The stride overlapping result classification.
    public enum Result {
        /// The layout is known to be overlapping.
        OVERLAPPING,
        /// The layout is known to be non-overlapping.
        NON_OVERLAPPING,
        /// It could not be determined if the layout is or is not overlapping.
        UNKNOWN
    }


    /// Simple record for storing axis information.
    ///
    /// @param maxDelta The maximum gap between strides.
    /// @param absStride The absolute value of a stride.
    private record Axis(int maxDelta, int absStride) {
    }


    /// Checks if any element in an nD-array with the given layout *may* have values which overlap in memory (e.g., a zero stride).
    ///
    /// <blockquote style="color: #306091; background-color: #9da7c2; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> This method does not perform an exhaustive check for all possible overlaps as that would have
    ///      worst-case exponential time. As such, this method <em>may</em> erroneously return {@code true} even if there
    ///     is no overlap. However, it will <em>never</em> mistakenly return {@code false}.
    /// </blockquote>
    ///
    /// To see the nature of overlap classification, use [#classifyOverlap(Layout)].
    ///
    /// @param layout The layout of interest.
    /// @return `false` if it can be determined that *absolutely no* elements overlap in memory; otherwise, `true`.
    ///
    /// @see #classifyOverlap(Layout)
    public static boolean mayHaveOverlap(Layout layout) {
        return classifyOverlap(layout) != Result.NON_OVERLAPPING;
    }


    /// Checks if any element in an nD-array with the given layout *may* have values which overlap in memory (e.g., a zero stride).
    ///
    /// <blockquote style="color: #306091; background-color: #9da7c2; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> This method does not perform an exhaustive check for all possible overlaps as that would
    ///     have worst-case exponential time. As such, this method may not be able to guarantee if there is or is not overlapping.
    ///     In such cases, {@link Result#UNKNOWN} is returned. See return for details.
    /// </blockquote>
    ///
    /// @param layout The layout of interest.
    /// @return - [Result#OVERLAPPING] if it could be guaranteed there is overlapping in the `layout`.
    /// - [Result#NON_OVERLAPPING] if it could be guaranteed there is *no* overlapping in the `layout`.
    /// - [Result#UNKNOWN] if it could not be guaranteed either way.
    public static Result classifyOverlap(Layout layout) {
        return classifyOverlap(layout.shape, layout.strides);
    }


    /// Checks if any element in an nD-array with the given layout *may* have values which overlap in memory (e.g., a zero stride).
    ///
    /// <blockquote style="color: #306091; background-color: #9da7c2; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> This method does not perform an exhaustive check for all possible overlaps as that would
    ///     have worst-case exponential time. As such, this method may not be able to guarantee if there is or is not overlapping.
    ///     In such cases, {@link Result#UNKNOWN} is returned. See return for details.
    /// </blockquote>
    ///
    /// @param shape The shape of the nD-array.
    /// @param strides The strides of the nD-array.
    /// @return - [Result#OVERLAPPING] if it could be guaranteed there is overlapping in the `layout`.
    /// - [Result#NON_OVERLAPPING] if it could be guaranteed there is *no* overlapping in the `layout`.
    /// - [Result#UNKNOWN] if it could not be guaranteed either way.
    static Result classifyOverlap(Shape shape, int[] strides) {
        int rank = shape.rank;
        int[] dims = shape.dims;

        // An empty array has no logical elements, hence no internal overlap.
        for (int i = 0; i < rank; i++) {
            var size = dims[i];
            if (size < 0) {
                throw new IllegalArgumentException("negative dimension");
            }

            if (size == 0) {
                return Result.NON_OVERLAPPING;
            }
        }

        List<Axis> axes = new ArrayList<>();

        for (int k = 0; k < rank; k++) {
            if (dims[k] <= 1) {
                continue; // This axis cannot distinguish two indices.
            }

            int stride = absWithMinSupport(strides[k]);

            if (stride == 0) {
                return Result.OVERLAPPING;
            }

            axes.add(new Axis(dims[k] - 1, stride));
        }

        axes.sort(Comparator.comparingLong(Axis::absStride));

        /*
         * span is the maximum address displacement obtainable from all
         * previously processed axes.
         */
        int span = 0;
        boolean uncertain = false;

        for (Axis axis : axes) {
            int stride = axis.absStride();

            if (stride == span) {
                /*
                 * One step on this axis can be exactly canceled by taking
                 * the maximum displacement on all previous axes.
                 */
                return Result.OVERLAPPING;
            }

            if (stride < span) {
                // Cancellation may or may not be possible.
                uncertain = true;
            }

            span = Math.addExact(
                    span,
                    Math.multiplyExact(axis.maxDelta(), stride)
            );
        }

        return uncertain ? Result.UNKNOWN : Result.NON_OVERLAPPING;
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
}
