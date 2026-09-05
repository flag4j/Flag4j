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

/// Contains static utility helper methods for multi-axes slicing.
final class SliceSupport {

    private SliceSupport() {
        // Hide default constructor for utility class.
    }


    /// Axis accounting for an index expression of an [layout][Layout].
    ///
    /// @param uncovered The number of source axes the multi-axis slice does not explicitly consume. These are
    /// filled by an [Slice#ELLIPSIS] if present or by implicit [Slice#ALL]'s at the tail end.
    /// @param outRank The rank of the layout the multi-axis slice produces.
    /// @see Layout#slice(org.flag4jv3.ndarrays.Slice...)
    record SliceArity(int uncovered, int outRank) {
    }


    /// Computes axis accounting for an index expression against the given `rank`.
    ///
    /// @param indexExpr The specification of the
    /// @param rank The rank of the layout being
    /// @throws IllegalArgumentException If `indexExpr` contains more than one [Slice#ELLIPSIS],
    /// or consumes more axes than the source has.
    static SliceArity arity(Slice[] indexExpr, int rank) {
        int consumed = 0, emitted = 0, ellipses = 0;

        for (Slice ix : indexExpr) {
            switch (ix) {
                case Slice.Point _ -> consumed++;              // consumes, emits nothing
                case Slice.AxisRange _ -> {
                    consumed++;
                    emitted++;
                }
                case Slice.Marker.ALL -> {
                    consumed++;
                    emitted++;
                }
                case Slice.Marker.NEW_AXIS -> emitted++;               // emits, consumes nothing
                case Slice.Marker.ELLIPSIS -> ellipses++;              // accounted for below
            }
        }

        if (ellipses > 1) {
            throw new IllegalArgumentException(
                    "At most one ELLIPSIS is permitted per slice indexExpr but found " + ellipses + ".");
        }

        int uncovered = rank - consumed;
        if (uncovered < 0) {
            throw new IllegalArgumentException(
                    "Slice indexExpr consumes " + consumed + " axes but rank is " + rank + ".");
        }

        return new SliceArity(uncovered, emitted + uncovered);
    }
}
