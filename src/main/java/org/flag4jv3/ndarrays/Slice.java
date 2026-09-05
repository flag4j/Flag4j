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

/// Represents a slice of a [dense nD-array][org.flag4jv3.ndarrays.base.NDArrayBase].
/// A slice is a single-axis slice expression used to build views of an [nD-arrays][org.flag4jv3.ndarrays.base.NDArrayBase].
///
/// Slices are supplied as an *index expression*: an ordered sequence of [Slice] values read
/// left to right against the axes of the source. Applying an index expression *never* copies
/// data; it derives a new [Layout] over the same buffer.
///
/// ## Slice Variants
/// | Variant     | Source axes consumed | Out axes emitted | Effect                                                                        |
/// |-------------|----------------------|------------------|-------------------------------------------------------------------------------|
/// | [Point]     | &#x2714;             | &#x2718;         | Selects one position; the axis is dropped.                                    |
/// | [Range]     | &#x2714;             | &#x2714;         | Strided sub-range; the axis is kept, possibly smaller.                        |
/// | [#ALL]      | &#x2714;             | &#x2714;         | The whole axis; equivalent to [range(0, n, 1)][#range(Integer, Integer, int)].|
/// | [#NEW_AXIS] | &#x2718;             | &#x2714;         | Inserts a new axis of extent `1`.                                             |
/// | [#ELLIPSIS] | *k*                  | *k*              | Expands to as many [#ALL] as needed to cover remaining axes.                  |
///
/// ## Restrictions
/// - An slice expression may contain **at most one** [#ELLIPSIS].
/// - Axes not covered by the expression are implicitly [#ALL], so a trailing [#ELLIPSIS] is allowed but redundant.
/// - The number of axis-consuming entries must not exceed the rank of the source.
///
/// ## Bound resolution
/// For a [Range] applied to an axis of size `n` with step `s`:
///
/// - A `null` `start` resolves to `0` when `s > 0`, and to `n - 1` when `s < 0`.
/// - A `null` `stop` resolves to `n` when `s > 0`, and to the position before slice `0` when `s < 0`.
/// - A negative bound `b` resolves to `b + n` (counted from the end).
/// - Resolved bounds are clamped to the axis; an empty (extent-`0`) axis is a valid result.
/// - The resulting extent is `max(0, ceilDiv(stop - start, step))`.
///
/// <blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
///     <strong>Examples:</strong>
/// Given an {@link org.flag4jv3.ndarrays.base.NDArrayBase nD-array}, {@code a}, with {@link Shape shape} {@code (4, 5, 6)},
/// and the <a href="https://numpy.org/doc/stable/user/basics.indexing.html">NumPy</a> equivalent shown on the right:
/// {@snippet :
/// // Flag4j Expression                        // Out shape      NumPy Equivalent
/// a.slice(point(1));                          // (5, 6)         a[1]
/// a.slice(ALL, point(0));                     // (4, 6)         a[:, 0]
/// a.slice(range(1, 3));                       // (2, 5, 6)      a[1:3]
/// a.slice(ALL, range(0, 5, 2));               // (4, 3, 6)      a[:, 0:5:2]
/// a.slice(from(2));                           // (2, 5, 6)      a[2:]
/// a.slice(to(2));                             // (2, 5, 6)      a[:2]
/// a.slice(rev());                             // (4, 5, 6)      a[::-1]
/// a.slice(ELLIPSIS, point(0));                // (4, 5)         a[..., 0]
/// a.slice(NEW_AXIS);                          // (1, 4, 5, 6)   a[None]
/// a.slice(point(0), NEW_AXIS);                // (1, 5, 6)      a[0, None]
/// a.slice(range(3, 0, -1), ALL, point(2));    // (3, 5)         a[3:0:-1, :, 2]
/// a.slice(point(-1));                         // (5, 6)         a[-1]
/// a.slice(range(2, 2));                       // (0, 5, 6)      a[2:2]
/// a.slice(range(2, null, -1), ELLIPSIS)       // (4, 2, 5)      a[2::-1, ...]
///}
///
/// A more complicated slice expression on a 7D array, `a`, with shape `(4, 2, 5, 8, 2, 15, 7)` might look like:
/// {@snippet :
/// // Flag4j Expression                                                  // NumPy Equivalent
/// a.slice(range(2, null, -1), ELLIPSIS, NEW_AXIS, range(2, 11), rev())  // a[2::-1, ..., None, 2:11, ::-1]
///}
/// </blockquote>
///
/// @see Layout#slice(Slice...)
/// @see org.flag4jv3.ndarrays.base.NDArrayBase#slice(Slice...)
public sealed interface Slice {

    /// Selects a single position along an axis; the axis is dropped from the result.
    ///
    /// A negative `i` counts from the end of the axis. E.g.,
    ///
    /// @param i The position to select. Must resolve into the bounds of the target axis.
    record Point(int i) implements Slice {

        /// Resolves a point against a specific `axisSize`.
        ///
        /// @param axisSize The size of the axis top resolve against.
        /// @return The resolved point index.
        /// - If `this.i() > 0`, then `i()` is returned.
        /// - If `this.i() > 0`, then `i + axisSize` is returned.
        ///
        /// @throws IndexOutOfBoundsException If the resolved point index is negative or greater than or equal to `axisSize`.
        public int resolve(int axisSize) {
            int iResolved = i < 0 ? i + axisSize : i;

            if (iResolved < 0 || iResolved >= axisSize) {
                throw new IndexOutOfBoundsException(
                        "Point index " + iResolved + " is outside axis of size " + axisSize + ".");
            }

            return iResolved;
        }
    }


    /// A range of a single axis. Ranges can be resolved as a [ResolvedRange] so that they do not contain any `null` or negative
    /// bounds or strides.
    sealed interface AxisRange extends Slice permits Range, ResolvedRange {
        /// Resolves this axis range so that it may be directly used to slice a strided nD-array.
        /// A resolved range does *not* have any `null` or negative values.
        ///
        /// @param axisSize The size of the axis being sliced.
        /// @return The [resolved range][ResolvedRange].
        ///
        /// @throws IndexOutOfBoundsException If the range does not fit the axis.
        ResolvedRange resolve(int axisSize);
    }


    /// A strided range along an axis; the axis is retained with extent
    /// `max(0, Math.ceilDiv(stop - start, step))`.
    ///
    /// A `null` bound is *open* and is resolved against the size of the target axis in the
    /// direction implied by `step`. Negative bounds count from the end.
    ///
    /// @param start The start of the range (inclusive), or `null` for the first element in traversal order.
    /// @param stop The end of the range (exclusive), or `null` for the last element in traversal order.
    /// @param step The stride between selected positions. Must be non-zero; may be negative.
    /// @throws IllegalArgumentException If `step` is zero.
    record Range(Integer start, Integer stop, int step) implements AxisRange {
        public Range {
            if (step == 0) throw new IllegalArgumentException("step cannot be zero in a Range.");
        }


        /// Resolves this [range][Range] so that it may be directly used to slice a strided nD-array.
        /// This converts any `null` or negative `start` or `stop`'s with concrete positive values.
        ///
        /// @param axisSize The size of the axis being sliced.
        /// @return The [resolved range][ResolvedRange].
        @Override
        public ResolvedRange resolve(int axisSize) {
            // Check for an empty axis.
            if (axisSize == 0) return new ResolvedRange(0, step, 0);

            // Note: The `stop` of a range can exceed the axis size, so we clamp it.

            final int s0, s1;
            if (step > 0) {
                s0 = (start == null) ? 0 : norm(start, axisSize);
                s1 = (stop == null) ? axisSize : Math.clamp(norm(stop, axisSize), 0, axisSize);
            } else {
                s0 = (start == null) ? axisSize - 1 : norm(start, axisSize);
                s1 = (stop == null) ? -1 : Math.clamp(norm(stop, axisSize), -1, axisSize - 1);
            }

            if (s0 < 0 || s0 >= axisSize) {
                throw new IndexOutOfBoundsException(
                        "Range start resolved to " + s0 + " which is outside axis of size " + axisSize + ".");
            }

            int extent = Math.max(0, Math.ceilDiv(s1 - s0, step));

            return new ResolvedRange(s0, step, extent);
        }


        /// Normalizes an index so that it is non-negative.
        ///
        /// @param i The, possibly non-negative, index to normalize.
        /// @param axisSize The size of the axis being indexed.
        /// @return The normalized index of `i`.
        private static int norm(int i, int axisSize) {
            return i < 0 ? i + axisSize : i;
        }
    }


    /// The concrete positions a [Range] selects on an axis of known size.
    ///
    /// The selected positions are `start, start + step, ..., start + (extent - 1) * step`,
    /// all of which lie within the axis. `extent` supersedes the range's `stop`.
    ///
    /// @param start The start of the range (inclusive)
    /// @param step The stride between selected positions. Must be non-zero; may be negative.
    /// @param extent The extent of the strided range. Must be positive.
    record ResolvedRange(int start, int step, int extent) implements AxisRange {
        public ResolvedRange {
            if (step == 0) throw new IllegalArgumentException("step cannot be zero.");
            if (extent < 0) throw new IllegalArgumentException("extent cannot be negative but got " + extent + ".");
        }


        /// Resolves this axis range so that it may be directly used to slice a strided nD-array.
        /// A resolved range does *not* have any `null` or negative values.
        ///
        /// @param axisSize The size of the axis being sliced.
        /// @return The [resolved range][ResolvedRange].
        ///
        /// @throws IndexOutOfBoundsException If the range does not fit the axis.
        @Override
        public ResolvedRange resolve(int axisSize) {
            if (extent == 0) return this; // Empty slice.
            int last = start + (extent - 1)*step;
            if (start < 0 || start >= axisSize || last < 0 || last >= axisSize) {
                throw new IndexOutOfBoundsException(
                        "Resolved range [" + start + ", " + last + "] does not fit axis of size " + axisSize + ".");
            }

            return this;
        }
    }


    /// Slices carrying no parameters.
    ///
    /// Prefer the [#ALL], [#NEW_AXIS], and [#ELLIPSIS] constants on [Slice] at call sites;
    /// these are the same objects.
    enum Marker implements Slice {ALL, NEW_AXIS, ELLIPSIS}

    /// Selects an entire axis. Equivalent to [range(0, n, 1)][#range(Integer, Integer, int)] for an axis of size `n`.
    public static final Slice ALL = Marker.ALL;

    /// Inserts a new axis of extent `1` at this position. Does not consume source axis.
    public static final Slice NEW_AXIS = Marker.NEW_AXIS;

    /// Expands to as many [#ALL] entries as needed so that the index expression covers every source axis.
    /// At most, one [#ELLIPSIS] may appear in an index expression.
    public static final Slice ELLIPSIS = Marker.ELLIPSIS;

    /// Selects a single position along an axis; the axis is dropped from the result. That is, given
    /// an nD-array `a` with shape `(5, 3, 6)`, `a.slice(ALL, point(1), ALL)` will produce a slice of `a` with shape `(5, 6)`.
    ///
    /// @param i The position to select. Negative values count from the end.
    /// @return A slice representing a single position along an axis.
    public static Slice point(int i) {
        return new Point(i);
    }


    /// Gets a contiguous range `[start, stop)` with step `1`.
    ///
    /// @param start The start of the range (inclusive), or `null` for the first element in traversal order.
    /// @param stop The end of the range (exclusive), or `null` for the last element in traversal order.
    /// @return The range representing `[start, stop)`
    public static Slice range(Integer start, Integer stop) {
        return new Range(start, stop, 1);
    }


    /// Gets a strided range from `start` (inclusive) to `stop` (exclusive) with the specified `step` between elements of the
    /// range.
    ///
    /// @param start The start of the range (inclusive), or `null` for the first element in traversal order.
    /// @param stop The end of the range (exclusive), or `null` for the last element in traversal order.
    /// @param step The stride between selected positions. Must be non-zero; may be negative.
    /// @return A strided range range from `start` to `stop` with the specified `step` between elements of the range.
    ///
    /// @throws IllegalArgumentException If `step` is zero.
    public static Slice range(Integer start, Integer stop, int step) {
        return new Range(start, stop, step);
    }

    /// All positions `start` (inclusive) to the end of the axis.
    ///
    /// This is equivalent to both [range(start, null][#range(Integer, Integer)]
    ///  and [range(start, null, 1)][#range(Integer, Integer, int)].
    ///
    /// @param start The start of the range (inclusive).
    /// @return A slice representing the range `[start, n]` where `n` is the size of the axis being sliced.
    ///
    /// @see #to(int)
    public static Slice from(int start) {
        return new Range(start, null, 1);
    }

    /// All positions from the beginning of the axis to `stop`.
    ///
    /// This is equivalent to both [range(null, stop)][#range(Integer, Integer)]
    ///  and [range(null, stop, 1)][#range(Integer, Integer, int)].
    ///
    /// @param stop The end of the range (exclusive).
    /// @return A slice representing the range `[0, stop)`.
    ///
    /// @see #from(int)
    public static Slice to(int stop) {
        return new Range(null, stop, 1);
    }


    /// The entire axis in reverse order. This is equivalent to both [range(null, null, -1)][#range(Integer, Integer, int)]
    /// and [rev(null, null)][#rev(Integer, Integer)].
    ///
    /// @return The entire axis in reverse order.
    ///
    /// @see #rev(Integer, Integer)
    public static Slice rev() {
        return new Range(null, null, -1);
    }


    /// The range `[start, stop)` in reverse order. This is equivalent to
    /// [range(start, stop, -1)][#range(Integer, Integer, int)].
    ///
    /// For a strided reversed range, use [#range(Integer, Integer, int)] with a negative `step` whose absolute value
    /// is desired stride.
    ///
    /// @param start The start of the range (inclusive), or `null` for the first element in traversal order.
    /// @param stop The end of the range (exclusive), or `null` for the last element in traversal order.
    /// @return The range `[start, stop)` in reverse order.
    ///
    /// @see #rev()
    public static Slice rev(Integer start, Integer stop) {
        return new Range(start, stop, -1);
    }
}