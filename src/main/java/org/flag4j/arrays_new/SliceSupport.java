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

package org.flag4j.arrays_new;

import org.flag4j.arrays.Shape;

/**
 * Shared implementation details for strided slices.
 */
final class SliceSupport {

    private SliceSupport() {
        // Utility class.
    }


    static void validateAxes(int[] starts, int[] ends, int[] strides) {
        if (starts.length != ends.length || starts.length != strides.length) {
            throw new IllegalArgumentException(
                    String.format(
                            "starts, ends, and strides must have the same length but got lengths %d, %d, and %d.",
                            starts.length,
                            ends.length,
                            strides.length
                    )
            );
        }

        for (int axis = 0; axis < starts.length; axis++) {
            validateAxis(starts[axis], ends[axis], strides[axis], axis);
        }
    }


    static void validateAxis(int start, int end, int stride, int axis) {
        if (start < 0) {
            throw new IllegalArgumentException(
                    String.format(
                            "Start must be non-negative but got start=%d for axis %d.",
                            start,
                            axis
                    )
            );
        }

        if (stride == 0) {
            throw new IllegalArgumentException(
                    String.format("Stride must be non-zero for axis %d.", axis)
            );
        }

        if (stride > 0) {
            if (end < 0) {
                throw new IllegalArgumentException(
                        String.format(
                                "With positive stride %d, end must be non-negative but got end=%d for axis %d.",
                                stride,
                                end,
                                axis
                        )
                );
            }

            if (start > end) {
                throw new IllegalArgumentException(
                        String.format(
                                "With positive stride %d, start must be less than or equal to end "
                                        + "but got start=%d, end=%d for axis %d.",
                                stride,
                                start,
                                end,
                                axis
                        )
                );
            }
        } else {
            if (end < NDArraySlice.REVERSE_END) {
                throw new IllegalArgumentException(
                        String.format(
                                "With negative stride %d, end must be at least %d but got end=%d for axis %d.",
                                stride,
                                NDArraySlice.REVERSE_END,
                                end,
                                axis
                        )
                );
            }

            if (start < end) {
                throw new IllegalArgumentException(
                        String.format(
                                "With negative stride %d, start must be greater than or equal to end "
                                        + "but got start=%d, end=%d for axis %d.",
                                stride,
                                start,
                                end,
                                axis
                        )
                );
            }
        }
    }


    static boolean canSliceAxis(int start, int end, int stride, int axisSize) {
        if (axisSize < 0 || start < 0 || stride == 0) {
            return false;
        }

        if (stride > 0) {
            return end >= 0
                    && start <= end
                    && end <= axisSize;
        }

        return end >= NDArraySlice.REVERSE_END
                && start >= end
                && (start == end
                ? start <= axisSize
                : start < axisSize);
    }


    static int sliceLength(int start, int end, int stride) {
        long distance = stride > 0
                ? (long) end - start
                : (long) start - end;

        long step = Math.abs((long) stride);

        return Math.toIntExact((distance + step - 1L) / step);
    }


    static Shape resultShape(int[] starts, int[] ends, int[] strides) {
        int[] resultDims = new int[starts.length];

        for (int axis = 0; axis < starts.length; axis++) {
            resultDims[axis] = sliceLength(
                    starts[axis],
                    ends[axis],
                    strides[axis]
            );
        }

        return new Shape(resultDims);
    }


    /**
     * Creates an IndexOutOfBoundsException for an invalid axis.
     * @param axis The invalid axis.
     * @param rank The rank of the slice.
     * @return An IndexOutOfBoundsException for an invalid axis.
     */
    static IndexOutOfBoundsException invalidAxis(int axis, int rank) {
        return new IndexOutOfBoundsException(
                String.format(
                        "Slice rank is %d, so valid axis indices are in [0, %d), but got %d.",
                        rank,
                        rank,
                        axis
                )
        );
    }


    /**
     * Validate the given axis.
     * @param axis The axis to validate.
     * @param rank The rank of the slice.
     * @throws IllegalArgumentException If {@code axis} is negative or greater than {@code rank}.
     */
    static void validateAxis(int axis, int rank) {
        if (axis < 0 || axis >= rank) {
            throw SliceSupport.invalidAxis(axis, rank);
        }
    }
}
