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

package org.flag4j.util;

import org.flag4j.arrays.Shape;

/**
 * Utility class for nD array related utilities such as validation, etc.
 */
public final class ValidateNDArray {

    private ValidateNDArray() {
        // Hide constructor for utility class.
    }


    /**
     * Validates that a given buffer size is compatible with the specified shape for an nD array with contiguous
     * memory layout. This method assumes each element in the buffer is one complete element of the nD array.
     * @param shape The shape of the nD array.
     * @param bufferSize The size of the 1D data buffer.
     *
     * @see #ensureValidLayout(Shape, int, int)
     * @see #ensureValidLayout(Shape, int, int, int[])
     * @see #ensureValidLayout(Shape, int, int, int[], int)
     */
    public static void ensureValidLayout(Shape shape, int bufferSize) {
        ensureValidLayout(shape, bufferSize, 1);
    }


    /**
     * Validates that a given buffer size is compatible with the specified shape and element width for an nD array with contiguous
     * memory layout.
     *
     * @param shape The shape of the nD array.
     * @param bufferSize The size of the 1D data buffer.
     * @param elementWidth The width of each element in the buffer. If this is <code>2</code>, then one element of the nD array
     * would occupy two consecutive entries in the buffer.
     *
     * @see #ensureValidLayout(Shape, int)
     * @see #ensureValidLayout(Shape, int, int, int[])
     * @see #ensureValidLayout(Shape, int, int, int[], int)
     */
    public static void ensureValidLayout(Shape shape, int bufferSize, int elementWidth) {
        if (shape.numelIntValueExact() * elementWidth != bufferSize) {
            throw new IllegalArgumentException("" +
                    "Invalid buffer size for contiguous layout with shape " + shape + " and element width " + elementWidth
            );
        }
    }


    /**
     * Validates that a given buffer size is compatible with the specified shape and strides for an
     * nD array with a strided/offset memory layout. This method assumes each element
     * in the buffer is one complete element of the nD array.
     *
     * @param shape The shape of the nD array.
     * @param bufferSize The size of the 1D data buffer for the nD array.
     * @param offset The offset of the first element in the buffer.
     * @param strides The strides of the nD array.
     */
    public static void ensureValidLayout(
            Shape shape,
            int bufferSize,
            int offset,
            int[] strides
    ) {
        ensureValidLayout(shape, bufferSize, offset, strides, 1);
    }


    /**
     * Validates that a given buffer size is compatible with the specified shape, element width, and strides for an
     * nD array with a strided/offset memory layout.
     *
     * @param shape The shape of the nD array.
     * @param bufferSize The size of the 1D data buffer for the nD array.
     * @param offset The offset of the first element in the buffer.
     * @param strides The strides of the nD array.
     * @param elementWidth The width of each element in the buffer. If this is <code>2</code>, then one element of the nD array
     * would occupy two consecutive entries in the buffer.
     */
    public static void ensureValidLayout(
            Shape shape,
            int bufferSize,
            int offset,
            int[] strides,
            int elementWidth
    ) {
        var shapeRank = shape.getRank();
        var shapeDims = shape.getDims();

        if (bufferSize < 0) {
            throw new IllegalArgumentException("bufferSize must be non-negative");
        }
        if (elementWidth <= 0) {
            throw new IllegalArgumentException("elementWidth must be positive");
        }
        if (bufferSize % elementWidth != 0) {
            throw new IllegalArgumentException(
                    "Buffer size must be divisible by elementWidth"
            );
        }
        if (shapeRank != strides.length) {
            throw new IllegalArgumentException(
                    "shape and strides must have the same rank"
            );
        }

        boolean empty = false;
        for (int dim : shapeDims) {
            if (dim < 0) {
                throw new IllegalArgumentException("Negative shape dimension");
            }
            if (dim == 0) {
                empty = true;
            }
        }

        int logicalCapacity = bufferSize / elementWidth;

        if (empty) {
            if (offset < 0 || offset > logicalCapacity) {
                throw new IndexOutOfBoundsException("Empty-view offset out of bounds");
            }
            return;
        }

        long minElementIndex = offset;
        long maxElementIndex = offset;

        for (int d = 0; d < shapeRank; d++) {
            long extent = Math.multiplyExact(
                    (long) shapeDims[d] - 1,
                    (long) strides[d]
            );

            if (extent >= 0) {
                maxElementIndex = Math.addExact(maxElementIndex, extent);
            } else {
                minElementIndex = Math.addExact(minElementIndex, extent);
            }
        }

        if (minElementIndex < 0 || maxElementIndex >= logicalCapacity) {
            throw new IndexOutOfBoundsException(
                    "View accesses logical element range [" +
                            minElementIndex + ", " + maxElementIndex +
                            "] but logical buffer capacity is " + logicalCapacity
            );
        }
    }
}
