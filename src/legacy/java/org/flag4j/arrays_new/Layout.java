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

import java.util.Arrays;
import java.util.Objects;

// TODO NOW: Finish docs.
/// An instance of the `Layout` class
public record Layout(int offset, int[] strides) {

    public Layout {
        Objects.requireNonNull(strides, "strides cannot be null.");
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    /// Factory for a [Layout] of a contiguous nD array with the specified [shape][NewShape].
    /// @param shape The [shape][NewShape] of the nD array to get a contiguous [Layout] for.
    public static Layout contiguous(NewShape shape) {
        return new Layout(0, shape.getContiguousStridesUnsafe());
    }

    /**
     * Converts an nD index to the 1D index of the items buffer of an nD array with this layout.
     * @param idxND The ND index to convert.
     * @return The 1D index of a nD array's items buffer corresponding to the nD index {idxND}.
     */
    public int toBufferIndex(int... idxND) {
        int p = offset;
        for (int a = 0; a < idxND.length; a++) p += idxND[a] * strides[a];
        return p;
    }



    /// Checks if this layout defined a contiguous nD array with respect ot the given `shape`.
    /// @param shape The `shape` of interest.
    /// @return `true` if this layout is contiguous with respect to `shape`; `false` otherwise.
    public boolean isContiguous(NewShape shape) {
        return Arrays.equals(strides, shape.getContiguousStridesUnsafe());
    }


    @Override
    public boolean equals(Object obj) {
        return obj instanceof Layout other
                && offset == other.offset
                && Arrays.equals(strides, other.strides);
    }

    @Override
    public int hashCode() {
        return 31 * Integer.hashCode(offset) + Arrays.hashCode(strides);
    }
}
