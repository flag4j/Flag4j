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

package org.flag4jv3.util.tuples;

import java.util.Arrays;
import java.util.Objects;
import java.util.StringJoiner;

/// Represents an immutible tuple of primitive `float`s.
///
/// @param items The elements of the tuple.
/// @see FloatPair
/// @see FloatTriple
public record FloatTuple(float... items) {

    public FloatTuple {
        Objects.requireNonNull(items, "items must not be null");
        items = items.clone();
    }


    @Override
    public float[] items() {
        return items.clone();
    }


    /**
     * Gets the size of the tuple.
     *
     * @return The size of this tuple.
     */
    public int size() {
        return items.length;
    }


    @Override
    public int hashCode() {
        return Arrays.hashCode(items);
    }


    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null) return false;
        if (obj.getClass() != getClass()) return false;

        return Arrays.equals(items, ((FloatTuple) obj).items);
    }


    @Override
    public String toString() {
        var joiner = new StringJoiner(", ", "(", ")");

        for (var d : items)
            joiner.add(String.valueOf(d));

        return "FloatTuple[items=" + joiner.toString() + "]";
    }
}
