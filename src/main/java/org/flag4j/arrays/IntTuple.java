/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.arrays;

import java.util.Arrays;
import java.util.StringJoiner;

/**
 * <p>Data record to store an ordered list of integers (i.e., an n-tuple).
 * <p>IntTuples are immutable.
 *
 * @param data The values of the integer tuple.
 * @see IntPair
 * @see Tuple
 * @see Pair
 * @see Triple
 */
public record IntTuple(int[] data) {

    /**
     * Gets the size of the tuple.
     * @return The size of this tuple.
     */
    public int size() {
        return data.length;
    }


    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }


    @Override
    public boolean equals(Object obj) {
        if(obj == null) return false;
        if(obj.getClass() != getClass()) return false;

        return Arrays.equals(data, ((IntTuple)obj).data);
    }


    @Override
    public String toString() {
        StringJoiner joiner = new StringJoiner(", ", "(", ")");

        for(int d : data)
            joiner.add(String.valueOf(d));

        return "Tuple[data=" + joiner.toString() + "]";
    }
}
