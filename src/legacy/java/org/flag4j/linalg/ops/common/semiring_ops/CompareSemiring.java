/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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

package org.flag4j.linalg.ops.common.semiring_ops;

import org.flag4j.numbers.FieldElement;
import org.flag4j.numbers.SemiringElement;

/**
 * This utility class provides methods useful for comparing elements of a {@link SemiringElement}.
 */
public final class CompareSemiring {

    private CompareSemiring() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Computes the maximum value in the specified array. This is done according to the ordering imposed by
     * {@link FieldElement#compareTo(FieldElement)}.
     * @param values Values to commute maximum of.
     * @return The maximum value in {@code values}. If {@code values.length} equals zero, then {@code null} is returned.
     */
    public static <T extends SemiringElement<T>> T max(T... values) {
        if(values.length == 0) return null;
        T max = values[0];

        for (int i = 1, size=values.length; i<size; i++)
            if (max.compareTo(values[i]) < 0) max = values[i];

        return max;
    }



    /**
     * Computes the minimum value in the specified array. This is done according to the ordering imposed by
     * {@link FieldElement#compareTo(FieldElement)}.
     * @param values Values to commute minimum of.
     * @return The minimum value in {@code values}. If {@code values.length} equals zero, then {@code null} is returned.
     */
    public static <T extends SemiringElement<T>> T min(T... values) {
        if(values.length == 0) return null;
        T min = values[0];

        for (int i=1, length=values.length; i < length; i++)
            if(values[i].compareTo(min) < 0) min = values[i];

        return min;
    }


    /**
     * Computes the index of the maximum value in the specified array. This is done according to the ordering imposed by
     * {@link SemiringElement#compareTo(SemiringElement)}.
     * @param values Values for which compute index of maximum value.
     * @return The index of the maximum value in {@code values}. If the maximum value occurs more than once, the index of the first
     * occurrence is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends SemiringElement<T>> int argmax(T... values) {
        if(values.length == 0) return -1;
        T max = values[0];
        int maxdex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            if(max.compareTo(values[i]) < 0) {
                max = values[i];
                maxdex = i;
            }
        }

        return maxdex;
    }


    /**
     * Computes the index of the minimum value in the specified array. This is done according to the ordering imposed by
     * {@link SemiringElement#compareTo(SemiringElement)}.
     * @param values Values for which compute index of the minimum value.
     * @return The index of the minimum value in {@code values}. If the minimum value occurs more than once, the index of the first
     * occurrence is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends SemiringElement<T>> int argmin(T... values) {
        if(values.length == 0) return -1;
        T min = values[0];
        int mindex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            if(values[i].compareTo(min) < 0) {
                min = values[i];
                mindex = i;
            }
        }

        return mindex;
    }
}
