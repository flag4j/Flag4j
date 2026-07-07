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

package org.flag4j.linalg.ops.common.ring_ops;

import org.flag4j.numbers.RingElement;


/**
 * This utility class provides methods useful for comparing elements of a {@link RingElement}.
 */
public final class CompareRing {

    private CompareRing() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Computes the maximum absolute value in the specified array. This is done according to the ordering imposed by
     * {@link Double#compareTo(Double) Double.compareTo(x.mag(), y.mag())} where x and y are elements of {@code values}.
     * @param values Values to commute maximum of.
     * @return The maximum value in {@code values}. If {@code values.length} equals zero, then {@link Double#NaN} is returned.
     */
    public static <T extends RingElement<T>> double maxAbs(T... values) {
        if(values.length == 0) return Double.NaN;
        double max = values[0].mag();

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(Double.compare(max, mag) < 0) max = mag;
        }

        return max;
    }


    /**
     * Computes the minimum absolute value in the specified array. This is done according to the ordering imposed by
     * {@link Double#compareTo(Double) Double.compareTo(x.mag(), y.mag())} where x and y are elements of {@code values}.
     * @param values Values to compute minimum of.
     * @return The minimum value in {@code values}. If {@code values.length} equals zero, then {@link Double#NaN} is returned.
     */
    public static <T extends RingElement<T>> double minAbs(T... values) {
        if(values.length == 0) return Double.NaN;
        double min = values[0].mag();

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(Double.compare(mag, min) < 0) min = mag;
        }

        return min;
    }


    /**
     * Computes the index of the maximum absolute value in the specified array.
     * @param values Values for which compute index of maximum absolute value.
     * @return The index of the maximum absolute value in {@code values}. If the maximum absolute value occurs more than once, the
     * index of the first occurrence is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends RingElement<T>> int argmaxAbs(T... values) {
        if(values.length == 0) return -1;
        double max = values[0].mag();
        int maxdex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(max < mag) {
                max = mag;
                maxdex = i;
            }
        }

        return maxdex;
    }


    /**
     * Computes the index of the minimum absolute value in the specified array.
     * @param values Values for which compute index of the minimum absolute value.
     * @return The index of the minimum absolute value in {@code values}. If the minimum absolute value occurs more than once,
     * the index of the first occurrence is returned. If {@code values.length} equals zero, then {@code -1} is returned.
     */
    public static <T extends RingElement<T>> int argminAbs(T... values) {
        if(values.length == 0) return -1;
        double min = values[0].mag();
        int mindex = 0;

        for (int i=1, length=values.length; i < length; i++) {
            double mag = values[i].mag();
            if(mag < min) {
                min = mag;
                mindex = i;
            }
        }

        return mindex;
    }


    /**
     * <p>Returns the maximum absolute value among {@code n} elements in the array {@code src},
     * starting at index {@code start} and advancing by {@code stride} for each subsequent element.
     *
     * <p>More formally, this method examines the elements at indices:
     * {@code start}, {@code start + stride}, {@code start + 2*stride}, ..., {@code start + (n-1)*stride}.
     *
     * <p>This method may be used to find the maximum absolute value within the row or column of a
     * {@link org.flag4j.arrays.dense.RingMatrix matrix} {@code a} as follows:
     * <ul>
     *     <li>Maximum absolute value within row {@code i}:
     *     <pre>{@code maxAbs(a.items, i*a.numCols, a.numCols, 1);}</pre></li>
     *     <li>Maximum absolute value within column {@code j}:
     *     <pre>{@code maxAbs(a.items, j, a.numRows, a.numRows);}</pre></li>
     * </ul>
     *
     * @param src The array to search for maximum absolute value within.
     * @param start The starting index in {@code src} to search.
     * @param n The number of elements to consider within {@code src1}.
     * @param stride The gap (in indices) between consecutive elements to search within {@code src}.
     * @return The maximum absolute value found among all elements considered in {@code src}.</li>
     * </ul>
     *
     * @throws IndexOutOfBoundsException If the specified range extends beyond the array bounds.
     */
    public static <T extends RingElement<T>> double maxAbs(T[] src, final int start, final int n, final int stride) {
        double currMax = 0;
        final int end = start + n*stride;

        for(int i=start; i<end; i+=stride)
            currMax = Math.max(src[i].abs(), currMax);

        return currMax;
    }


    /**
     * <p>Returns the minimum absolute value among {@code n} elements in the array {@code src},
     * starting at index {@code start} and advancing by {@code stride} for each subsequent element.
     *
     * <p>More formally, this method examines the elements at indices:
     * {@code start}, {@code start + stride}, {@code start + 2*stride}, ..., {@code start + (n-1)*stride}.
     *
     * <p>This method may be used to find the minimum absolute value within the row or column of a
     * {@link org.flag4j.arrays.dense.RingMatrix matrix} {@code a} as follows:
     * <ul>
     *     <li>Minimum absolute value within row {@code i}:
     *     <pre>{@code maxAbs(a.items, i*a.numCols, a.numCols, 1);}</pre></li>
     *     <li>Minimum absolute value within column {@code j}:
     *     <pre>{@code maxAbs(a.items, j, a.numRows, a.numRows);}</pre></li>
     * </ul>
     *
     * @param src The array to search for Minimum absolute value within.
     * @param start The starting index in {@code src} to search.
     * @param n The number of elements to consider within {@code src1}.
     * @param stride The gap (in indices) between consecutive elements to search within {@code src}.
     * @return
     * <ul>
     *     <li>If {@code src.length  == 0} then {@link Double#POSITIVE_INFINITY} will be returned.</li>
     *     <li>Otherwise, the minimum absolute value found among all elements considered inn{@code src}.</li>
     * </ul>
     *
     * @throws IndexOutOfBoundsException If the specified range extends beyond the array bounds.
     */
    public static <T extends RingElement<T>> double minAbs(T[] src, final int start, final int n, final int stride) {
        double currMin = Double.POSITIVE_INFINITY;
        final int end = start + n*stride;

        for(int i=start; i<end; i+=stride)
            currMin = Math.min(src[i].abs(), currMin);

        return currMin;
    }
}
