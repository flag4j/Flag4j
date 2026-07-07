/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

import java.util.Arrays;


/**
 * The {@code ArrayJoiner} class provides utility methods for combining and repeating ndarrays.
 * It simplifies operations involving concatenating and repeating ndarrays.
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Join two ndarrays
 * int[] array1 = {1, 2, 3};
 * int[] array2 = {4, 5, 6};
 * int[] joinedArray = ArrayJoiner.join(array1, array2); // {1, 2, 3, 4, 5, 6}
 *
 * // Repeat an array
 * int[] repeatedArray = ArrayJoiner.repeat(3, array1); // {1, 2, 3, 1, 2, 3, 1, 2, 3}
 * }</pre>
 *
 * <p><strong>Note:</strong> This class is a utility class and cannot be instantiated.</p>
 */
public final class ArrayJoiner {

    // TODO: A lot of array stacking and extending could be moved into this class from org.linalg.ops...

    private ArrayJoiner() {
        // Hide default constructor for utility class.
    }


    /**
     * Joins two ndarrays together.
     *
     * @param src1 First array to join.
     * @param src2 Second array to join.
     * @return A single array of length {@code src1.length + src2.length} containing the elements of {@code src1}
     * followed by the elements of {@code src2}.
     */
    public static double[] join(double[] src1, double[] src2) {
        double[] concatenate = Arrays.copyOf(src1, src1.length + src2.length);
        System.arraycopy(src2, 0, concatenate, src1.length, src2.length);

        return concatenate;
    }


    /**
     * Joins two ndarrays together.
     *
     * @param src1 First array to join.
     * @param src2 Second array to join.
     * @return A single array of length {@code src1.length + src2.length} containing the elements of {@code src1}
     * followed by the elements of {@code src2}.
     */
    public static int[] join(int[] src1, int[] src2) {
        int[] concatenate = Arrays.copyOf(src1, src1.length + src2.length);
        System.arraycopy(src2, 0, concatenate, src1.length, src2.length);
        return concatenate;
    }


    /**
     * Joins two ndarrays together. The elements in the two ndarrays will be shallowly copied.
     *
     * @param src1 First array to join.
     * @param src2 Second array to join.
     * @return A single array of length {@code src1.length + src2.length} containing the elements of {@code src1}
     * followed by the elements of {@code src2}.
     */
    public static <T> T[] join(T[] src1, T[] src2) {
        T[] concatenate = Arrays.copyOf(src1, src1.length + src2.length);
        System.arraycopy(src2, 0, concatenate, src1.length, src2.length);
        return concatenate;
    }


    /**
     * Repeats an array a specified number of times.
     *
     * @param numTimes Number of times to repeat the array.
     * @param src      The source array to repeat.
     * @return The {@code src} array repeated {@code numTimes times}.
     */
    public static int[] repeat(int numTimes, int[] src) {
        int[] repeated = new int[src.length * numTimes];

        for(int i = 0, size=repeated.length, step=src.length; i < size; i += step)
            System.arraycopy(src, 0, repeated, i, src.length);

        return repeated;
    }
}
