/*
 * MIT License
 *
 * Copyright (c) 2025-2026. Jacob Watters
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

package org.flag4jv3.util.arrays;

import org.flag4jv3.algebra.elements.Complex128;
import org.flag4jv3.util.ErrorMessages;
import org.flag4jv3.util.NewValidateParameters;
import org.flag4jv3.util.ValidateParameters;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Supplier;


/**
 * <p>The {@code ArrayBuilder} class provides a collection of static utility methods to construct, initialize,
 * and manipulate ndarrays in various ways. It is designed to simplify array handling tasks,
 * such as creating ndarrays with default values, filling ndarrays with specific values, and
 * generating ranges of elements.
 *
 * <p>This class supports multiple array types, including primitive types ({@code int[]}, {@code double[]}) as well as
 * {@code DoublePair} and {@code Complex64}.
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Ensure an array exists, if not create an array of size 10.
 * int[] myArray = ArrayBuilder.getOrCreateArray(null, 10);
 *
 * // Create a range of integers.
 * int[] range = ArrayBuilder.intRange(0, 10);
 *
 * // Fill an array with a specific value.
 * int[] filledArray = ArrayBuilder.filledArray(5, 2);
 *
 * // Perform a strided fill of zeros values.
 * double[] stridedArray = new double[10];
 * ArrayBuilder.stridedFillZeros(stridedArray, 2, 3);
 * }</pre>
 *
 * <h2>Restrictions:</h2>
 * <ul>
 *   <li>This class is <i>not</i> designed for jagged (non-rectangular) ndarrays. All methods dealing with multidimensional ndarrays
 *   assume, without an explicit check, that all ndarrays are rectangular.</li>
 * </ul>
 *
 * <p><strong>Note:</strong> This class is a utility class and cannot be instantiated.</p>
 */
public final class ArrayBuilder {

    private ArrayBuilder() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * In the case where the array is <em>not</em> {@code null}, a check will be made to ensure that the actual array
     * size matches the expected {@code size}.
     *
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * If {@code arr != null}, this will be used to verify the array is this size.
     * @return <ul>
     * <li>If {@code arr == null} then a new array with length {@code size} is created and returned.</li>
     * <li>If {@code arr != null} then a reference to {@code arr} is returned.</li>
     * </ul>
     *
     * @throws IllegalArgumentException If {@code arr != null && arr.length != size}.
     */
    public static boolean[] getOrCreateArray(boolean[] arr, int size) {
        boolean isNull = arr == null;

        if (!isNull && arr.length != size) {
            throw new IllegalArgumentException(
                    ErrorMessages.getArrayHasLengthErr(size, arr.length, null)
            );
        }

        return isNull ? new boolean[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * In the case where the array is <em>not</em> {@code null}, a check will be made to ensure that the actual array
     * size matches the expected {@code size}.
     *
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * If {@code arr != null}, this will be used to verify the array is this size.
     * @return <ul>
     * <li>If {@code arr == null} then a new array with length {@code size} is created and returned.</li>
     * <li>If {@code arr != null} then a reference to {@code arr} is returned.</li>
     * </ul>
     *
     * @throws IllegalArgumentException If {@code arr != null && arr.length != size}.
     */
    public static int[] getOrCreateArray(int[] arr, int size) {
        boolean isNull = arr == null;

        if (!isNull && arr.length != size) {
            throw new IllegalArgumentException(
                    ErrorMessages.getArrayHasLengthErr(size, arr.length, null)
            );
        }

        return isNull ? new int[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * In the case where the array is <em>not</em> {@code null}, a check will be made to ensure that the actual array
     * size matches the expected {@code size}.
     *
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * If {@code arr != null}, this will be used to verify the array is this size.
     * @return <ul>
     * <li>If {@code arr == null} then a new array with length {@code size} is created and returned.</li>
     * <li>If {@code arr != null} then a reference to {@code arr} is returned.</li>
     * </ul>
     *
     * @throws IllegalArgumentException If {@code arr != null && arr.length != size}.
     */
    public static long[] getOrCreateArray(long[] arr, int size) {
        boolean isNull = arr == null;

        if (!isNull && arr.length != size) {
            throw new IllegalArgumentException(
                    ErrorMessages.getArrayHasLengthErr(size, arr.length, null)
            );
        }

        return isNull ? new long[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * In the case where the array is <em>not</em> {@code null}, a check will be made to ensure that the actual array
     * size matches the expected {@code size}.
     *
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * If {@code arr != null}, this will be used to verify the array is this size.
     * @return <ul>
     * <li>If {@code arr == null} then a new array with length {@code size} is created and returned.</li>
     * <li>If {@code arr != null} then a reference to {@code arr} is returned.</li>
     * </ul>
     *
     * @throws IllegalArgumentException If {@code arr != null && arr.length != size}.
     */
    public static double[] getOrCreateArray(double[] arr, int size) {
        boolean isNull = arr == null;

        if (!isNull && arr.length != size) {
            throw new IllegalArgumentException(
                    ErrorMessages.getArrayHasLengthErr(size, arr.length, null)
            );
        }

        return isNull ? new double[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * In the case where the array is <em>not</em> {@code null}, a check will be made to ensure that the actual array
     * size matches the expected {@code size}.
     *
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * If {@code arr != null}, this will be used to verify the array is this size.
     * @return <ul>
     * <li>If {@code arr == null} then a new array with length {@code size} is created and returned.</li>
     * <li>If {@code arr != null} then a reference to {@code arr} is returned.</li>
     * </ul>
     *
     * @throws IllegalArgumentException If {@code arr != null && arr.length != size}.
     */
    public static float[] getOrCreateArray(float[] arr, int size) {
        boolean isNull = arr == null;

        if (!isNull && arr.length != size) {
            throw new IllegalArgumentException(
                    ErrorMessages.getArrayHasLengthErr(size, arr.length, null)
            );
        }

        return isNull ? new float[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array using the specified {@code initializer} if so.
     *
     * @param arr Array of interest.
     * @param initializer Supplier which constructs a new array of desired size in the case {@code arr==null}.
     * @return If {@code arr == null} then a new array is created by {@code initializer} and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static <T> T[] getOrCreateArray(T[] array, Supplier<T[]> initializer) {
        return array == null ? initializer.get() : array;
    }


    /**
     * Fills an array of complex elements with zeros.
     *
     * @param dest Array to fill with zeros.
     */
    public static void fillZeros(Complex128[][] dest) {
        for (Complex128[] row : dest)
            Arrays.fill(row, Complex128.ZERO);
    }


    /**
     * <p>
     * Fills an array with zeros separated by the given stride.
     *
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     *
     * @param dest Array to fill with strided zeros.
     * @param start Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(double[] dest, int start, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride);
        ValidateParameters.ensureGreaterEq(0, start);

        for (int i = start, stop = dest.length; i < stop; i += stride)
            dest[i] = 0;
    }


    /**
     * <p>
     * Fills an array with zeros separated by the given stride.
     *
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     *
     * @param dest Array to fill with strided zeros.
     * @param start Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(Complex128[] dest, int start, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride);
        ValidateParameters.ensureGreaterEq(0, start);

        for (int i = start; i < dest.length; i += stride)
            dest[i] = Complex128.ZERO;
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each separated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified n. Each range of zeros will be separated by
     * the stride.
     *
     *
     * <p>
     * If {@code stride=2}, {@code n=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     *
     * @param dest Array to fill with strided zeros.
     * @param start Starting point to apply strided zero fill.
     * @param n Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If {@code stride} or {@code n} is less than one.
     * @throws IllegalArgumentException If {@code start} is less than zero.
     */
    public static void stridedFillZeros(Complex128[] dest, int start, int n, int stride) {
        ValidateParameters.ensureAllGreaterEq(1, stride, n);
        ValidateParameters.ensureGreaterEq(0, start);
        int step = stride + n;

        for (int i = start; i < dest.length; i += step) {
            for (int j = i, jStop = n + i; j < jStop; j++)
                dest[i] = Complex128.ZERO;
        }
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each separated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified length. Each range of zeros will be separated by
     * the stride.
     *
     *
     * <p>
     * If {@code stride=2}, {@code length=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     *
     * @param dest Array to fill with strided zeros.
     * @param start Starting point to apply strided zero fill.
     * @param length Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride or length is less than one.
     * @throws IllegalArgumentException If start is less than zero.
     */
    public static void stridedFillZeros(double[] dest, int start, int length, int stride) {
        ValidateParameters.ensureAllGreaterEq(1, stride, length);
        ValidateParameters.ensureGreaterEq(0, start);
        int step = stride + length;

        for (int i = start, stop = dest.length; i < stop; i += step) {
            for (int j = i, jStop = length + i; j < jStop; j++)
                dest[i] = 0;
        }
    }


    /**
     * Fills an array with specified value.
     *
     * @param dest Array to fill.
     * @param fillValue Value to fill array with. This will be converted to a member of the field as if by
     * {@code dest[0].getZero().add(fillValue)}
     */
    public static void fill(Complex128[] dest, double fillValue) {
        Complex128 fillValueComplex = new Complex128(fillValue);
        Arrays.fill(dest, fillValueComplex);
    }


    /**
     * Fills an array with specified value.
     *
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static <T> void fill(T[][] dest, T fillValue) {
        for (T[] row : dest)
            Arrays.fill(row, fillValue);
    }


    /**
     * Fills range of an array with specified value.
     *
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     * @param from Staring slice of range (inclusive).
     * @param to Ending slice of range (exclusive).
     */
    public static void fill(Complex128[] dest, double fillValue, int from, int to) {
        ValidateParameters.ensureLessEq(to, from + 1);
        Complex128 complexFillValue = new Complex128(fillValue);
        Arrays.fill(dest, from, to, complexFillValue);
    }


    /**
     * Fills an array with the specified value;
     *
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(double[][] dest, double fillValue) {
        for (double[] doubles : dest)
            Arrays.fill(doubles, fillValue);
    }


    /**
     * Constructs an integer array filled with a specific value.
     *
     * @param size Size of the array.
     * @param value Value to set each slice of the array.
     * @return An array of specified {@code size} filled with the specified {@code value}.
     *
     * @throws NegativeArraySizeException If {@code} is negative.
     */
    public static int[] filledArray(int size, int value) {
        int[] dest = new int[size];
        Arrays.fill(dest, value);

        return dest;
    }


    /**
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive)
     *
     * @param start Staring value (inclusive).
     * @param end Stopping value (exclusive).
     * @return An array containing the integer range {@code [start, end)}.
     *
     * @throws IllegalArgumentException If {@code end < start}.
     */
    public static double[] range(int start, int end) {
        ValidateParameters.ensureGreaterEq(start, end);
        double[] rangeArr = new double[end - start];

        int rangeIdx = 0;
        for (int i = start; i < end; i++)
            rangeArr[rangeIdx++] = i;

        return rangeArr;
    }


    /// Creates an array filled with `int`'s from a range starting with `start` (inclusive) and going to `end` (exclusive)
    /// where each `int` is repeated {@code stride} times.
    ///
    /// <blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
    ///     <strong>Example:</strong>
    ///     {@snippet :
    ///     int[] range = intRange(5, 10);
    ///     System.out.println(Arrays.toString(range));}
    ///     Output: <code>[5, 6, 7, 8, 9]</code>
    /// </blockquote>
    ///
    /// @param start Starting slice (inclusive).
    /// @param end Stopping slice (exclusive). Must be greater than `start`.
    /// @param stride Number of times to repeat each integer in the range `[start, end)`. Must be positive.
    /// @return An array of length `end - start` containing exactly the range [start, end).
    ///
    /// @throws IllegalArgumentException If `start >= end`.
    /// @see #intRange(int, int, int)
    public static int[] intRange(int start, int end) {
        NewValidateParameters.ensureGreaterThan(end, start, null);
        int[] rangeArr = new int[end - start];

        int rangeIdx = 0;
        for (int i = start; i < end; i++)
            rangeArr[rangeIdx++] = i;

        return rangeArr;
    }


    /// Creates an array filled with `int`'s from a range starting with `start` (inclusive) and going to `end` (exclusive)
    /// where each `int` is repeated {@code stride} times.
    ///
    /// <blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
    ///     <strong>Example:</strong>
    /// {@snippet :
    /// int[] range = intRange(5, 8, 3);
    /// System.out.println(Arrays.toString(range));}
    ///     Output: <code>[5, 5, 5, 6, 6, 6, 7, 7, 7]</code>
    /// </blockquote>
    ///
    /// @param start Starting slice (inclusive).
    /// @param end Stopping slice (exclusive). Must be greater than `start`.
    /// @param stride Number of times to repeat each integer in the range `[start, end)`. Must be positive.
    /// @return An array containing
    ///
    /// @throws IllegalArgumentException If `stride` is non-positive.
    /// @throws IllegalArgumentException If `start >= end`.
    /// @see #intRange(int, int)
    public static int[] intRange(int start, int end, int stride) {
        NewValidateParameters.ensureGreaterThan(end, start, null);
        NewValidateParameters.ensureSign(stride, NewValidateParameters.Sign.POSITIVE, null);
        int[] rangeArr = new int[(end - start)*stride];

        int k = 0;
        for (int i = start; i < end; i++) {
            Arrays.fill(rangeArr, k, k + stride, i);
            k += stride;
        }

        return rangeArr;
    }


    /// Constructs a generic array of the specified length.
    ///
    /// @param template The reference template array. Must not be `null`.
    /// @param length The length of the array to construct.
    /// @param <T> The type of the elements of the array.
    /// @return A new array of the specified `length`.
    ///
    /// @throws NullPointerException If `template` is `null`.
    @SuppressWarnings("unchecked")
    public static <T> T[] newArrayLike(T[] template, int length) {
        Objects.requireNonNull(template, "The template array cannot be null.");
        return (T[]) Array.newInstance(template.getClass().getComponentType(), length);
    }
}
