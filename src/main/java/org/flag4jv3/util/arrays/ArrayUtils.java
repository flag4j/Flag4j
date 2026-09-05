/*
 * MIT License
 *
 * Copyright (c) 2022-2026. Jacob Watters
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
import org.flag4jv3.algebra.elements.Complex64;
import org.flag4jv3.algebra.elements.FieldElement;
import org.flag4jv3.algebra.elements.SemiringElement;
import org.flag4jv3.util.NewValidateParameters;
import org.flag4jv3.util.ValidateParameters;
import org.flag4jv3.util.tuples.Pair;

import java.lang.reflect.Array;
import java.util.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * <p>The {@code ArrayUtils} class provides a set of utility methods for performing operations on ndarrays.
 * This includes transformations, comparisons, cumulative operations, validations, and copying.
 *
 * <p>For other array operations see: {@link ArrayBuilder}, {@link ArrayConversions}, and {@link ArrayJoiner}.
 *
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Deep comparison and copying of multidimensional ndarrays.</li>
 *   <li>Element-wise operations, including transformations, swaps, and permutations.</li>
 *   <li>Efficient searching and validation, including methods for finding unique values, indices, and membership checks.</li>
 *   <li>Flattening and reshaping of multidimensional ndarrays.</li>
 * </ul>
 *
 * <h2>Usage Examples</h2>
 * <pre>{@code
 * // Compute the cumulative sum of an integer array
 * int[] array = {1, 2, 3, 4};
 * int[] cumSumArray = ArrayUtils.cumSum(array, null);
 *
 * // Check if two 2D ndarrays are element-wise equal
 * int[][] array1 = {{1, 2}, {3, 4}};
 * int[][] array2 = {{1, 2}, {3, 4}};
 * boolean isEqual = ArrayUtils.deepEquals2D(array1, array2);
 *
 * // Swap elements in an array
 * int[] elements = {1, 2, 3};
 * ArrayUtils.swap(elements, 0, 2); // Result: {3, 2, 1}
 *
 * // Find unique values in an array
 * int[] values = {1, 2, 2, 3};
 * int[] uniqueValues = ArrayUtils.uniqueSorted(values); // Result: {1, 2, 3}
 *
 * // Apply a transformation to a numeric array
 * double[] items = {1.0, 2.0, 3.0};
 * ArrayUtils.applyTransform(items, x -> x * x); // Squares each value in the array
 * }</pre>
 *
 * <h2>Restrictions</h2>
 * <ul>
 *   <li>Multidimensional ndarrays are expected to be rectangular for all methods in this class.
 *   However, this is not explicitly enforced and jagged ndarrays may cause unexpected behavior.</li>
 *   <li>Many methods assume that input ndarrays are non-null unless explicitly stated otherwise.</li>
 *   <li>Sorting is required but not enforced for some methods to function correctly, such as {@link #contains(int[], int)} and
 *   {@link #findFirstLast(int[], int)}. Passing non-sorted ndarrays to such a method results in undefined behavior.</li>
 * </ul>
 *
 * <p><strong>Note:</strong> This class is a utility class and cannot be instantiated.
 *
 * @see ArrayBuilder
 * @see ArrayConversions
 * @see ArrayJoiner
 */
public final class ArrayUtils {

    private ArrayUtils() {
        // Hide constructor for utility class.
    }


    /**
     * Computes the cumulative sum of the elements in an array.
     *
     * @param src Source array to compute the cumulative sum of.
     * @param dest Array to store the result of the cumulative sum in. This may be the same array as {@code src} or {@code null}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * the appropriate size will be constructed and returned.
     *
     * @throws IllegalArgumentException If {@code dest != null && dest.length != src.length}.
     * @throws ArithmeticException      If at any point, the cumulative sum overflows an int.
     */
    public static int[] cumSum(int[] src, int[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, src.length);

        if (src.length > 0) dest[0] = src[0];
        for (int i = 1, size = src.length; i < size; i++) {
            dest[i] = Math.addExact(dest[i - 1], src[i]);
        }

        return dest;
    }


    /**
     * Computes the cumulative sum of the elements in an array. Same as {@link #cumSum(int[], int[])} but
     * <em>does not</em> provide checks for integer overflow.
     *
     * @param src Source array to compute the cumulative sum of.
     * @param dest Array to store the result of the cumulative sum in. This may be the same array as `src` or `null`.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. If {@code dest == null` then a new array of
     * the appropriate size will be constructed and returned.
     *
     * @throws IllegalArgumentException If {@code dest != null && dest.length != src.length}.
     * @throws ArithmeticException      If at any point, the cumulative sum overflows an int.
     * @see #cumSum(int[], int[])
     */
    public static int[] cumSumUnchecked(int[] src, int[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, src.length);

        if (src.length > 0) dest[0] = src[0];
        for (int i = 1, size = src.length; i < size; i++) {
            dest[i] = dest[i - 1] + src[i];
        }

        return dest;
    }


    /**
     * Computes the cumulative product of the elements in an array.
     *
     * @param src Source array to compute the cumulative product of.
     * @param dest Array to store the result of the cumulative product in. This may be the same array as {@code src} or {@code null}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * the appropriate size will be constructed and returned.
     *
     * @throws IllegalArgumentException If {@code dest != null && dest.length != src.length}.
     * @throws ArithmeticException      If at any point, the cumulative product overflows an int.
     */
    public static int[] cumProd(int[] src, int[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, src.length);

        if (src.length > 0) dest[0] = src[0];
        for (int i = 1, size = src.length; i < size; i++) {
            dest[i] = Math.multiplyExact(dest[i - 1], src[i]);
        }

        return dest;
    }


    /**
     * Computes the cumulative product of the elements in an array. Same as {@link #cumProd(int[], int[])} but
     * <em>does not</em> provide checks for integer overflow.
     *
     * @param src Source array to compute the cumulative product of.
     * @param dest Array to store the result of the cumulative product in. This may be the same array as `src` or `null`.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. If {@code dest == null` then a new array of
     * the appropriate size will be constructed and returned.
     *
     * @throws IllegalArgumentException If {@code dest != null && dest.length != src.length}.
     * @throws ArithmeticException      If at any point, the cumulative product overflows an int.
     * @see #cumproduct(int[], int[])
     */
    public static int[] cumProdUnchecked(int[] src, int[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, src.length);

        if (src.length > 0) dest[0] = src[0];
        for (int i = 1, size = src.length; i < size; i++) {
            dest[i] = dest[i - 1]*src[i];
        }

        return dest;
    }


    /**
     * Checks if two primitive 2D integer ndarrays are element-wise equal.
     *
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return The
     */
    public static boolean deepEquals2D(int[][] src1, int[][] src2) {
        if (src1 == src2) return true;
        if (src1 == null || src2 == null) return false;
        if (src1.length != src2.length) return false;

        for (int i = 0, size = src1.length; i < size; i++)
            if (!Arrays.equals(src1[i], src2[i])) return false;

        return true;
    }


    /**
     * Checks if a double array is numerically equal to a {@link Complex128 complex number} array.
     *
     * @param src1 Double array.
     * @param src2 Complex number array.
     * @return {@code true} if all items in {@code src2} have zero imaginary parts and real parts equal to the
     * corresponding entry in {@code src1}; {@code false} otherwise.
     */
    public static boolean equals(double[] src1, Complex128[] src2) {
        boolean equal = true;

        if (src1.length != src2.length) {
            equal = false;
        } else {
            for (int i = 0, size = src1.length; i < size; i++) {
                if (src1[i] != src2[i].re() || src2[i].im() != 0) {
                    equal = false;
                    break; // No need to continue.
                }
            }
        }

        return equal;
    }


    /**
     * Creates a deep copy of a 2D array. Assumes ndarrays are <em>not</em> jagged.
     *
     * @param src Source array to copy.
     * @param dest Destination array of copy. If {@code null}, a new array will be initialized.
     * @return A reference to {@code dest} if it was not {@code null}. In the case where {@code dest} is {@code null}, then a new
     * array will be initialized and returned.
     *
     * @throws IllegalArgumentException If the two ndarrays do <em>not</em> have the same shape.
     */
    public static int[][] deepCopy2D(int[][] src, int[][] dest) {
        if (dest == null) dest = new int[src.length][src[0].length];
        if (src == dest) {
            return dest;
        } else {
            ValidateParameters.ensureGreaterEq(src.length, dest.length);
        }

        for (int i = 0, size = src.length; i < size; i++) {
            dest[i] = new int[src[i].length];
            System.arraycopy(src[i], 0, dest[i], 0, src[i].length);
        }

        return dest;
    }


    /**
     * Performs an array copy similar to {@link System#arraycopy(Object, int, Object, int, int)} but wraps doubles as complex elements.
     *
     * @param src The source array.
     * @param srcPos The starting position from which to copy elements of the source array.
     * @param dest The destination array for the copy.
     * @param destPos Starting slice to place copied elements in the destination array.
     * @param length The number of array elements to be copied.
     * @throws ArrayIndexOutOfBoundsException If the destPos parameter plus the length parameter exceeds the length of the
     *                                        source array length or the destination array length.
     */
    public static void arraycopy(double[] src, int srcPos, Complex128[] dest, int destPos, int length) {
        for (int i = 0; i < length; i++)
            dest[i + destPos] = new Complex128(src[i + srcPos]);
    }


    /**
     * Performs an array copy similar to {@link System#arraycopy(Object, int, Object, int, int)} but wraps floats as complex elements.
     *
     * @param src The source array.
     * @param srcPos The starting position from which to copy elements of the source array.
     * @param dest The destination array for the copy.
     * @param destPos Starting slice to place copied elements in the destination array.
     * @param length The number of array elements to be copied.
     * @throws ArrayIndexOutOfBoundsException If the destPos parameter plus the length parameter exceeds the length of the
     *                                        source array length or the destination array length.
     */
    public static void arraycopy(float[] src, int srcPos, Complex64[] dest, int destPos, int length) {
        for (int i = 0; i < length; i++)
            dest[i + destPos] = new Complex64(src[i + srcPos]);
    }


    /**
     * Swaps elements in an array according to a specified permutation.
     *
     * @param src Array to swap elements within.
     * @param indices Array containing indices of the permutation. If the {@code src} array has length {@code N}, then
     * the array must be a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If {@code indices} is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void permute(int[] src, int[] indices) {
        ValidateParameters.ensurePermutation(indices);

        int[] swapped = new int[src.length];
        int i = 0;

        for (int value : indices)
            swapped[i++] = src[value];

        System.arraycopy(swapped, 0, src, 0, swapped.length);
    }


    /**
     * Swaps elements in an array according to a specified permutation. This method should be used with extreme caution as unlike
     * {@link #permute(int[], int[])}, this method does <em>not</em> verify that {@code indices} is a permutation.
     *
     * @param src Array to swap elements within.
     * @param indices Array containing indices of the permutation. If the {@code src} array has length {@code N}, then
     * the array must be a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void permuteUnsafe(int[] src, int[] indices) {
        int[] swapped = new int[src.length];
        int i = 0;

        for (int value : indices)
            swapped[i++] = src[value];

        System.arraycopy(swapped, 0, src, 0, swapped.length);
    }


    /**
     * Swaps to elements in an array. This is done in-place.
     *
     * @param arr Array to swap elements in. This array is modified.
     * @param i Index of the first value to swap.
     * @param j Index of the second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Swaps to elements in an array. This is done in-place.
     *
     * @param arr Array to swap elements in. This array is modified.
     * @param i Index of the first value to swap.
     * @param j Index of the second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(double[] arr, int i, int j) {
        double temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Swaps to elements in an array. This is done in-place.
     *
     * @param arr Array to swap elements in. This array is modified.
     * @param i Index of the first value to swap.
     * @param j Index of the second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(Object[] arr, int i, int j) {
        Object temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Checks if a key is in an array.
     *
     * @param src Source array. Must be sorted, if not, call {@link Arrays#sort(double[])} first. Otherwise, the
     * behavior of this method is undefined.
     * @param key Values to check if they are in the source array.
     * @return A boolean describing if the specified key is in the array or not.
     */
    public static boolean notContains(int[] src, int key) {
        return !contains(src, key);
    }


    /**
     * Checks if a set of keys are in an array.
     *
     * @param src Source array. Must be sorted, if not, call {@link Arrays#sort(double[])} first. Otherwise, the
     * behavior of this method is undefined.
     * @param keys Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code keys} describing if the associated keys are in the
     * array.
     */
    public static boolean[] contains(double[] src, double... keys) {
        boolean[] result = new boolean[keys.length];

        for (int i = 0, size = keys.length; i < size; i++) {
            result[i] = contains(src, keys[i]);
        }

        return result;
    }


    /**
     * Checks if a set of keys is in an array.
     *
     * @param src Source array. Must be sorted, if not, call {@link Arrays#sort(int[])} first. Otherwise, the
     * behavior of this method is undefined.
     * @param keys Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code keys} describing if the associated keys are in the
     * array.
     */
    public static boolean[] contains(int[] src, int... keys) {
        boolean[] result = new boolean[keys.length];

        for (int i = 0, size = keys.length; i < size; i++) {
            result[i] = contains(src, keys[i]);
        }

        return result;
    }


    /**
     * Checks if an array contains a specified value. This method assumes that the array is sorted as it uses the binary
     * search algorithm. If the array is not sorted, use {@link Arrays#sort(int[])} first.
     *
     * @param arr Array of interest.
     * @param key Value to check for in the {@code arr} array.
     * @return True if the {@code key} value is found in the array. False otherwise.
     *
     * @see Arrays#sort(int[])
     */
    public static boolean contains(int[] arr, int key) {
        return Arrays.binarySearch(arr, key) >= 0;
    }


    /**
     * Checks if an array contains a specified value. This method assumes that the array is sorted as it uses the binary
     * search algorithm. If the array is not sorted, use {@link Arrays#sort(int[])} first.
     *
     * @param arr Array of interest.
     * @param key Value to check for in the {@code arr} array.
     * @return True if the {@code key} value is found in the array. False otherwise.
     *
     * @see Arrays#sort(int[])
     */
    public static boolean contains(double[] arr, double key) {
        return Arrays.binarySearch(arr, key) >= 0;
    }

    // TODO NOW: Add a variant multiDimFlatten2Double that converts to a primitive `double[]` using
    //  Array.getDouble so that things like int[][][], long[][], float[][] can be flattened to `double[]`.
    //  This would be particularly useful for constructing a dense double nD-array from such objects.


    /// Flattens a Java array of multiple dimensions to a 1D Java array (e.g., double[][][] -> double[]).
    ///
    /// <blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Info:</strong> In general, this is <em>not</em> an efficient method and relies on reflection. This
    ///     method is provided for convenience but should be avoided if highly performant code is desired.
    /// </blockquote>
    ///
    /// @param src The source multidimensional array to flatten. This *must not* be a jagged array.
    /// @param out The output 1D array to store the result in. If `null`, then a new output array will be allocated. Otherwise,
    /// the component type and number of elements must match the component type of `src` and the total number of leaf components
    /// in `src`.
    /// @param clazz The required leaf component (i.e., non-array component) class of `src`.
    /// If `null`, this is ignored and *no* restriction on `src`'s leaf component type is enforced.
    /// @return A [Pair] containing, in order, the flattened 1D array and an `int[]` array containing the size of each dimension
    /// of the multidimensional `src` array. The 1D array will be:
    /// - A reference to `out` if `out` was not `null`.
    /// - Otherwise, a freshly allocated 1D array.
    ///
    /// @throws NullPointerException     If `src == null`.
    /// @throws IllegalArgumentException If `src` is not an array or is jagged.
    /// @throws IllegalArgumentException If `out != null` and is not properly sized or has a different component type as `src`.
    /// @throws IllegalArgumentException If `clazz != null` and the leaf component type of `src` is not assignable to this.
    public static Pair<Object, int[]> multiDimFlatten(Object src, Object out, Class<?> clazz) {
        Objects.requireNonNull(src);

        Class<?> arrayClass = src.getClass();
        if (!arrayClass.isArray()) {
            throw new IllegalArgumentException("Input is not an array.");
        }

        int rank = 0;
        Class<?> leaf = arrayClass;
        while (leaf.isArray()) {
            leaf = leaf.getComponentType();
            rank++;
        }

        int[] dims = new int[rank];
        long total = 1;
        Object current = src;

        for (int d = 0; d < rank; d++) {
            if (current == null) {
                throw new IllegalArgumentException("Encountered null sub-array at depth " + d + "; shape cannot be inferred.");
            }

            int len = Array.getLength(current);
            dims[d] = len;

            if (len == 0) {
                total = 0;
                break;
            }  // Trailing dims are unknowable; leave them zero. Rank must still increase for broadcasting.
            total *= len;
            current = Array.get(current, d == rank - 1 ? 0 : 0);
        }

        if (total > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("Flattened array would exceed max Java array length: " + total);
        }

        // Find the leaf component type in the source multidimensional array.
        Class<?> leafComponentType = arrayClass;
        while (leafComponentType.isArray()) {
            leafComponentType = leafComponentType.getComponentType();
        }

        if (clazz != null && !clazz.isAssignableFrom(leafComponentType)) {
            throw new IllegalArgumentException("src component class " + leafComponentType.getName() +
                    " is not assignable to the required class " + clazz.getName());
        }

        if (out == null) {
            // Allocate fresh array.
            out = Array.newInstance(leafComponentType, (int) total);
        } else {
            // Validate the provided array.
            var outClazz = out.getClass();

            if (!outClazz.isArray() || outClazz.getComponentType().isArray()) {
                throw new IllegalArgumentException("out is not a 1D array.");
            }

            if (leafComponentType != outClazz.getComponentType()) {
                throw new IllegalArgumentException("component classes of src and out do not match: " +
                        leafComponentType.getName() + " != " + outClazz.getComponentType().getName());
            }

            int len;
            if ((len = Array.getLength(out)) != total) {
                throw new IllegalArgumentException("out array with length " + len + " cannot fit the " +
                        total + " total elements contained in src.");
            }
        }

        // Do the flattening
        multiDimFlattenRec(src, 0, dims, out, 0);

        return new Pair<>(out, dims);
    }


    /// Helper to recursively flatten a multidimensional Java array.
    ///
    /// @param element The current element (array or component) being flattened.
    /// @param depth The current depth in the multidimensional array.
    /// @param dims The expected dimension sized. Used to enforce non-jagged structure.
    /// @param out The 1D array to write to.
    /// @param writeIndex An array of length `1` containing the current index of `out` that should be written to.
    private static int multiDimFlattenRec(
            Object element, int depth,
            int[] dims,
            Object out, int idx
    ) {
        if (depth == dims.length) { // Scalar leaf component.
            Array.set(out, idx++, element);
            return idx;
        }
        if (element == null || !element.getClass().isArray()) {
            throw new IllegalArgumentException("Jagged array: expected an array at depth " + depth + ".");
        }

        int len = Array.getLength(element);
        if (len != dims[depth]) {
            throw new IllegalArgumentException(
                    "Jagged array at depth " + depth + ": " + len + " != " + dims[depth] + ".");
        }

        // The innermost axis with a matching component type. Copy the whole thing over at once.
        if (depth == dims.length - 1 && element.getClass() == out.getClass()) {
            System.arraycopy(element, 0, out, idx, len);
            return idx + len;
        }

        for (int i = 0; i < len; i++)
            idx = multiDimFlattenRec(Array.get(element, i), depth + 1, dims, out, idx);

        return idx;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static int[] flatten(int[][] src) {
        int[] flat = new int[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (int[] row : src)
            for (int value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static double[] flatten(double[][] src) {
        double[] flat = new double[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (double[] row : src)
            for (double value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array and unboxes.
     *
     * @param src Array to flatten and unbox.
     * @return The flattened array.
     */
    public static double[] unboxFlatten(Double[][] src) {
        double[] flat = new double[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (Double[] row : src)
            for (double value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static Complex128[] flatten(Complex128[][] src) {
        Complex128[] flat = new Complex128[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (Complex128[] row : src)
            for (Complex128 value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static <T extends FieldElement<T>> T[] flatten(T[][] src) {
        T[] flat = (T[]) new FieldElement[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (T[] row : src)
            for (T value : row)
                flat[flatIdx++] = value;

        return (T[]) flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static <T extends SemiringElement<T>> T[] flatten(T[][] src) {
        T[] flat = (T[]) new SemiringElement[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (T[] row : src)
            for (T value : row)
                flat[flatIdx++] = value;

        return (T[]) flat;
    }


    /**
     * Given a list of integers, {@code srcAxes}, which is a subset of {@code {0, 1, 2, ...., dim-1}}
     * in no particular order, compute the integers which are in {@code {0, 1, 2, ...., dim-1}} but not in
     * {@code srcAxes}.
     *
     * @param srcAxes Source axes that contain a subset of {@code {0, 1, 2, ...., dim-1}} in no particular order.
     * @param dim Dimension of space that contains the axes of interest.
     * @return An array containing the set subtraction {@code {0, 1, 2, ...., dim-1}} - srcAxes.
     */
    public static int[] notInAxes(int[] srcAxes, int dim) {
        int[] notIn = new int[dim - srcAxes.length];

        // Copy and sort array.
        int[] srcAxesCopy = srcAxes.clone();
        Arrays.sort(srcAxesCopy);

        int srcIndex = 0;
        int notInIndex = 0;

        for (int i = 0; i < dim; i++) {
            if (srcIndex < srcAxesCopy.length && srcAxesCopy[srcIndex] == i) {
                srcIndex++;
            } else {
                notIn[notInIndex++] = i;
            }
        }

        return notIn;
    }


    /**
     * Shifts all elements of an array by a specified amount.
     *
     * @param shift Amount to shift array elements by.
     * @param arr Array of to shift.
     * @return A reference to {@code arr}.
     *
     * @see #shiftRange(int, int[], int, int)
     */
    public static int[] shift(int shift, int[] arr) {
        return shiftRange(shift, arr, 0, arr.length);
    }


    /**
     * Shifts a range of elements in an array by a specified amount.
     *
     * @param shift Amount to shift array elements by.
     * @param arr Array to shift.
     * @param start Starting index of range to shift (inclusive).
     * @param stop Stopping index of range to shift (exclusive).
     * @return A reference to {@code arr}.
     *
     * @throws ArrayIndexOutOfBoundsException If start or stop is not within the bounds of the {@code arr} array.
     * @see #shift(int, int[])
     */
    public static int[] shiftRange(int shift, int[] arr, int start, int stop) {
        for (int i = start; i < stop; i++)
            arr[i] += shift;

        return arr;
    }


    /**
     * Gets the unique values from an array and sorts them.
     *
     * @param src The array to get unique values from.
     * @return A sorted array containing all unique values in the {@code src} array.
     */
    public static int[] uniqueSorted(int[] src) {
        HashSet<Integer> hashSet = new HashSet<>();

        for (int j : src)
            hashSet.add(j);

        return hashSet.stream().mapToInt(Integer::intValue).sorted().toArray();
    }


    /**
     * Finds the fist slice of the specified {@code key} within an array. If the element does not exist, then {@code -1}
     * is returned.
     *
     * @param arr Array of interest.
     * @param key Key value to search for.
     * @return Returns the first slice of the value {@code key} within the {@code arr} array. If the {@code key} does
     * not occur in the array, {@code -1} will be returned.
     */
    public static int indexOf(int[] arr, int key) {
        for (int i = 0, size = arr.length; i < size; i++)
            if (arr[i] == key) return i;

        return -1;
    }


    /**
     * Counts the number of unique elements in an array.
     *
     * @param arr Array to count unique elements in.
     * @return The number of unique elements in {@code arr}.
     */
    public static int numUnique(double[] arr) {
        // For very large ndarrays, HashMap is quite a bit faster than HashSet.
        Map<Double, Double> map = new HashMap<>(arr.length);

        for (double a : arr)
            if (!map.containsKey(a)) map.put(a, a);

        return map.keySet().size();
    }


    /**
     * Counts the number of unique elements in an array.
     *
     * @param arr Array to count unique elements in.
     * @return The number of unique elements in {@code arr}.
     */
    public static int numUnique(int[] arr) {
        // For very large ndarrays, HashMap is quite a bit faster than HashSet.
        Map<Integer, Integer> map = new HashMap<>(arr.length);

        for (int a : arr)
            if (!map.containsKey(a)) map.put(a, a);

        return map.keySet().size();
    }


    /**
     * Creates a mapping of unique values in {code arr} to integers such that each unique value is mapped to a unique integer and those
     * integers range from {@code 0} to {@link #numUnique(int[]) numUnique(arr) - 1}.
     *
     * @param arr Array to create a mapping for.
     * @return A mapping of unique values in {code arr} to integers such that each unique value is
     * mapped to a unique integer and those integers range from {@code 0} to
     * {@link #numUnique(int[]) numUnique(arr) - 1}.
     */
    @SuppressWarnings("ConstantConditions")
    public static HashMap<Integer, Integer> createUniqueMapping(int[] arr) {
        if (arr.length == 0 || arr == null) return new HashMap<>();

        int[] arrSorted = Arrays.copyOf(arr, arr.length);
        Arrays.sort(arrSorted);

        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(arrSorted[0], 0);

        for (int i = 1; i < arrSorted.length; i++) {
            if (arrSorted[i - 1] != arrSorted[i]) {
                map.put(arrSorted[i], ++count);
            }
        }

        return map;
    }


    /**
     * Finds the first and last slice of a specified key within a sorted array.
     *
     * @param src The source array to search within. This array is assumed to be sorted. If the array is not sorted,
     * call {@link Arrays#sort(int[]) Arrays.sort(src)} before this method. If this is not done, and an
     * unsorted array is passed to this method, the results are undefined.
     * @param key The key value to find the first and last slice of within the {@code src} array.
     * @return An array of length 2 containing the first (inclusive) and last (exclusive) slice of the {@code key} within the {@code src} array.
     * If the {@code key} value does not exist in the array, then both first and last slice will be
     * {@code (-insertion_point - 1)} where {@code insertion_point} is defined as the slice the {@code key} would be
     * inserted into the sorted array.
     */
    public static int[] findFirstLast(int[] src, int key) {
        int keyIdx = Arrays.binarySearch(src, key);

        if (keyIdx < 0) return new int[]{keyIdx, keyIdx}; // Row not found.

        // Find the first entry with the specified row key.
        int lowerBound = keyIdx;
        for (int i = keyIdx; i >= 0; i--) {
            if (src[i] == key) {
                lowerBound = i;
            } else {
                break;
            }
        }

        int upperBound = keyIdx + 1;
        for (int i = upperBound; i < src.length; i++) {
            if (src[i] == key) {
                upperBound = i + 1;
            } else {
                break;
            }
        }

        return new int[]{lowerBound, upperBound};
    }


    /**
     * Applies a transform to an array. This is done in-place.
     *
     * @param src Array to apply transform to. Modified.
     * @param opp Operation to use to transform the array.
     * @return A reference to the {@code src} array.
     */
    public static double[] applyTransform(double[] src, UnaryOperator<Double> opp) {
        for (int i = 0, size = src.length; i < size; i++)
            src[i] = opp.apply(src[i]);

        return src;
    }


    /**
     * Applies a transform to an array. This is done in-place.
     *
     * @param src Array to apply transform to. Modified.
     * @param opp Operation to use to transform the array.
     * @return A reference to the {@code src} array.
     */
    public static <T> T[] applyTransform(T[] src, UnaryOperator<T> opp) {
        for (int i = 0, size = src.length; i < size; i++)
            src[i] = opp.apply(src[i]);

        return src;
    }


    /**
     * Applies a transform to an array. Note, unlike {@link #applyTransform(double[], UnaryOperator)} and
     * {@link #applyTransform(Object[], UnaryOperator)}, this method does <b>not</b> work in place.
     *
     * @param src Array to apply transform to.
     * @param opp Operation to use to transform the array.
     * @return A new array containing the result of the transformation.
     *
     * @see #applyTransform(Object[], UnaryOperator)
     * @see #applyTransform(double[], UnaryOperator)
     */
    public static Complex128[] applyTransform(double[] src, Function<Double, Complex128> opp) {
        Complex128[] dest = new Complex128[src.length];

        for (int i = 0, size = src.length; i < size; i++)
            dest[i] = opp.apply(src[i]);

        return dest;
    }


    /**
     * Applies a transform to an array. Note, unlike {@link #applyTransform(double[], UnaryOperator)} and
     * {@link #applyTransform(Object[], UnaryOperator)}, this method does <b>not</b> work in place.
     *
     * @param src Array to apply transform to.
     * @param opp Operation to use to transform the array.
     * @return A new array containing the result of the transformation.
     *
     * @see #applyTransform(Object[], UnaryOperator)
     * @see #applyTransform(double[], UnaryOperator)
     */
    public static double[] applyTransform(Complex128[] src, Function<Complex128, Double> opp) {
        double[] dest = new double[src.length];

        for (int i = 0, size = src.length; i < size; i++)
            dest[i] = opp.apply(src[i]);

        return dest;
    }


    /// Copies a strided range of an array.
    ///
    /// <blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
    ///     <strong>Examples:</strong>
    /// {@snippet :
    /// int[] arr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    /// copyOfStridedRange(arr, 4, 9, 1);   // [4, 5, 6, 7, 8]
    /// copyOfStridedRange(arr, 4, 9, 3);   // [4, 7]
    /// copyOfStridedRange(arr, 9, 4, -1);  // [9, 8, 7, 6, 5]
    /// copyOfStridedRange(arr, 9, 4, -3);  // [9, 6]}
    /// </blockquote>
    ///
    /// @param src The source array stop copy.
    /// @param start The starting index in `src` of the copy (inclusive).
    /// @param stop The ending index in `src` of the range stop copy (exclusive).
    /// @param stride The stride between elements of `src` stop copy. May be negative; must *not* be zero. If `stride` is negative,
    /// then the range is copied in reverse order (see examples).
    /// - If `stride > 0`, then `start <= stop` is required.
    /// - If `stride < 0`, then `start >= stop` is required.
    /// @return A copy of the specified strided range of `src`.
    ///
    /// @throws IllegalArgumentException If `stride == 0` or if `stride > 0` and `start > stop` or if `stride < 0` and `start < stop`.
    /// @throws IllegalArgumentException If `start` or `stop` is negative or larger than the length of `src`.
    public static int[] copyOfStridedRange(int[] src, int start, int stop, int stride) {
        NewValidateParameters.ensureSign(stride, NewValidateParameters.Sign.NON_ZERO, null);

        if (src.length == 0 && start == 0 && stop == 0) return new int[0]; // Quick return for empty src array.

        NewValidateParameters.ensureInRange(start, 0, src.length, NewValidateParameters.Inclusivity.INCLUSIVE_LOWER, null);
        NewValidateParameters.ensureInRange(stop, 0, src.length, NewValidateParameters.Inclusivity.INCLUSIVE, null);

        boolean reverse = stride < 0;
        var destSize = Math.ceilDiv(stop - start, stride);
        int[] dest = new int[Math.abs(destSize)];
        int c = 0;

        if (!reverse) {
            NewValidateParameters.ensureGreaterThanEq(stop, start, null);
            for (int j = start; j < stop; j += stride)
                dest[c++] = src[j];
        } else {
            NewValidateParameters.ensureGreaterThanEq(start, stop, null);
            for (int j = start; j > stop; j += stride)
                dest[c++] = src[j];
        }

        return dest;
    }
}
