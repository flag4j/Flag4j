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

import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Complex64;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>The {@code ArrayConversions} class provides utility methods for converting between various array types
 * and formats. This includes conversions from primitive ndarrays to object ndarrays and between lists and ndarrays.
 *
 * <p>Designed to handle common conversion use cases efficiently, this class supports operations such as:
 * <ul>
 *   <li>Converting primitive (or boxed primitive) ndarrays to {@link Complex128} and {@link Complex64} representations.</li>
 *   <li>Boxing and unboxing of primitive and object types (e.g., {@code int[]} to {@code Integer[]}).</li>
 *   <li>Converting ndarrays to {@link java.util.ArrayList ArrayList} and vice versa.</li>
 *   <li>Transforming between primitive and custom numerical representations (e.g., {@code int[]} to {@code double[]}).</li>
 * </ul>
 *
 * <h2>Usage Examples</h2>
 * <pre>{@code
 * // Convert an array of integers to Complex128 array.
 * int[] intArray = {1, 2, 3};
 * Complex128[] complexArray = ArrayConversions.toComplex128(intArray, null);
 *
 * // Convert an ArrayList of Integers to an int array.
 * List<Integer> integerList = List.of(1, 2, 3);
 * int[] intArrayFromList = ArrayConversions.fromIntegerList(integerList);
 *
 * // Convert a double array to an ArrayList.
 * double[] doubleArray = {1.1, 2.2, 3.3};
 * ArrayList<Double> doubleList = ArrayConversions.toArrayList(doubleArray);
 *
 * // Box a primitive int array to Integer[].
 * int[] primitiveIntArray = {1, 2, 3};
 * Integer[] boxedArray = ArrayConversions.boxed(primitiveIntArray);
 * }</pre>
 *
 * <h2>Restrictions:</h2>
 * <ul>
 *   <li>All source ndarrays and lists must be non-{@code null} unless explicitly stated otherwise.</li>
 *   <li>The caller must ensure that destination ndarrays have sufficient capacity when provided.</li>
 * </ul>
 *
 * <p><strong>Note:</strong> This class is a utility class and cannot be instantiated.</p>
 */
public final class ArrayConversions {

    private ArrayConversions() {
        // Hide default constructor for utility class.
    }


    /**
     * Converts an array of {@code int}s to one of {@link Complex128}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] toComplex128(int[] src, Complex128[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex128[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@code double}s to one of {@link Complex128}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is {@code null}, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] toComplex128(double[] src, Complex128[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex128[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@link Integer}s to one of {@link Complex128}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] toComplex128(Integer[] src, Complex128[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex128[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@code Double}s to one of {@link Complex128}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] toComplex128(Double[] src, Complex128[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex128[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@link Complex64}s to an array of {@link Complex128}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] toComplex128(Complex64[] src, Complex128[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex128[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@link String}s to one of {@link Complex128}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] toComplex128(String[] src, Complex128[] dest) {
        if(dest == null) {
            dest = new Complex128[src.length];
        }
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }
    
    
    /**
     * Converts an array of {@code int}s to one of {@link Complex64}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex64[] toComplex64(int[] src, Complex64[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex64[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex64(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@code float}s to one of {@link Complex64}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is {@code null}, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex64[] toComplex64(float[] src, Complex64[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex64[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex64(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@link src} to one of {@link Complex64}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex64[] toComplex64(Integer[] src, Complex64[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex64[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex64(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@link Float}s to one of {@link Complex64}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex64[] toComplex64(Float[] src, Complex64[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, () -> new Complex64[src.length]);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex64(src[i]);

        return dest;
    }


    /**
     * Converts an array of {@link String}s to one of {@link Complex64}s.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination ndarrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex64[] toComplex64(String[] src, Complex64[] dest) {
        if(dest == null) {
            dest = new Complex64[src.length];
        }
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex64(src[i]);

        return dest;
    }


    /**
     * Converts an array of doubles to an {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Double> toArrayList(double[] src) {
        ArrayList<Double> list = new ArrayList<>(src.length);

        for(double value : src)
            list.add(value);

        return list;
    }


    /**
     * Converts an array of complex numbers to an {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Complex128> toArrayList(Complex128[] src) {
        ArrayList<Complex128> list = new ArrayList<>(src.length);
        list.addAll(Arrays.asList(src));
        return list;
    }


    /**
     * Converts an array of doubles to a complex {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent complex array list.
     */
    public static ArrayList<Complex128> toComplexArrayList(double[] src) {
        ArrayList<Complex128> list = new ArrayList<>(src.length);

        for (double value : src)
            list.add(new Complex128(value));

        return list;
    }


    /**
     * Converts an array of doubles to an {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Integer> toArrayList(int[] src) {
        ArrayList<Integer> list = new ArrayList<>(src.length);

        for (int value : src)
            list.add(value);

        return list;
    }


    /**
     * Converts a list of {@link Double Doubles} objects to a primitive array.
     *
     * @param src The source list to convert.
     * @return An array containing the same values as the {@code src} list.
     */
    public static double[] fromDoubleList(List<Double> src) {
        double[] dest = new double[src.size()];

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = src.get(i);

        return dest;
    }


    /**
     * Converts a list of {@link Integer Integer} objects to a primitive array.
     *
     * @param src The source list to convert.
     * @return An array containing the same values as the {@code src} list.
     */
    public static int[] fromIntegerList(List<Integer> src) {
        int[] dest = new int[src.size()];

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = src.get(i);

        return dest;
    }


    /**
     * Converts a list of {@link Integer Integer} objects to a primitive array.
     *
     * @param src  The source list to convert.
     * @param dest Destination array to store values from {@code src} in (modified). Must be at least as large as {@code src}.
     * @return A reference to the {@code dest} array.
     */
    public static int[] fromIntegerList(List<Integer> src, int[] dest) {
        ValidateParameters.ensureGreaterEq(src.size(), dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = src.get(i);

        return dest;
    }


    /**
     * Converts a list to an array.
     *
     * @param src The Source list to convert.
     * @param dest Destination array to store values from {@code src} in (modified). Must be at least as large as {@code src}.
     * @return A reference to the {@code dest} array.
     * @throws IllegalArgumentException If the {@code dest} array is not large enough to store all items of {@code src}
     *                                  list.
     */
    public static <T> T[] fromList(List<T> src, T[] dest) {
        ValidateParameters.ensureGreaterEq(src.size(), dest.length);
        return src.toArray(dest);
    }


    /**
     * Converts an array of {@link Double} objects to a primitive array (i.e., unboxing).
     *
     * @param arr Array to unbox.
     * @param dest Destination array for the unboxed values. May be {@code null}.
     * @return If dest was <em>not</em> {@code null} then a reference to {@code dest} is returned. Otherwise, a new array with the unboxed
     * values is returned.
     */
    public static double[] unbox(Double[] arr, double[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, arr.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = arr[i];

        return dest;
    }


    /**
     * Converts an array of {@link Integer} objects to a primitive array (i.e., unboxing).
     * @param arr Array to unbox.
     * @param dest Destination array for the unboxed values. May be {@code null}.
     * @return If dest was <em>not</em> {@code null} then a reference to {@code dest} is returned. Otherwise, a new array with the unboxed
     * values is returned.
     */
    public static int[] unbox(Integer[] arr, int[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, arr.length);

        for (int i = 0, size=dest.length; i < size; i++)
            dest[i] = arr[i];

        return dest;
    }


    /**
     * Converts a primitive array to an array of an equivalent boxed type.
     *
     * @param src The source primitive array to box.
     * @return A boxed array equivalent to the {@code src} primitive array.
     */
    public static Double[] boxed(double[] src) {
        int size = src.length;
        Double[] boxed = new Double[size];

        for (int i = 0; i < size; i++)
            boxed[i] = src[i];

        return boxed;
    }


    /**
     * Converts a primitive array to an array of an equivalent boxed type.
     *
     * @param src The source primitive array to box.
     * @return A boxed array equivalent to the {@code src} primitive array.
     */
    public static Integer[] boxed(int[] src) {
        int size = src.length;
        Integer[] boxed = new Integer[size];

        for (int i = 0; i < size; i++)
            boxed[i] = src[i];

        return boxed;
    }


    /**
     * Converts an array of ints to an array of doubles.
     *
     * @param src  Source array to convert.
     * @param dest Destination array to store double values equivalent to the values in the {@code src} array.
     *             If null, a new double array with the same size as {@code src} will be created.
     * @return A reference to the {@code dest} array.
     */
    public static double[] asDouble(int[] src, double[] dest) {
        dest = ArrayBuilder.getOrCreateArray(dest, src.length);

        for (int i = 0, size=src.length; i < size; i++)
            dest[i] = src[i];

        return dest;
    }


    /**
     * Converts an array of {@link Integer Integers} to an array of doubles.
     *
     * @param src  Source array to convert.
     * @param dest Destination array to store double values equivalent to the values in the {@code src} array.
     *             If null, a new double array with the same size as {@code src} will be created.
     * @return A reference to the {@code dest} array.
     */
    public static double[] asDouble(Integer[] src, double[] dest) {
        if (dest == null) dest = new double[src.length];

        for(int i = 0, size=src.length; i < size; i++)
            dest[i] = src[i];

        return dest;
    }
}
