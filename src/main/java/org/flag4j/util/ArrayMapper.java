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

import java.util.BitSet;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.function.LongUnaryOperator;
import java.util.function.UnaryOperator;


/**
 * A utility class for applying a mapping function to all elements of an array.
 * This class contains implementations for {@link Object} arrays and all primitive type arrays.
 */
public final class ArrayMapper {

    // TODO: Consider adding concurrent variants and possibly dispatchers.

    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @param <T> The element type of the array.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final <T> void map(T[] arr, UnaryOperator<T> mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.apply(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(boolean[] arr, UnaryOperator<Boolean> mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.apply(arr[i]);
    }


    /**
     * Applies a mapping function to each element in a {@link BitSet}.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static void map(BitSet data, UnaryOperator<Boolean> mapper) {
        for(int i=0, size=data.size(); i<size; i++)
            data.set(i, mapper.apply(data.get(i)));
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(byte[] arr, UnaryOperator<Byte> mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.apply(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(short[] arr, UnaryOperator<Short> mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.apply(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(int[] arr, IntUnaryOperator mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.applyAsInt(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(long[] arr, LongUnaryOperator mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.applyAsLong(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(float[] arr, UnaryOperator<Float> mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.apply(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     */
    public static final void map(double[] arr, DoubleUnaryOperator mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.applyAsDouble(arr[i]);
    }


    /**
     * Applies a mapping function to each element in an array.
     * @param arr The array to apply the mapping function to.
     * @param mapper The mapping function to apply to {@code arr}.
     * @throws NullPointerException If {@code arr} or {@code mapper} is {@code null}.
     * @see #map(double[], DoubleUnaryOperator)
     */
    public static final void map(double[] arr, UnaryOperator<Double> mapper) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = mapper.apply(arr[i]);
    }
}
