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


import org.flag4j.numbers.SemiringElement;

import java.util.function.BinaryOperator;
import java.util.function.Function;

/**
 * A utility class for computing transformations on {@link SemiringElement} ndarrays.
 */
public final class TransformSemiRing {

    private TransformSemiRing() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Applies a map-reduce procedure to an array of {@link SemiringElement} values. This is equivalent to
     * {@code reduce(map(values, mapper), accumulator)}.
     * @param values Values to apply map-reduce to; must not be null.
     * @param mapper Mapper to apply during the {@link #map(SemiringElement[], Function) mapping} step; must not be null.
     * @param accumulator Accumulator to use during the {@link #reduce(SemiringElement[], BinaryOperator) reduction} step.
     * @return The result of applying the map-reduce procedure using the specified {@code mapper} and {@code accumulator} to the
     * {@code values} array.
     */
    public static <T extends SemiringElement<T>> SemiringElement mapReduce(T[] values,
                                                                           Function<T, T> mapper,
                                                                           BinaryOperator<T> accumulator) {
        return reduce(map(values, mapper), accumulator);
    }



    /**
     * <p>Applies a specified function to each element of the given array and returns a new array containing the results of the
     * mapping operation. This produces a new array where each element is the result of applying the provided {@code mapper}
     * function to the corresponding element in the input array.
     *
     * <p>For example, if the array contains elements [a, b, c] and the {@code mapper} function multiplies each element by 2,
     * the returned array will contain [2a, 2b, 2c].
     *
     * @param values the array of field elements to map; must not be null
     * @param mapper the mapping function to apply to each element; must not be null
     * @return a new array containing the results of applying the {@code mapper} function to each element in the {@code values} array
     * @throws NullPointerException if {@code values} or {@code mapper} is null
     * @throws IllegalArgumentException if {@code values} is empty
     */
    public static <T extends SemiringElement<T>> T[] map(T[] values, Function<T, T> mapper) {
        if (values == null || mapper == null)
            throw new NullPointerException("Values and mapper must not be null.");
        if (values.length == 0)
            throw new IllegalArgumentException("Cannot map array with length zero.");

        T[] result = (T[]) new SemiringElement[values.length];

        for (int i=0, size=values.length; i<size; i++)
            result[i] = mapper.apply(values[i]);

        return result;
    }


    /**
     * <p>Reduces the given array of field elements to a single element by repeatedly applying a binary operator
     * to combine elements. The reduction is performed from left to right, starting with the first element
     * in the array and hence is a "left-reduce" or "left fold".
     *
     * <p>The {@code reduce} operation applies the specified {@code accumulator} function to the first
     * element and the second element, then applies it to the result and the third element, and so on
     * until all elements have been combined into a single result. The operation assumes that the array
     * contains at least one element.
     *
     * <p>For example, if the array contains elements [a, b, c, d] and the accumulator is a function
     * which adds elements, the method would compute (((a + b) + c) + d).
     *
     * @param values The array of field elements to reduce. Cannot be empty or null.
     * @param accumulator The binary operator to combine elements.
     * @return the result of lef-reducing all elements {@code values} using the specified {@code accumulator}.
     * @throws IllegalArgumentException if {@code values} is empty
     * @throws NullPointerException if {@code values} or {@code accumulator} is null
     */
    public static <T extends SemiringElement<T>> SemiringElement<T> reduce(T[] values, BinaryOperator<T> accumulator) {
        if(values == null || accumulator == null)
            throw new NullPointerException("Values and accumulator must not be null.");
        if(values.length == 0)
            throw new IllegalArgumentException("Cannot reduce array with length zero.");

        T result = values[0];

        for(int i = 1, size = values.length; i < size; i++)
            result = accumulator.apply(result, values[i]);

        return result;
    }
}
