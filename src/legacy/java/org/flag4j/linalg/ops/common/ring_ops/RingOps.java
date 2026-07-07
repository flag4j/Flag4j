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
 * Utility class useful for computing ops on {@link RingElement} tensors.
 */
public final class RingOps {

    private RingOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Subtracts a scalar value from each entry of an array.
     * @param src Array to subtract scalar from.
     * @param scalar Scalar value to subtract from {@code src}.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <T extends RingElement<T>> void sub(T[] src, T scalar, T[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].sub(scalar);
    }


    /**
     * Computes the element-wise absolute value of an array.
     * @param src Array to compute element-wise absolute value of.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <T extends RingElement<T>> void abs(T[] src, double[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].abs();
    }


    /**
     * Computes the element-wise conjugation of an array.
     * @param src Array to compute element-wise conjugation of.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <T extends RingElement<T>> void conj(T[] src, T[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].conj();
    }
}
