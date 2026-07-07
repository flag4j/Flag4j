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
 * This utility class contains ops for tensors whose elements are members of a
 * {@link FieldElement}. The implementations in this class are agnostic
 */
public final class SemiringOps {


    private SemiringOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the scalar multiplication of a tensor.
     *
     * @param src Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @param dest Array to store result in.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not at least the size of {@code src}.
     */
    public static <T extends SemiringElement<T>> void scalMult(
            T[] src, T factor, T[] dest) {
        int size = src.length;

        for(int i=0; i<size; i++)
            dest[i] = src[i].mult(factor);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     *
     * @param src Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @param dest Array to store result in.
     * @param start Starting index of scalar multiplication.
     * @param stop Stopping index of scalar multiplication.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not the size of {@code src}.
     */
    public static <T extends SemiringElement<T>> void scalMult(
            T[] src, T factor, T[] dest, int start, int stop) {

        for(int i=start; i<stop; i++)
            dest[i] = src[i].mult(factor);
    }


    /**
     * Sums a value to each entry of tensor.
     *
     * @param src Entries of the tensor (non-zero items if tensor is sparse).
     * @param summand Value to sum to each entry of the tensor.
     * @param dest Array to store result in.
     */
    public static <T extends SemiringElement<T>> void add(T[] src, T summand, T[] dest) {
        T val = summand;
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].add(val);
    }
}
