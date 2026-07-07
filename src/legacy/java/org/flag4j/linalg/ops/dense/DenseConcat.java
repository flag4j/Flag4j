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

package org.flag4j.linalg.ops.dense;


import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

import static org.flag4j.util.ArrayBuilder.getOrCreateArray;

/**
 * This utility class provides implementations for the concatenation of dense tensors.
 */
public final class DenseConcat {

    private DenseConcat() {
        // Hide default constructor for utility class.
    }


    /**
     * Concatenates two ndarrays and stores the result in {@code dest}.
     * @param src1 First array in the concatenation.
     * @param src2 Second array in the concatenation.
     * @param dest The array to store the array concatenation in. May be {@code null}. If not null, must be at least as large as
     * {@code src1.length + src2.length}.
     * @return If {@code dest != null} a reference to {@code dest} is returned. Otherwise, a new array is created and returned.
     * @throws IllegalArgumentException If {@code dest != null && dest.length < (src1.length + src2.length)}.
     */
    public static Object[] concat(Object[] src1, Object[] src2, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src1.length + src2.length]);
        if(dest.length < src1.length + src2.length) {
            throw new IllegalArgumentException(String.format("The size of the dest array must be at least as large sum of the sizes " +
                    "of the two source ndarrays but got sizes: dest=%d, src1=%d, src2=%d", dest.length, src1.length, src2.length));
        }

        System.arraycopy(src1, 0, dest, 0, src1.length);
        System.arraycopy(src2, 0, dest, src1.length, src2.length);

        return dest;
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param src Source array to repeat.
     * @param n Number of times to repeat the array.
     * @param axis Axis along which to repeat array:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     * @param dest The array to store the array repeated {@code src} array in. May be {@code null}. If not null, must be at least as
     * large as {@code src1.length + src2.length}.
     * @return If {@code dest != null} a reference to {@code dest} is returned. Otherwise, a new array is created and returned.
     * @throws IllegalArgumentException If {@code dest != null && dest.length < (src1.length + src2.length)}.
     */
    public static Object[] repeat(Object[] src, int n, int axis, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src.length*n]);
        if(dest.length < src.length*n) {
            throw new IllegalArgumentException(String.format("The size of the dest array must be able to store %d repetitions of" +
                    "the src array but was too small.", n));
        }
        ValidateParameters.ensureValidAxes(2, axis);
        ValidateParameters.ensureNonNegative(n);
        int size = src.length;

        if(axis==0) {
            for(int i=0; i<n; i++) // Set each row of the tiled matrix to be the vector values.
                System.arraycopy(src, 0, dest, i*size, size);
        } else {
            for(int i=0; i<size; i++) // Fill each row of the tiled matrix with a single value from the vector.
                Arrays.fill(dest, i*n, (i+1)*n, src[i]);
        }

        return dest;
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2&times;n} matrix.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n&times;2} matrix.
     * 
     *
     * @param src1 Entries of the first vector. Must be the same size as {@code src2}.
     * @param src2 Entries of the second vector. Must be the same size as {@code src1}.
     * @param axis Axis along which to stack vectors. Must be 1 or 0.
     * <ul>
     *     <li>If {@code axis=0}, then vectors are stacked as if they are row vectors. </li>
     *     <li>If {@code axis=1}, then vectors are stacked as if they are column vectors.</li>
     * </ul>
     * @param dest Array to store the stacked vectors in. May be {@code null}. If not {@code null}, must be at least as large as the
     * sum of the src array lengths.
     *
     * @return If {@code dest != null} a reference to {@code dest} is returned. Otherwise, a new array is created and returned.
     *
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     * @throws IllegalArgumentException If {@code dest != null} and {@code dest.length < src1.length + src2.length}.
     */
    public static Object[] stack(Object[] src1, Object[] src2, int axis, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src1.length + src2.length]);
        if(dest.length < src1.length + src2.length) {
            throw new IllegalArgumentException(String.format("The size of the dest array must be at least as large sum of the sizes " +
                    "of the two source ndarrays but got sizes: dest=%d, src1=%d, src2=%d", dest.length, src1.length, src2.length));
        }
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        ValidateParameters.ensureAxis2D(axis);

        if(axis==0) {
            // Copy items from each vector to the matrix.
            System.arraycopy(src1, 0, dest, 0, src1.length);
            System.arraycopy(src2, 0, dest, src1.length, src2.length);
        } else {
            int count = 0;
            for(int i=0, size=dest.length; i<size; i+=2) {
                dest[i] = src1[count];
                dest[i+1] = src2[count++];
            }
        }

        return dest;
    }
}
