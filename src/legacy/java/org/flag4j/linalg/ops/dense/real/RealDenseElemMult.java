/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense.real;


import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains low-level implementations of element-wise multiplications algorithms for real dense tensors.
 */
public final class RealDenseElemMult {

    /**
     * Minimum number of items in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 30_000_000;

    private RealDenseElemMult() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise multiplication of two tensors.
     *
     * @param src1 First tensor in element-wise multiplication.
     * @param src2 Second tensor in element-wise multiplication.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src1} or {@code src2}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array of the appropriate size is
     * created and returned.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src1.length}.
     */
    public static double[] elemMult(double[] src1, double[] src2, double[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        dest = ArrayBuilder.getOrCreateArray(dest, src1.length);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i]*src2[i];

        return dest;
    }


    /**
     * Computes the element-wise multiplication of two tensors using a concurrent algorithm.
     *
     * @param src1 First tensor in element-wise multiplication.
     * @param src2 Second tensor in element-wise multiplication.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src1} or {@code src2}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array of the appropriate size is
     * created and returned.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src1.length}.
     */
    public static double[] elemMultConcurrent(double[] src1, double[] src2, double[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        dest = ArrayBuilder.getOrCreateArray(dest, src1.length);

        double[] finalDest = dest;
        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++)
                finalDest[i] = src1[i]*src2[i];
        });

        return dest;
    }


    /**
     * <p>Dynamically chooses and applies element-wise multiplication algorithm to use based on the number of items in the tensors.
     *
     * @param src1 Entries of first tensor.
     * @param src2 Entries of second tensor.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src1} or {@code src1}.
     *
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * size will be created and returned.
     */
    public static double[] dispatch(double[] src1, double[] src2, double[] dest) {
        if(src1.length < CONCURRENT_THRESHOLD)
            return elemMult(src1, src2, dest);
        else
            return elemMultConcurrent(src1, src2, dest);
    }
}
