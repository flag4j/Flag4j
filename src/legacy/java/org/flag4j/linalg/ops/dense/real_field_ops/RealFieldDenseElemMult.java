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

package org.flag4j.linalg.ops.dense.real_field_ops;

import org.flag4j.numbers.FieldElement;
import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ValidateParameters;


/**
 * This class contains low-level implementations of element-wise tensor multiplication for a
 * dense real and dense field tensor.
 */
public final class RealFieldDenseElemMult {

    // TODO: This should be configurable.
    /**
     * Minimum number of items in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 800_000;


    private RealFieldDenseElemMult() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise multiplication of two tensors. Also called the Hadamard product.
     * @param src1 First tensor in element-wise multiplication.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise multiplication.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise product in (modified). May be the same array as {@code src1}.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> void elemMult(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].mult(src2[i]);
    }


    /**
     * Computes the element-wise multiplication of two tensors using a concurrent algorithm. Also called the Hadamard product.
     * @param src1 First tensor in element-wise multiplication.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise multiplication.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise product in (modified). May be the same array as {@code src1}.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> void elemMultConcurrent(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++)
                dest[i] = src1[i].mult(src2[i]);
        });
    }


    /**
     * Dynamically chooses and applies element-wise multiplication algorithm to use based on the number of items in the tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @param dest Array to store the result of the element-wise product in (modified). May be the same array as {@code src1}.
     */
    public static <T extends FieldElement<T>> void dispatch(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        if(src1.length < CONCURRENT_THRESHOLD)
            elemMult(src1, shape1, src2, shape2, dest);
        else
            elemMultConcurrent(src1, shape1, src2, shape2, dest);
    }
}
