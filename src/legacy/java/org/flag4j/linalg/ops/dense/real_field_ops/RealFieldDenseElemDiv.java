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

package org.flag4j.linalg.ops.dense.real_field_ops;


import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains low-level implementations of element-wise tensor division for a dense real and dense
 * field tensor.
 */
public final class RealFieldDenseElemDiv {

    // TODO: This threshold should be configurable
    /**
     * Minimum number of items in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 15_625;

    private RealFieldDenseElemDiv() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise division in (modified). May be the same array as {@code src1}.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> void elemDiv(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].div(src2[i]);
    }


    /**
     * Computes the element-wise division of two tensors using a concurrent algorithm.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise division in (modified). May be the same array as {@code src1}.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> void elemDivConcurrent(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++)
                dest[i] = src1[i].div(src2[i]);
        });
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise division in (modified). May be the same array as {@code src2}.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> void elemDiv(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src2[i].invert().mult(src1[i]);
    }


    /**
     * Computes the element-wise division of two tensors using a concurrent algorithm.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise division in (modified). May be the same array as {@code src2}.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> void elemDivConcurrent(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=0, size=src1.length; i<size; i++)
                dest[i] = src2[i].invert().mult(src1[i]);
        });
    }


    /**
     * Chooses if a concurrent algorithm for element-wise multiplication should be used based on the shape of the two tensors.
     * @param numEntries Total items in the tensors to multiply.
     * @return {@code true} if a concurrent algorithm should be used for element-wise multiplication; {@code false} otherwise.
     */
    private static boolean useConcurrent(int numEntries) {
        return numEntries >= CONCURRENT_THRESHOLD;
    }


    /**
     * Dynamically chooses and applies the appropriate algorithm for element-wise tensor multiplication.
     *
     * @param shape1 Shape of first tensor.
     * @param src1 Entries of first tensor.
     * @param shape2 Shape of second tensor.
     * @param src2 Entries of second tensor.
     *
     * @param dest Array to store the result of the element-wise division in (modified). May be the same array as {@code src2}.
     */
    public static <T extends FieldElement<T>> void dispatch(Shape shape1, double[] src1, Shape shape2, T[] src2, T[] dest) {
        if(useConcurrent(src1.length)) {
            // Use concurrent algorithm.
            elemDivConcurrent(src1, shape1, src2, shape2, dest);
        } else {
            // Then use standard algorithm.
            elemDiv(src1, shape1, src2, shape2, dest);
        }
    }


    /**
     * Dynamically chooses and applies the appropriate algorithm for element-wise tensor multiplication.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @param dest Array to store the result of the element-wise division in (modified). May be the same array as {@code src1}.
     */
    public static <T extends FieldElement<T>> void dispatch(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        if(useConcurrent(src1.length)) {
            // Use concurrent algorithm.
            elemDivConcurrent(src1, shape1, src2, shape2, dest);
        } else {
            // Then use standard algorithm.
            elemDiv(src1, shape1, src2, shape2, dest);
        }
    }
}
