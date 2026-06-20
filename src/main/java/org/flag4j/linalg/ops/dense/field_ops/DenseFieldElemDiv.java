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

package org.flag4j.linalg.ops.dense.field_ops;


import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;


/**
 * This class contains low-level implementations of element-wise tensor division between two {@link Field} tensors.
 */
public final class DenseFieldElemDiv {
    // TODO: The CONCURRENT_THRESHOLD should be configurable. This should serve as a default value and an overloaded method should
    //  be provided for specifying the threshold so that different Field implementations can specify their own value.

    /**
     * Minimum number of data in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 1250;


    private DenseFieldElemDiv() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise division in.
     * @throws NDArrayShapeException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> void elemDiv(T[] src1, Shape shape1,
                                                    T[] src2, Shape shape2,
                                                    T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0; i<dest.length; i++)
            dest[i] = src1[i].div(src2[i]);
    }


    /**
     * Computes the element-wise division of two tensors using a concurrent algorithm.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the element-wise division in.
     * @throws NDArrayShapeException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> void elemDivConcurrent(T[] src1, Shape shape1,
                                                              T[] src2, Shape shape2,
                                                              T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        ThreadManager.concurrentOperation(dest.length, (start, end) -> {
            for(int i=start; i<end; i++)
                dest[i] = src1[i].div(src2[i]);
        });
    }


    /**
     * Dynamically chooses and applies element-wise division algorithm to use based on the number of data in the tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @param dest Array to store the result of the element-wise division in.
     * @throws NDArrayShapeException
     */
    public static <T extends Field<T>> void dispatch(T[] src1, Shape shape1,
                                                     T[] src2, Shape shape2,
                                                     T[] dest) {
        if(src1.length < CONCURRENT_THRESHOLD)
            elemDiv(src1, shape1, src2, shape2, dest);
        else
            elemDivConcurrent(src1, shape1, src2, shape2, dest);
    }
}
