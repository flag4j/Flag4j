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
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;


/**
 * This class provides low level methods for computing ops on dense field tensors.
 */
public final class DenseFieldOps {

    private DenseFieldOps() {
        // Hide constructor for utility class.
    }


    /**
     * Computes the element-wise division between two tensors.
     * @param shape1 Shape of the first tensor.
     * @param src1 Entries of the first tensor.
     * @param shape2 Shape of the second tenor.
     * @param src2 Entries of the second tensor.
     * @param dest Array to store the result in. May be the same array as either {@code src1} or {@code src2}.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src2.length < src1.length || dest.length < src1.length}.
     */
    public static <T extends Field<T>> void div(Shape shape1, T[] src1,
                                                Shape shape2, T[] src2,
                                                T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].div(src2[i]);
    }
}
