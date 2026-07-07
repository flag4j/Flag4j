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

import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.FieldElement;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldTensor;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.util.ArrayUtils;

/**
 * This class provides methods for checking the equality of one dense real and one dense field tensors.
 */
public final class RealFieldDenseEquals {

    private RealFieldDenseEquals() {
        // Hide default constructor.
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static <T extends FieldElement<T>> boolean matrixEquals(Matrix A, AbstractDenseFieldMatrix<?, ?, T> B) {
        return tensorEquals(A.data, A.shape, B.data, B.shape);
    }


    /**
     * Checks if two real dense tensors are equal.
     * @param A First tensor in comparison.
     * @param B Second tensor in comparison.
     * @return True if the two tensors are numerically element-wise equivalent.
     */
    public static <T extends FieldElement<T>> boolean tensorEquals(Tensor A, AbstractDenseFieldTensor<?, T> B) {
        return tensorEquals(A.data, A.shape, B.data, B.shape);
    }


    /**
     * Checks if two dense tensors are equal.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return True if the two tensors are numerically element-wise equivalent.
     */
    public static <T extends FieldElement<T>> boolean tensorEquals(double[] src1, Shape shape1, T[] src2, Shape shape2) {
        return shape1.equals(shape2) && ArrayUtils.equals(src1, (Complex128[]) src2);
    }
}
