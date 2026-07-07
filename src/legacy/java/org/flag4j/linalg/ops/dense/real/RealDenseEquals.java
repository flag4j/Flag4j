/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;

import java.util.Arrays;


/**
 * This class provides methods for checking the equality of real dense tensors.
 */
public final class RealDenseEquals {

    private RealDenseEquals() {
        // Hide constructor for utility class..
        
    }


    /**
     * Checks if two real dense vectors are equal.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return {@code true} if the two vectors are element-wise numerically equal; {@code false} otherwise.
     */
    public static boolean vectorEquals(double[] src1, double[] src2) {
        return Arrays.equals(src1, src2);
    }



    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(Matrix A, Matrix B) {
        return tensorEquals(A.data, A.shape, B.data, B.shape);
    }


    /**
     * Checks if two real dense tensors are equal.
     * @param A First tensor in comparison.
     * @param B Second tensor in comparison.
     * @return True if the two tensors are numerically element-wise equivalent.
     */
    public static boolean tensorEquals(Tensor A, Tensor B) {
        return tensorEquals(A.data, A.shape, B.data, B.shape);
    }


    /**
     * Checks if two dense tensors are equal. For the purposes of this method, {@link Double#NaN} values are considered equal.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return True if the two tensors are the same shape and numerically element-wise equivalent.
     */
    public static boolean tensorEquals(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        // Early return for mismatch in shapes.
        if(!shape1.equals(shape2)) return false;

        for(int i=0; i<src1.length; i++) {
            if(src1[i]!=src2[i] && !(Double.isNaN(src1[i]) && Double.isNaN(src2[i]))) {
                // Then tensors are not equal.
                return false;
            }
        }

        return true;
    }
}
