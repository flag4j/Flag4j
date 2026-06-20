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

package org.flag4j.linalg.ops.dense.ring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.numbers.Ring;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * Utility class for computing ops between two dense {@link Ring} tensors.
 */
public final class DenseRingTensorOps {

    private DenseRingTensorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise difference between two dense tensors.
     * @param shape1 Shape of the first tensor in the element-wise difference.
     * @param src1 Entries of the first tensor in the element-wise difference.
     * @param shape2 Shape of the second tensor in the element-wise difference.
     * @param src2 Entries of the second tensor in the element-wise difference.
     * @param dest Array to store the resulting element-wise difference. May be the same array as either {@code src1} or {@code src2}.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src2.length < src2.length || dest.length < src1.length}
     */
    public static <T extends Ring<T>> void sub(Shape shape1, T[] src1,
                                               Shape shape2, T[] src2,
                                               T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].sub(src2[i]);
    }


    /**
     * Checks if a complex dense matrix is Hermitian. That is, if the matrix is equal to its conjugate transpose.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    public static <T extends Ring<T>> boolean isHermitian(Shape shape, T[] src) {
        if(shape.getSize(0)!=shape.getSize(1)) return false;

        int numCols = shape.getSize(1);

        for(int i = 0, rows = shape.getSize(0); i<rows; i++) {
            int count1 = i*numCols;
            int count2 = i;
            int stop = count1 + i;

            while(count1 < stop) {
                if(!src[count1++].equals(src[count2].conj())) return false;
                count2 += numCols;
            }
        }

        return true;
    }


    /**
     * <p>Checks if a matrix is the identity matrix approximately.
     *
     * <p>Specifically, if the diagonal data are no farther than
     * {@code 1.001E-5} in absolute value from {@code 1.0} and the non-diagonal data are no larger than {@code 1e-08} in absolute
     * value.
     *
     * <p>These thresholds correspond to the thresholds from the
     * {@link org.flag4j.linalg.ops.common.ring_ops.RingProperties#allClose(Ring[], Ring[])} method.
     *
     *
     * @param src Matrix of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is close the identity matrix or if the matrix has zero data.
     */
    public static <T extends Ring<T>> boolean isCloseToIdentity(Shape shape, T[] src) {
        int numRows = shape.getSize(0);
        int numCols = shape.getSize(1);

        if(src == null || numRows!=numCols) return false;
        if(src.length == 0) return true;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;
        final T ONE = src[0].getOne();
        int rows = numRows;
        int cols = numCols;
        int pos = 0;

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                if((i==j && src[pos].sub(ONE).mag() > diagTol)
                        || (i!=j && src[pos].mag() > nonDiagTol)) {
                    return false;
                }

                pos++;
            }
        }

        return true; // If we make it here, the matrix is "close" to the identity matrix.
    }
}
