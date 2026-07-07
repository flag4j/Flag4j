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

package org.flag4j.arrays.backend.ring_arrays;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringMatrix;
import org.flag4j.linalg.MatrixNorms;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;
import org.flag4j.linalg.ops.dense.ring_ops.DenseRingTensorOps;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * The base class for all dense matrices whose elements are members of a {@link RingElement}.
 * @param <T> The type of this matrix.
 * @param <U> The type of the vector which is of similar type to {@link T &lt;T&gt;}.
 * @param <V> The type of the ndarrays the items of the matrix belong to.
 */
public abstract class AbstractDenseRingMatrix<T extends AbstractDenseRingMatrix<T, U, V>,
        U extends AbstractDenseRingVector<U, T, V>, V extends RingElement<V>>
        extends AbstractDenseSemiringMatrix<T, U, V>
        implements RingTensorMixin<T, T, V>, MatrixMixin<T, T, U, V> {


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     */
    protected AbstractDenseRingMatrix(Shape shape, V[] data) {
        super(shape, data);
    }


    /**
     * Computes the element-wise difference of two matrices.
     *
     * @param b Second matrix in the element-wise difference.
     *
     * @return The element-wise difference of this matrix and {@code b}.
     */
    @Override
    public T sub(T b) {
        V[] dest = makeEmptyDataArray(data.length);
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Computes the element-wise difference between two matrices of the same shape and stores the result in this matrix.
     *
     * @param b Second matrix in the element-wise difference.
     *
     * @throws NDArrayShapeException If this matrix and {@code b} do not have the same shape.
     */
    public void subEq(T b) {
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, data);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        return T();
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public T H(int... axes) {
        return T(axes);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    @Override
    public boolean isHermitian() {
        if(this==null) return false;
        if(this.data.length==0) return true;

        return numRows==numCols && DenseRingTensorOps.isHermitian(shape, data);
    }


    /**
     * Checks if the matrix is "close" to an identity matrix. Two entries {@code x} and {@code y} are considered
     * "close" if they satisfy the following:
     * <pre>{@code
     *      |x-y| <= (1E-08 + 1E-05*|y|)
     * }</pre>
     *
     * @return {@code true} if the matrix is approximately an identity matrix, otherwise {@code false}.
     */
    public boolean isCloseToIdentity() {
        return DenseRingTensorOps.isCloseToIdentity(shape, data);
    }


    /**
     * Checks if two sparse CSR ring matrices are element-wise equal within the following tolerance for two entries {@code x}
     * and {@code y}:
     * <pre>{@code
     *  |x-y| <= (1e-08 + 1e-05*|y|)
     * }</pre>
     *
     * To specify the relative and absolute tolerances use {@link #allClose(AbstractDenseRingMatrix, double, double)}
     *
     * @return {@code true} if this matrix and {@code b} element-wise equal within the tolerance {@code |x-y| <= (1e-08 + 1e-05*|y|)}.
     * @see #allClose(AbstractDenseRingMatrix, double, double)
     */
    public boolean allClose(T b) {
        return allClose(b, 1e-05, 1e-08);
    }


    /**
     * Checks if two matrices are element-wise equal within the tolerance specified by {@code relTol} and {@code absTol}. Two elements
     * {@code x} and {@code y} are considered "close" if they satisfy the following:
     * <pre>{@code
     *  |x-y| <= (absTol + relTol*|y|)
     * }</pre>
     * @param b Matrix to compare to this matrix.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return {@code true} if the {@code src1} matrix is the same shape as the {@code src2} matrix and all items
     * are "close", i.e., elements {@code a} and {@code b} at the same positions in the two matrices respectively
     * satisfy {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns {@code false}.
     * @see #allClose(AbstractDenseRingMatrix)
     */
    public boolean allClose(T b, double relRol, double absTol) {
        return RingProperties.allClose(data, b.data, relRol, absTol);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm(int p, int q) {
        return MatrixNorms.norm(this, p, q);
    }
}
