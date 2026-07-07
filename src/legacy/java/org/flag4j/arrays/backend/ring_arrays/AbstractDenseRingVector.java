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
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringVector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.dense.ring_ops.DenseRingTensorOps;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * <p>The base class for all dense vectors whose items are {@link RingElement} elements.
 *
 * <p>Vectors are 1D tensors (i.e., rank 1 tensor).
 *
 * <p>AbstractDenseSemiringVectors have mutable {@link #data} but a fixed {@link #shape}.
 *
 * @param <T> Type of the vector.
 * @param <U> Type of matrix equivalent to this vector.
 * @param <V> Type of the {@link RingElement} element of this vector.
 */
public abstract class AbstractDenseRingVector<T extends AbstractDenseRingVector<T, U, V>,
        U extends AbstractDenseRingMatrix<U, T, V>, V extends RingElement<V>>
        extends AbstractDenseSemiringVector<T, U, V>
        implements RingTensorMixin<T, T, V>, VectorMixin<T, U, U, V> {


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     */
    protected AbstractDenseRingVector(Shape shape, V[] data) {
        super(shape, data);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public T normalize() {
        throw new UnsupportedOperationException("Normalization not supported for ndarrays vectors.");
    }


    /**
     * Computes the squared magnitude of this vector.
     *
     * @return The squared magnitude of this vector.
     *
     * @see #mag()
     */
    @Override
    public double magSquared() {
        return VectorNorms.normSquared(data) ;
    }


    /**
     * Computes the norm of this vector. This is the same as {@link #mag()}.
     * @return The norm (specifically &ell;<sup>2</sup>) of this vector.
     * @see #mag()
     * @see #magSquared()
     */
    @Override
    public double norm() {
        return VectorNorms.norm(data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm(double p) {
        return VectorNorms.norm(data, p);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws NDArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        V[] diff = makeEmptyDataArray(data.length);
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, diff);
        return makeLikeNDArray(shape, diff);
    }


    /**
     * Computes the element-wise difference between two vectors of the same shape and stores the result in this vector.
     *
     * @param b Second vectors in the element-wise difference.
     *
     * @throws NDArrayShapeException If this vectors and {@code b} do not have the same shape.
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
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        return conj();
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
        ValidateParameters.ensureValidAxes(shape, axes);
        return conj();
    }
}
