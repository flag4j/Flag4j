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

package org.flag4j.arrays.backend.ring_arrays;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringTensor;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.linalg.ops.dense.ring_ops.DenseRingTensorOps;
import org.flag4j.numbers.Ring;
import org.flag4j.util.exceptions.ArrayShapeException;

/**
 * <p>The base class for all dense {@link Ring} tensors.
 * <p>The {@link #data} of an AbstractDenseRingTensor are mutable but the {@link #shape} is fixed.
 *
 * @param <T> The type of this dense tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}. This type parameter is required because some ops (e.g.
 * {@link #toCoo()}) may result in a sparse tensor.
 * @param <V> The type of the {@link Ring} which this tensor's data belong to.
 */
public abstract class AbstractDenseRingTensor<T extends AbstractDenseRingTensor<T, V>, V extends Ring<V>>
        extends AbstractDenseSemiringTensor<T, V>
        implements RingTensorMixin<T, T, V> {

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseRingTensor(Shape shape, V[] data) {
        super(shape, data);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        V[] diff = makeEmptyDataArray(data.length);
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, diff);
        return makeLikeNDArray(shape, diff);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
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
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensorHermitian(shape, data, axis1, axis2, dest);
        return makeLikeNDArray(shape.swapAxes(axis1, axis2), dest);
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
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensorHermitian(shape, data, axes, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return shape.getNdIndices(CompareRing.argminAbs(data));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return shape.getNdIndices(CompareRing.argmaxAbs(data));
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return CompareRing.minAbs(data);
    }


    /**
     * Finds the maximum absolute value in this tensor.
     *
     * @return The maximum absolute value in this tensor.
     */
    @Override
    public double maxAbs() {
        return CompareRing.maxAbs(data);
    }
}
