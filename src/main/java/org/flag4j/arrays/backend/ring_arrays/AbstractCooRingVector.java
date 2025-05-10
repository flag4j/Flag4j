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
import org.flag4j.arrays.SparseVectorData;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringVector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.sparse.coo.ring_ops.CooRingVectorOps;
import org.flag4j.numbers.Ring;
import org.flag4j.util.exceptions.ArrayShapeException;

/**
 * <p>A sparse vector stored in coordinate list (COO) format. The {@link #data} of this COO vector are
 * elements of a {@link Ring}.
 *
 * <p>The {@link #data non-zero data} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero data is fixed.
 *
 * <p>Sparse vectors allow for the efficient storage of and ops on large vectors that contain many zero values.
 *
 * <p>COO vectors are optimized for large hyper-sparse vectors (i.e., vectors which contain almost all zeros relative to the size of the
 * vector).
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #data} of the vector. All other data in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the data of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.
 *
 * @param <T> Type of this vector.
 * @param <U> Type of equivalent dense vector.
 * @param <V> Type of matrix equivalent to {@code T}.
 * @param <W> Type of dense matrix equivalent to {@code U}.
 * @param <Y> Type of the array elements in this vector.
 */
public abstract class AbstractCooRingVector<
        T extends AbstractCooRingVector<T, U, V, W, Y>,
        U extends AbstractDenseRingVector<U, W, Y>,
        V extends AbstractCooRingMatrix<V, W, T, Y>,
        W extends AbstractDenseRingMatrix<W, U, Y>,
        Y extends Ring<Y>>
        extends AbstractCooSemiringVector<T, U, V, W, Y>
        implements RingTensorMixin<T, U, Y>, VectorMixin<T, V, W, Y> {


    /**
     * Creates a COO vector with the specified data and shape.
     *
     * @param shape The full shape of the vector.
     * @param entries The non-zero entries of the COO vector.
     * @param indices The non-zero indices of the COO vector.
     */
    protected AbstractCooRingVector(Shape shape, Y[] entries, int[] indices) {
        super(shape, entries, indices);
    }


    /**
     * Creates a tensor with the specified data and shape without performing <em>any</em> validation on the parameters.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero entries of the tensor.
     * @param indices Non-zero entries of the tensor.
     * @param dummy Dummy object to distinguish this constructor from the safe variant.
     */
    protected AbstractCooRingVector(Shape shape, Y[] data, int[] indices, Object dummy) {
        super(shape, data, indices, dummy);
    }


    /**
     * <p>Computes the inner product between two vectors.
     *
     * <p>Note: this method is distinct from {@code dot(AbstractCooSemiringVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     */
    @Override
    public Y inner(T b) {
        return CooRingVectorOps.inner(this, b);
    }


    /**
     * Computes the norm of this vector. This is the same as {@link #mag()}.
     *
     * @return The norm (specifically &ell;<sup>2</sup>) of this vector.
     *
     * @see #mag()
     * @see #magSquared()
     */
    @Override
    public double norm() {
        return VectorNorms.norm(data) ;
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
        return VectorNorms.normSquared(data);
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
        SparseVectorData<Y> result = CooRingVectorOps.sub(
                shape, data, indices, b.shape, b.data, b.indices);
        return makeLikeTensor(shape, result.data(), result.indices());
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
        return T(axis1, axis2);
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
}
