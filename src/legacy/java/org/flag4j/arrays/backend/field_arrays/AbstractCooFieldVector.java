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

package org.flag4j.arrays.backend.field_arrays;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.SparseValidation;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;


/**
 * <p>A sparse vector stored in coordinate list (COO) format. The {@link #data} of this COO vector are
 * elements of a {@link FieldElement}.
 *
 * <p>The {@link #data non-zero items} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero items is fixed.
 *
 * <p>Sparse vectors allow for the efficient storage of and ops on large vectors that contain many zero values.
 *
 * <p>COO vectors are optimized for large hyper-sparse vectors (i.e., vectors which contain almost all zeros relative to the size of the
 * vector).
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #data} of the vector. All other items in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the items of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.
 *
 * @param <T> Type of this vector.
 * @param <U> Type of equivalent dense vector.
 * @param <V> Type of matrix equivalent to {@code T}.
 * @param <W> Type of dense matrix equivalent to {@code U}.
 * @param <Y> Type of the field element in this vector.
 */
public abstract class AbstractCooFieldVector<
        T extends AbstractCooFieldVector<T, U, V, W, Y>,
        U extends AbstractDenseFieldVector<U, W, Y>,
        V extends AbstractCooFieldMatrix<V, W, T, Y>,
        W extends AbstractDenseFieldMatrix<W, U, Y>,
        Y extends FieldElement<Y>>
        extends AbstractCooRingVector<T, U, V, W, Y>
        implements FieldTensorMixin<T, U, Y>, VectorMixin<T, V, W, Y> {


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero entries of the tensor.
     * @param indices Non-zero entries of the tensor.
     */
    protected AbstractCooFieldVector(Shape shape, Y[] data, int[] indices) {
        super(shape, data, indices);
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);  // Validate parameters.
    }


    /**
     * Creates a tensor with the specified items and shape without performing <em>any</em> validation on the parameters.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero entries of the tensor.
     * @param indices Non-zero entries of the tensor.
     * @param dummy Dummy object to distinguish this constructor from the safe variant.
     */
    protected AbstractCooFieldVector(Shape shape, Y[] data, int[] indices, Object dummy) {
        super(shape, data, indices, dummy);
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CooVector abs() {
        double[] abs = new double[data.length];
        RingOps.abs(data, abs);
        return CooVector.unsafeMake(getShape(), abs, indices.clone());
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
        return T(axes);
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     * <p><b>WARNING</b>: This method is not supported for sparse tensors. If called on a sparse tensor,
     * an {@link UnsupportedOperationException} will be thrown. Element-wise division is undefined for sparse matrices as it
     * would almost certainly result in a division by zero.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     * @throws UnsupportedOperationException if this method is ever invoked on a sparse tensor.
     */
    @Override
    public T div(T b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse vectors.");
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        Y[] dest = makeEmptyDataArray(data.length);
        FieldOps.sqrt(data, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values; {@code false} otherwise.
     *
     * @see #containsInf()
     * @see #containsNaN()
     */
    @Override
    public boolean isAllFinite() {
        return FieldOps.isFinite(data);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value; {@code false} otherwise.
     *
     * @see #isAllFinite()
     * @see #containsNaN()
     */
    @Override
    public boolean containsInf() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value; {@code false} otherwise.
     *
     * @see #isAllFinite()
     * @see #containsInf()
     */
    @Override
    public boolean containsNaN() {
        return FieldOps.isAllNaN(data);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public T normalize() {
        return div(mag());
    }
}
