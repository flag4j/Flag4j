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
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringTensor;
import org.flag4j.linalg.ops.sparse.coo.ring_ops.CooRingTensorOps;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * <p>Base class for all sparse {@link RingElement} tensors stored in coordinate list (COO) format. The items of this COO tensor are
 * elements of a {@link RingElement}.
 *
 * <p>The {@link #data non-zero items} and {@link #indices non-zero indices} of a COO tensor are mutable but the {@link #shape}
 * and total {@link #nnz number of non-zero items} is fixed.
 *
 * <p>Sparse tensors allow for the efficient storage of and ops on tensors that contain many zero values.
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e., tensors which contain almost all zeros relative to the size of the
 * tensor).
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #data} of the tensor. All other items in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #data}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many ops assume indices to be sorted in a
 *     row-major format (i.e., last index increased fastest) but often this is not explicitly verified.
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero items in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the nD
 *     index of {@code items[i]}.
 *     </li>
 * </ul>
 *
 * @param <T> Type of this sparse COO tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. This type parameter is required because some ops (e.g.
 * {@link AbstractCooSemiringTensor#tensorDot(AbstractCooSemiringTensor, int[], int[])} ) between two sparse tensors results in a dense
 * tensor.
 * @param <V> Type of the {@link RingElement} which the items of this tensor belong to.
 */
public abstract class AbstractCooRingTensor<T extends AbstractCooRingTensor<T, U, V>,
        U extends AbstractDenseRingTensor<U, V>, V extends RingElement<V>>
        extends AbstractCooSemiringTensor<T, U, V>
        implements RingTensorMixin<T, T, V> {


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     */
    protected AbstractCooRingTensor(Shape shape, V[] data, int[][] indices) {
        super(shape, data, indices);
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing COO tensors.
     * @param shape The shape of the tensor to construct.
     * @param data The Non-zero items in this tensor.
     * @param indices The indices of the non-zero items.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    protected AbstractCooRingTensor(Shape shape, V[] data, int[][] indices, Object dummy) {
        super(shape, data, indices, dummy);
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
        SparseTensorData<V> diff = CooRingTensorOps.sub(
                shape, data, indices,
                b.shape, b.data, b.indices);
        V[] dest = makeEmptyDataArray(diff.data().size());
        diff.data().toArray(dest);
        return makeLikeTensor(diff.shape(),
                dest,
                diff.indices().toArray(new int[diff.indices().size()][]));
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
