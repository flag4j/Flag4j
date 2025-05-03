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

package org.flag4j.arrays.backend.semiring_arrays;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.sparse.SparseValidation;
import org.flag4j.linalg.ops.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.ops.sparse.SparseElementSearch;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.coo.*;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringTensorOps;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;

/**
 * <p>Base class for all sparse tensors stored in coordinate list (COO) format. The data of this COO tensor are elements of a
 * {@link Semiring}
 *
 * <p>The {@link #data non-zero data} and {@link #indices non-zero indices} of a COO tensor are mutable but the {@link #shape}
 * and total {@link #nnz number of non-zero data} is fixed.
 *
 * <p>Sparse tensors allow for the efficient storage of and ops on tensors that contain many zero values.
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e., tensors which contain almost all zeros relative to the size of the
 * tensor).
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #data} of the tensor. All other data in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #data}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many ops assume indices to be sorted in a
 *     row-major format (i.e., last index increased fastest) but often this is not explicitly verified.
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero data in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the nD
 *     index of {@code data[i]}.
 *     </li>
 * </ul>
 *
 * @param <T> Type of this sparse COO tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. This type parameter is required because some ops (e.g.
 * {@link #tensorDot(AbstractCooSemiringTensor, int[], int[])} ) between two sparse tensors results in a dense tensor.
 * @param <V> Type of the {@link Semiring} which the data of this tensor belong to.
 */
public abstract class AbstractCooSemiringTensor<T extends AbstractCooSemiringTensor<T, U, V>,
        U extends AbstractDenseSemiringTensor<U, V>, V extends Semiring<V>>
        extends AbstractTensor<T, V[], V>
        implements SemiringTensorMixin<T, T, V> {

    /**
     * The zero-element for the semiring that this tensor's elements belong to.
     */
    protected V zeroElement;
    /**
     * <p>The non-zero indices of this sparse tensor.
     *
     * <p>Has shape {@code (nnz, rank)} where {@code nnz} is the number of non-zero data in this sparse tensor.
     */
    public final int[][] indices;
    /**
     * The number of non-zero data in this sparse tensor.
     */
    public final int nnz;
    /**
     * Stores the sparsity of this matrix.
     */
    private double sparsity = -1.0;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero data of this tensor of this tensor. If this tensor is dense, this specifies all data within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractCooSemiringTensor(Shape shape, V[] data, int[][] indices) {
        super(shape, data);
        SparseValidation.validateCoo(shape, data.length, indices);

        this.indices = indices;
        this.nnz = data.length;

        // Attempt to set the zero-element for the semiring.
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing COO tensors.
     * @param shape The shape of the tensor to construct.
     * @param data The non-zero data of this tensor.
     * @param indices The indices of the non-zero data.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    protected AbstractCooSemiringTensor(Shape shape, V[] data, int[][] indices, Object dummy) {
        super(shape, data);

        this.indices = indices;
        this.nnz = data.length;

        // Attempt to set the zero-element for the semiring.
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
    }

    /**
     * Gets the size of the 1D data array backing this tensor.
     *
     * @return The size of the 1D data array backing this tensor.
     */
    @Override
    public int dataLength() {
        return data.length;
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     * @param shape Shape of the tensor to construct.
     * @param data Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    public abstract T makeLikeTensor(Shape shape, V[] data, int[][] indices);


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     * @param shape Shape of the tensor to construct.
     * @param data Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    public abstract T makeLikeTensor(Shape shape, List<V> data, List<int[]> indices);


    /**
     * Constructs a dense tensor that is a similar type as this sparse COO tensor.
     * @param shape Shape of the tensor to construct.
     * @param entries The data of the dense tensor to construct.
     * @return A dense tensor that is a similar type as this sparse COO tensor.
     */
    public abstract U makeLikeDenseTensor(Shape shape, V[] entries);


    /**
     * Gets the zero-element for this tenor's field.
     * @return The zero-element for this tenor's field. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Semiring)} then {@code null} will be returned.
     */
    public V getZeroElement() {
        return zeroElement;
    }


    /**
     * Sets the zero-element for this tenor's field.
     * @param zeroElement The zero-element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the semiring.
     */
    public void setZeroElement(V zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }

    /**
     * Gets the sparsity of this tensor as a decimal percentage.
     * That is, the percentage of data in this tensor that are zero.
     * @return The sparsity of this tensor as a decimal percentage.
     * @see #getDensity()
     */
    public double getSparsity() {
        // Check if the sparsity has already been computed.
        if (this.sparsity < 0)
            this.sparsity = SparseUtils.computeSparsity(shape, nnz);

        return sparsity;
    }


    /**
     * Gets the density of this tensor as a decimal percentage.
     * That is, the percentage of data in this tensor that are non-zero.
     * @return The density of this tensor as a decimal percentage.
     * @see #getSparsity()
     */
    public double getDensity() {
        return 1.0 - sparsity;
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        SparseTensorData<V> data = CooSemiringTensorOps.add(
                shape, this.data, indices,
                b.shape, b.data, b.indices
        );

        return makeLikeTensor(data.shape(), data.data(), data.indices());
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T elemMult(T b) {
        SparseTensorData<V> data = CooSemiringTensorOps.elemMult(
                shape, this.data, indices, b.shape, b.data, b.indices);
        return makeLikeTensor(data.shape(), data.data(), data.indices());
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 Tensor to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public U tensorDot(T src2, int[] aAxes, int[] bAxes) {
        CooTensorDot<V> problem = new CooTensorDot<>(shape, data, indices,
                src2.shape, src2.data, src2.indices,
                aAxes, bAxes);
        V[] dest= makeEmptyDataArray(problem.getOutputSize());
        problem.compute(dest);
        return makeLikeDenseTensor(problem.getOutputShape(), dest);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values in the 2D subarrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param axis1 First axis for 2D subarray.
     * @param axis2 Second axis for 2D subarray.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e., the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        SparseTensorData<V> data = CooSemiringTensorOps.tensorTr(
                shape, this.data, indices, axis1, axis2);
        return makeLikeTensor(data.shape(), data.data(), data.indices());
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public T T() {
        V[] destEntries= makeEmptyDataArray(nnz);
        int[][] destIndices = new int[nnz][rank];
        CooTranspose.tensorTranspose(shape, data, indices,0, shape.getRank()-1, destEntries, destIndices);
        return makeLikeTensor(shape.swapAxes(0, rank-1), destEntries, destIndices);
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public T T(int axis1, int axis2) {
        V[] destEntries= makeEmptyDataArray(nnz);
        int[][] destIndices = new int[nnz][rank];
        CooTranspose.tensorTranspose(shape, data, indices, axis1, axis2, destEntries, destIndices);
        return makeLikeTensor(shape.swapAxes(axis1, axis2), destEntries, destIndices);
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public T T(int... axes) {
        V[] destEntries= makeEmptyDataArray(nnz);
        int[][] destIndices = new int[nnz][rank];
        CooTranspose.tensorTranspose(shape, data, indices, axes, destEntries, destIndices);
        return makeLikeTensor(shape.permuteAxes(axes), destEntries, destIndices);
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, data.clone());
    }


    /**
     * Finds the minimum (non-zero) value in this tensor. If this tensor is complex, then this method finds the smallest value in
     * magnitude.
     *
     * @return The minimum (non-zero) value in this tensor.
     */
    @Override
    public V min() {
        return CompareSemiring.min(data);
    }


    /**
     * Finds the maximum (non-zero) value in this tensor.
     *
     * @return The maximum (non-zero) value in this tensor.
     */
    @Override
    public V max() {
        return CompareSemiring.max(data);
    }


    /**
     * Finds the indices of the minimum (non-zero) value in this tensor.
     *
     * @return The indices of the minimum (non-zero) value in this tensor.
     */
    @Override
    public int[] argmin() {
        return indices[CompareSemiring.argmin(data)];
    }


    /**
     * Finds the indices of the maximum (non-zero) value in this tensor.
     *
     * @return The indices of the maximum (non-zero) value in this tensor.
     */
    @Override
    public int[] argmax() {
        return indices[CompareSemiring.argmin(data)];
    }


    /**
     * Gets the element of this tensor at the specified target.
     *
     * @param target Index of the element to get.
     *
     * @return The element of this tensor at the specified index. If there is a non-zero value with the specified index, that value
     * will be returned. If there is no non-zero value at the specified index than the zero-element will attempt to be
     * returned (i.e., the additive identity of the semiring). However, if the zero-element could not be determined during
     * construction or if it was not set with {@link #setZeroElement(Semiring)} then
     * {@code null} will be returned.
     *
     * @throws ArrayIndexOutOfBoundsException If any target are not within this tensor.
     */
    @Override
    public V get(int... target) {
        ValidateParameters.validateTensorIndex(shape, target);
        V value = CooGetSet.getCoo(data, indices, target);
        return (value == null) ? getZeroElement() : value;
    }


    /**
     * Sets the element of this tensor at the specified target.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param target Index of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code target} is not within the bounds of this tensor.
     */
    @Override
    public T set(V value, int... target) {
        ValidateParameters.validateTensorIndex(shape, target);
        int idx = SparseElementSearch.binarySearchCoo(indices, target);

        V[] destEntries;
        int[][] destIndices;

        if (idx >= 0) {
            // Target index found.
            destEntries = data.clone();
            destIndices = ArrayUtils.deepCopy2D(indices, null);
            destEntries[idx] = value;
            destIndices[idx] = target;
        } else {
            // Target not found, insert new value and index.
            destEntries= makeEmptyDataArray(nnz + 1);
            destIndices = new int[nnz + 1][rank];
            int insertionPoint = - (idx + 1);
            CooGetSet.cooInsertNewValue(value, target, data, indices, insertionPoint, destEntries, destIndices);
        }

        return makeLikeTensor(shape, destEntries, destIndices);
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return makeLikeTensor(
                shape.flatten(),
                data.clone(),
                SparseUtils.cooFlattenIndices(shape, indices));
    }


    /**
     * Flattens a tensor along the specified axis. Unlike {@link #flatten()}
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        int[] destShape = new int[indices[0].length];
        Arrays.fill(destShape, 1);
        destShape[axis] = shape.totalEntries().intValueExact();

        return makeLikeTensor(
                new Shape(destShape),
                data.clone(),
                SparseUtils.cooFlattenIndices(shape, indices, axis));
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws TensorShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        return makeLikeTensor(newShape, data.clone(), SparseUtils.cooReshape(shape, newShape, indices));
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        CooDataSorter.wrap(data, indices).sparseSort().unwrap(data, indices);
    }


    /**
     * Converts this COO tensor to an equivalent dense tensor.
     * @return A dense tensor which is equivalent to this COO tensor.
     * @throws ArithmeticException If the number of data in the dense tensor exceeds 2,147,483,647.
     */
    public U toDense() {
        V[] denseEntries= makeEmptyDataArray(shape.totalEntriesIntValueExact());
        CooConversions.toDense(shape, data, indices, denseEntries);
        return makeLikeDenseTensor(shape, denseEntries);
    }


    /**
     * Coalesces this sparse COO tensor. An uncoalesced tensor is a sparse tensor with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by summing duplicated data. If another form of aggregation other
     * than summation is desired, use {@link #coalesce(BinaryOperator)}.
     * @return A new coalesced sparse COO tensor which is equivalent to this COO tensor.
     * @see #coalesce(BinaryOperator)
     */
    public T coalesce() {
        SparseTensorData<V> tensor = SparseUtils.coalesce(Semiring::add, shape, data, indices);
        return makeLikeTensor(tensor.shape(), tensor.data(), tensor.indices());
    }


    /**
     * Coalesces this sparse COO tensor. An uncoalesced tensor is a sparse tensor with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data using {@code aggregator}.
     * @param aggregator Custom aggregation function to combine multiple.
     * @return A new coalesced sparse COO tensor which is equivalent to this COO tensor.
     * @see #coalesce()
     */
    public T coalesce(BinaryOperator<V> aggregator) {
        SparseTensorData<V> tensor = SparseUtils.coalesce(aggregator, shape, data, indices);
        return makeLikeTensor(tensor.shape(), tensor.data(), tensor.indices());
    }


    /**
     * Drops any explicit zeros in this sparse COO tensor.
     * @return A copy of this COO tensor with any explicitly stored zeros removed.
     */
    public T dropZeros() {
        SparseTensorData<V> tensor = SparseUtils.dropZeros(shape, data, indices);
        return makeLikeTensor(tensor.shape(), tensor.data(), tensor.indices());
    }
}
