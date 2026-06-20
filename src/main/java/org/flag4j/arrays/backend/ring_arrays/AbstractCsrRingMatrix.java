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
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCsrSemiringMatrix;
import org.flag4j.linalg.ops.sparse.csr.CsrOps;
import org.flag4j.linalg.ops.sparse.csr.ring_ops.CsrRingProperties;
import org.flag4j.numbers.Ring;
import org.flag4j.util.exceptions.NDArrayShapeException;


/**
 * <p>A sparse matrix stored in compressed sparse row (CSR) format. The {@link #data} of this CSR matrix are
 * elements of a {@link Ring}.
 *
 * <p>The {@link #data non-zero data} and non-zero indices of a CSR matrix are mutable but the {@link #shape}
 * and {@link #nnz total number of non-zero data} is fixed.
 *
 * <p>Sparse matrices allow for the efficient storage of and ops on matrices that contain many zero values.
 *
 * <p>A sparse CSR matrix is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #data} of the matrix. All other data in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #rowPointers row pointers} of the non-zero values in the CSR matrix. Has size {@link #numRows numRows + 1}
 *     <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all values in row</li>
 *     {@code i}.
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the data of the CSR matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any ops implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 *
 * @param <T> Type of this CSR field matrix.
 * @param <U> Type of dense field matrix equivalent to {@code T}.
 * @param <V> Type of vector equivalent to {@code V}.
 * @param <W> Type of field element of this matrix.
 */
public abstract class AbstractCsrRingMatrix<T extends AbstractCsrRingMatrix<T, U, V, W>,
        U extends AbstractDenseRingMatrix<U, ?, W>,
        V extends AbstractCooRingVector<V, ?, ?, U, W>,
        W extends Ring<W>>
        extends AbstractCsrSemiringMatrix<T, U, V, W>
        implements RingTensorMixin<T, U, W>, MatrixMixin<T, U, V, W> {

    /**
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero data, row pointers, and non-zero column indices.
     *
     * @param shape Shape of this tensor.
     * @param data The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     */
    protected AbstractCsrRingMatrix(Shape shape, W[] data, int[] rowPointers, int[] colIndices) {
        super(shape, data, rowPointers, colIndices);
    }


    /**
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero data, row pointers, and non-zero column indices.
     *
     * @param shape Shape of this tensor.
     * @param data The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    protected AbstractCsrRingMatrix(Shape shape, W[] data, int[] rowPointers, int[] colIndices, Object dummy) {
        super(shape, data, rowPointers, colIndices, dummy);
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
        SparseMatrixData<W> destData = CsrOps.applyBinOpp(
                shape,  data, rowPointers, colIndices,
                b.shape,  b.data, b.rowPointers, b.colIndices,
                Ring::add, Ring::addInv);

        return makeLikeTensor(shape, destData.data(), destData.rowData(), destData.colData());
    }


    /**
     * <p>Warning: throws {@link UnsupportedOperationException} as division is not defined for general ring matrices.
     *
     * <p>{@inheritDoc}
     */
    @Override
    public T div(T b) {
        throw new UnsupportedOperationException("Division not supported for matrix type: " + getClass().getName());
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        return T();
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


    /**
     * Checks if two sparse CSR ring matrices are element-wise equal within the following tolerance for two entries {@code x}
     * and {@code y}:
     * <pre>{@code
     *  |x-y| <= (1e-08 + 1e-05*|y|)
     * }</pre>
     *
     * To specify the relative and absolute tolerances use {@link #allClose(AbstractCsrRingMatrix, double, double)}
     *
     * @return {@code true} if this matrix and {@code b} element-wise equal within the tolerance {@code |x-y| <= (1e-08 + 1e-05*|y|)}.
     * @see #allClose(AbstractCsrRingMatrix, double, double)
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
     * @return {@code true} if the {@code src1} matrix is the same shape as the {@code src2} matrix and all data
     * are "close", i.e., elements {@code a} and {@code b} at the same positions in the two matrices respectively
     * satisfy {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns {@code false}.
     * @see #allClose(AbstractCsrRingMatrix)
     */
    public boolean allClose(T b, double relRol, double absTol) {
        return CsrRingProperties.allClose(this, b, relRol, absTol);
    }
}
