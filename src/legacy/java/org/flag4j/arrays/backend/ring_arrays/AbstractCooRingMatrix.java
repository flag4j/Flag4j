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
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringMatrix;
import org.flag4j.linalg.ops.sparse.coo.CooConversions;
import org.flag4j.linalg.ops.sparse.coo.ring_ops.CooRingMatrixOps;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * <p>A sparse matrix stored in coordinate list (COO) format. The {@link #data} of this COO matrix are
 * elements of a {@link RingElement}.
 *
 * <p>The {@link #data non-zero items} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero items is fixed.
 *
 * <p>Sparse matrices allow for the efficient storage of and ops on matrices that contain many zero values.
 *
 * <p>COO matrices are optimized for hyper-sparse matrices (i.e., matrices which contain almost all zeros relative to the size of the
 * matrix).
 *
 * <h2>COO Representation:</h2>
 * A sparse COO matrix is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #data} of the matrix. All other items in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the items of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified but any ops implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 *
 * @param <T> Type of this sparse COO matrix.
 * @param <U> Type of dense matrix which is similar to {@code T}.
 * @param <V> Type of sparse COO vector which is similar to {@code T}.
 * @param <W> Type of the ndarrays element in this matrix.
 */
public abstract class AbstractCooRingMatrix<T extends AbstractCooRingMatrix<T, U, V, W>,
        U extends AbstractDenseRingMatrix<U, ?, W>,
        V extends AbstractCooRingVector<V, ?, T, U, W>,
        W extends RingElement<W>>
        extends AbstractCooSemiringMatrix<T, U, V, W>
        implements RingTensorMixin<T, U, W>, MatrixMixin<T, U, V, W> {


    /**
     * Creates a sparse coo matrix with the specified non-zero items, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero items of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    protected AbstractCooRingMatrix(Shape shape, W[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero items, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero items of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant.
     */
    protected AbstractCooRingMatrix(Shape shape, W[] entries, int[] rowIndices, int[] colIndices, Object dummy) {
        super(shape, entries, rowIndices, colIndices, dummy);
    }


    /**
     * Constructs a sparse CSR matrix of a similar type to this sparse COO matrix.
     * @param shape Shape of the CSR matrix to construct.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A CSR matrix of a similar type to this sparse COO matrix.
     */
    public abstract AbstractCsrRingMatrix<?, U, V, W> makeLikeCsrMatrix(
            Shape shape, W[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    public AbstractCsrRingMatrix<?, U, V, W> toCsr() {
        W[] csrEntries = (W[]) new RingElement[data.length];
        int[] csrRowPointers = new int[numRows + 1];
        int[] csrColPointers = new int[colIndices.length];
        CooConversions.toCsr(shape, data, rowIndices, colIndices, csrEntries, csrRowPointers, csrColPointers);
        return makeLikeCsrMatrix(shape, csrEntries, csrRowPointers, csrColPointers);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public abstract AbstractCooRingTensor<?, ?, W> toTensor();


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as
     * {@link #shape this.shape}.
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    public abstract AbstractCooRingTensor<?, ?, W> toTensor(Shape newShape);


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
        SparseMatrixData<W> data = CooRingMatrixOps.sub(
                shape, this.data, rowIndices, colIndices,
                b.shape, b.data, b.rowIndices, b.colIndices);

        return makeLikeNDArray(data.shape(), data.data(), data.rowData(), data.colData());
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
     * Checks if the matrix is "close" to an identity matrix. Two entries {@code x} and {@code y} are considered
     * "close" if they satisfy the following:
     * <pre>{@code
     *      |x-y| <= (1E-08 + 1E-05*|y|)
     * }</pre>
     *
     * @return {@code true} if the matrix is approximately an identity matrix, otherwise {@code false}.
     */
    public boolean isCloseToI() {
        return CooRingMatrixOps.isCloseToIdentity(this);
    }
}
