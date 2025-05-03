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
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.ring_arrays.AbstractCsrRingMatrix;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCsrSemiringMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.sparse.csr.CsrOps;
import org.flag4j.linalg.ops.sparse.csr.ring_ops.CsrRingProperties;
import org.flag4j.linalg.ops.sparse.csr.semiring_ops.SemiringCsrMatMult;
import org.flag4j.numbers.Field;

public abstract class AbstractCsrFieldMatrix<T extends AbstractCsrFieldMatrix<T, U, V, W>,
        U extends AbstractDenseFieldMatrix<U, ?, W>,
        V extends AbstractCooFieldVector<V, ?, ?, U, W>,
        W extends Field<W>>
        extends AbstractCsrRingMatrix<T, U, V, W>
        implements FieldTensorMixin<T, U, W>, MatrixMixin<T, U, V, W> {


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
    protected AbstractCsrFieldMatrix(Shape shape, W[] data, int[] rowPointers, int[] colIndices) {
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
    protected AbstractCsrFieldMatrix(Shape shape, W[] data, int[] rowPointers, int[] colIndices, Object dummy) {
        super(shape, data, rowPointers, colIndices, dummy);
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices and stores the result in a sparse matrix.
     * <p>Warning: this method should be used with caution as sparse-sparse matrix multiplication may result in a dense matrix.
     * In such a case, this method will likely be significantly slower than {@link #mult(AbstractCsrSemiringMatrix)}.
     * @param b Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code b} as a sparse CSR matrix.
     * @see #mult(AbstractCsrSemiringMatrix)
     */
    public T mult2Csr(T b) {
        SparseMatrixData<W> data = SemiringCsrMatMult.standardToSparse(
                shape, this.data, rowPointers, colIndices, b.shape,
                b.data, b.rowPointers, b.colIndices);

        return makeLikeTensor(data.shape(), data.data(), data.rowData(), data.colData());
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    @Override
    public boolean isHermitian() {
        // For a field matrix, same as isSymmetric.
        return CsrRingProperties.isHermitian(shape, data, rowPointers, colIndices);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return {@code true} if this matrix it is orthogonal; {@code false} otherwise.
     */
    public boolean isUnitary() {
        if(isSquare()) return mult(H()).isI();
        else return false;
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CsrMatrix abs() {
        double[] abs = new double[data.length];
        RingOps.abs(data, abs);
        return new CsrMatrix(getShape(), abs, rowPointers.clone(), colIndices.clone());
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        W[] dest = makeEmptyDataArray(data.length);
        int[] destRowPointers = new int[numCols+1];
        int[] destColIndices = new int[data.length];
        CsrOps.hermTranspose(data, rowPointers, colIndices, dest, destRowPointers, destColIndices);

        return makeLikeTensor(shape.swapAxes(0, 1), dest, destRowPointers, destColIndices);
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
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
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
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse matrices.");
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        W[] dest = makeEmptyDataArray(data.length);
        FieldOps.sqrt(data, dest);
        return makeLikeTensor(shape, dest);
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
}
