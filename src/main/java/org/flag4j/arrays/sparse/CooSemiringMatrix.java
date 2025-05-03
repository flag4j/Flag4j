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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringMatrix;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCsrSemiringMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.SemiringMatrix;
import org.flag4j.arrays.dense.SemiringTensor;
import org.flag4j.arrays.dense.SemiringVector;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatMult;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.List;
import java.util.function.BinaryOperator;

/**
 * Represents a sparse matrix whose non-zero-elements are stored in Coordinate List (COO) format, with all data elements
 * belonging to a specified {@link Semiring} type.
 *
 * <p>The COO format stores sparse matrix data as a list of coordinates (row and column indices) coupled with their
 * corresponding non-zero values, rather than allocating memory for every element in the full matrix shape. This
 * allows efficient representation and manipulation of large matrices containing a substantial number of zeros.
 *
 * <h2>COO Representation:</h2>
 * A sparse COO matrix is stored as:
 * <ul>
 *     <li><b>Shape:</b> The full {@link #shape} of the matrix specifying its number of rows and columns.</li>
 *
 *     <li><b>Data:</b> Non-zero values are stored in a one-dimensional array, {@link #data}. Any element not specified in
 *     {@code data} is implicitly zero. It is also possible to explicitly store zero values in this array, although this
 *     is generally not desirable. To remove explicitly defined zeros, use {@link #dropZeros()}.</li>
 *
 *     <li><b>Indices:</b> Non-zero values are associated with their coordinates in the matrix via two parallel 1D arrays:
 *     {@link #rowIndices} and {@link #colIndices}. These arrays specify the row and column positions of each
 *     non-zero entry in {@link #data}. The total number of non-zero-elements is given by {@link #nnz}.
 *     Each pair of indices corresponds directly to the position of a single non-zero value in {@link #data}.</li>
 * </ul>
 *
 * <p>The total number of non-zero-elements ({@link #nnz}) and the shape are fixed for a given instance, but the values
 * in {@link #data} and their corresponding {@link #rowIndices} and {@link #colIndices} may be updated. Many operations
 * assume that the indices are sorted lexicographically by row, and then by column, but this is not strictly enforced.
 * All provided operations preserve the lexicographical sorting of indices. If there is any doubt about the ordering of
 * indices, use {@link #sortIndices()} to ensure they are explicitly sorted. COO tensors may also store multiple entries
 * for the same index (referred to as an uncoalesced tensor). To combine all duplicated entries use {@link #coalesce()} or
 * {@link #coalesce(BinaryOperator)}.
 *
 * <p>COO matrices are optimized for "hyper-sparse" scenarios where the proportion of non-zero-elements is extremely low,
 * offering significant memory savings and potentially more efficient computational operations than equivalent dense
 * representations.
 *
 * <pre>{@code
 * // shape, data, and indices for COO matrix.
 * Shape shape = new Shape(512, 1024);
 * BoolSemiring[] data = {
 *      new BoolSemiring(true), new BoolSemiring(false), new BoolSemiring(false),
 *      new BoolSemiring(true), new BoolSemiring(true),  new BoolSemiring(false),
 * };
 * int[] rowIndices = {0, 4, 128, 128, 128, 256};
 * int[] colIndices = {16, 2, 5, 512, 1000, 28};
 *
 * // Create COO matrix.
 * CooSemiringMatrix<BoolSemiring> matrix = new CooSemiringMatrix(
 *      shape, data, rowIndices, colIndices
 * );
 *
 * // Sum matrices.
 * CooSemiringMatrix<BoolSemiring> sum = matrix.add(matrix);
 *
 * // Multiply matrix to it's transpose.
 * SemiringMatrix<BoolSemiring> prod = matrix.mult(matrix.T());
 * prod = matrix.multTranspose(matrix);
 * }</pre>
 *
 * @param <T> The type of elements stored in this matrix, constrained by the {@link Semiring} interface.
 * @see CooSemiringTensor
 * @see CooSemiringVector
 * @see SemiringMatrix
 * @see Semiring
 */
public class CooSemiringMatrix<T extends Semiring<T>> extends AbstractCooSemiringMatrix<
        CooSemiringMatrix<T>, SemiringMatrix<T>, CooSemiringVector<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooSemiringMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooSemiringMatrix(Shape shape, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape,
                (T[]) entries.toArray(new Semiring[entries.size()]),
                ArrayConversions.fromIntegerList(rowIndices),
                ArrayConversions.fromIntegerList(colIndices));
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param rows Rows in the coo matrix.
     * @param cols Columns in the coo matrix.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooSemiringMatrix(int rows, int cols, T[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param rows Rows in the coo matrix.
     * @param cols Columns in the coo matrix.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooSemiringMatrix(int rows, int cols, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(new Shape(rows, cols),
                (T[]) entries.toArray(new Semiring[entries.size()]),
                ArrayConversions.fromIntegerList(rowIndices),
                ArrayConversions.fromIntegerList(colIndices));
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing COO matrices.
     * @param shape The shape of the matrix to construct.
     * @param data The non-zero data of this COO matrix.
     * @param rowIndices The non-zero row indices of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooSemiringMatrix(Shape shape, T[] data, int[] rowIndices, int[] colIndices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO tensor in this manner is unsafe.
        super(shape, data, rowIndices, colIndices, dummy);
    }


    /**
     * <p>Factory to construct a COO matrix which bypasses any validation checks on the data and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param shape The full size of the COO matrix.
     * @param data The non-zero data of the COO matrix.
     * @param rowIndices The non-zero row indices of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @return A COO matrix constructed from the provided parameters.
     */
    public static <T extends Semiring<T>> CooSemiringMatrix<T> unsafeMake(
            Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        return new CooSemiringMatrix(shape, data, rowIndices, colIndices, null);
    }



    /**
     * Constructs a sparse COO tensor of the same type as this tensor with the specified non-zero data and indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     *
     * @return A sparse COO tensor of the same type as this tensor with the specified non-zero data and indices.
     */
    @Override
    public CooSemiringMatrix<T> makeLikeTensor(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooSemiringMatrix<>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     *
     * @return A COO matrix with the specified shape, non-zero data, and non-zero indices.
     */
    @Override
    public CooSemiringMatrix<T> makeLikeTensor(Shape shape, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooSemiringMatrix<>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a sparse COO vector of a similar type to this COO matrix.
     *
     * @param shape Shape of the vector. Must be rank 1.
     * @param entries Non-zero data of the COO vector.
     * @param indices Non-zero indices of the COO vector.
     *
     * @return A sparse COO vector of a similar type to this COO matrix.
     */
    @Override
    public CooSemiringVector<T> makeLikeVector(Shape shape, T[] entries, int[] indices) {
        return new CooSemiringVector<>(shape, entries, indices);
    }


    /**
     * Constructs a dense tensor with the specified {@code shape} and {@code data} which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     *
     * @return A dense tensor with the specified {@code shape} and {@code data} which is a similar type to this sparse tensor.
     */
    @Override
    public SemiringMatrix<T> makeLikeDenseTensor(Shape shape, T[] entries) {
        return new SemiringMatrix<>(shape, entries);
    }


    /**
     * Constructs a sparse CSR matrix of a similar type to this sparse COO matrix.
     *
     * @param shape Shape of the CSR matrix to construct.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A CSR matrix of a similar type to this sparse COO matrix.
     */
    @Override
    public AbstractCsrSemiringMatrix<?, SemiringMatrix<T>, CooSemiringVector<T>, T> makeLikeCsrMatrix(
            Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrSemiringMatrix<>(shape, entries, rowPointers, colIndices);
    }


    /**
     * Converts this matrix to an equivalent rank 2 tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooSemiringTensor<T> toTensor() {
        int[][] tIndices = RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices});
        return new CooSemiringTensor<>(shape, data.clone(), tIndices);
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooSemiringTensor<T> toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CooSemiringMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooSemiringMatrix<>(shape, entries, rowIndices.clone(), colIndices.clone());
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
    public SemiringTensor<T> tensorDot(CooSemiringMatrix<T> src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of
     *                                {@code b}.
     */
    @Override
    public SemiringVector<T> mult(CooSemiringVector<T> b) {
        T[] dest = makeEmptyDataArray(numRows);
        CooSemiringMatMult.standardVector(data, rowIndices, colIndices, shape, b.data, b.indices, dest);
        return new SemiringVector<T>(dest);
    }


    /**
     * <p>Converts this sparse COO matrix to an equivalent compressed sparse row (CSR) matrix.
     * <p>It is often easier and more efficient to construct a matrix in COO format first then convert to a CSR matrix for efficient
     * computations.
     *
     * @return A CSR matrix equivalent to this COO matrix.
     */
    @Override
    public CsrSemiringMatrix<T> toCsr() {
        int[] rowPointers = new int[numRows + 1];

        // Count number of data per row.
        for(int i=0; i<nnz; i++)
            rowPointers[rowIndices[i] + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<numRows; i++)
            rowPointers[i + 1] += rowPointers[i];

        return new CsrSemiringMatrix<T>(shape, data.clone(), rowPointers, colIndices.clone());
    }


    /**
     * Accepts a visitor that implements the {@link MatrixVisitor} interface.
     * This method is part of the "Visitor Pattern" and allows operations to be performed
     * on the matrix without modifying the matrix's class directly.
     *
     * @param visitor The visitor implementing the operation to be performed.
     *
     * @return The result of the visitor's operation, typically another matrix or a scalar value.
     *
     * @throws NullPointerException if the visitor is {@code null}.
     */
    @Override
    public <R> R accept(MatrixVisitor<R> visitor) {
        return visitor.visit(this);
    }


    /**
     * {@inheritDoc}
     *
     * <p><b>Warning:</b> This method will throw a {@link UnsupportedOperationException} as subtraction is not supported for
     * semiring tensors.
     */
    @Override
    public CooSemiringMatrix<T> sub(CooSemiringMatrix<T> b) {
        throw new UnsupportedOperationException("Subtraction not supported for matrix type: " + getClass().getName());
    }


    /**
     * {@inheritDoc}
     *
     * <p><b>Warning:</b> This method will throw a {@link UnsupportedOperationException} as division is not supported for
     * semiring tensors.
     */
    @Override
    public CooSemiringMatrix<T> div(CooSemiringMatrix<T> b) {
        throw new UnsupportedOperationException("Division not supported for matrix type: " + getClass().getName());
    }
}
