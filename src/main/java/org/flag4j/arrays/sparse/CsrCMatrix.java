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
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.field_arrays.AbstractCsrFieldMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.*;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrMatMult;
import org.flag4j.linalg.ops.dense_sparse.csr.semiring_ops.DenseCsrSemiringMatMult;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.csr.CsrConversions;
import org.flag4j.linalg.ops.sparse.csr.real_complex.RealComplexCsrMatMult;
import org.flag4j.linalg.ops.sparse.csr.semiring_ops.SemiringCsrMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.List;
import java.util.function.BinaryOperator;

// TODO: update javadoc to be like that of CsrMatrix.java
/**
 * <p>Instances of this class represent a complex sparse matrix using the compressed sparse row (CSR) format.
 * This class is optimized for efficient storage and operations on matrices with a high proportion of zero-elements.
 * The non-zero values of the matrix are stored in a compact form, reducing memory usage and improving performance for many matrix
 * operations.
 *
 * <h2>CSR Representation:</h2>
 * A CSR matrix is represented internally using three main arrays:
 * <ul>
 *   <li><b>Data:</b> Non-zero values are stored in a one-dimensional array {@link #data} of length {@link #nnz}. Any element not
 *   specified in {@code data} is implicitly zero. It is also possible to explicitly store zero values in this array, although this
 *   is generally not desirable. To remove explicitly defined zeros, use {@link #dropZeros()}</li>
 *
 *   <li><b>Row Pointers:</b> A 1D array {@link #rowPointers} of length {@code numRows + 1} where {@code rowPointers[i]} indicates
 *   the starting index in the {@code data} and {@code colIndices} arrays for row {@code i}. The last entry of {@code rowPointers}
 *   equals the length of {@code data}. That is, all non-zero values in {@code data} which are in row {@code i} are between
 *   {@code data[rowIndices[i]} (inclusive) and {@code data[rowIndices[i + 1]} (exclusive).</li>
 *
 *   <li><b>Column Indices:</b> A 1D array {@link #colIndices} of length {@link #nnz} storing the column indices corresponding to each non-zero
 *   value in {@code data}.</li>
 * </ul>
 *
 * <p>The total number of non-zero-elements ({@link #nnz}) and the shape are fixed for a given instance, but the values
 * in {@link #data} and their corresponding {@link #rowPointers} and {@link #colIndices} may be updated. Many operations
 * assume that the indices are sorted lexicographically by row, and then by column, but this is not strictly enforced.
 * All provided operations preserve the lexicographical row-major sorting of data and indices. If there is any doubt about the
 * ordering of indices, use {@link #sortIndices()} to ensure they are explicitly sorted. CSR tensors may also store multiple entries
 * for the same index (referred to as an uncoalesced tensor). To combine all duplicated entries use {@link #coalesce()} or
 * {@link #coalesce(BinaryOperator)}.
 *
 * <p>CSR matrices are optimized for efficient storage and operations on matrices with a high proportion of zero-elements.
 * CSR matrices are ideal for row-wise operations and matrix-vector multiplications. In general, CSR matrices are not efficient at
 * handling many incremental updates. In this case {@link CooMatrix COO matrices} are usually preferred.
 *
 * <p>Conversion to other formats, such as COO or dense matrices, can be performed using {@link #toCoo()} or {@link #toDense()}.
 *
 * <h2>Usage Examples:</h2>
 * <pre>{@code
 * // Define matrix data.
 * Shape shape = new Shape(8, 8);
 * double[] data = {1.0, 2.0, 3.0, 4.0};
 * int[] rowPointers = {0, 1, 1, 1, 1, 3, 3, 3, 4}
 * int[] colIndices = {0, 0, 5, 2};
 *
 * // Create CSR matrix.
 * CsrMatrix matrix = new CsrMatrix(shape, data, rowPointers, colIndices);
 *
 * // Add matrices.
 * CsrMatrix sum = matrix.add(matrix);
 *
 * // Compute matrix-matrix multiplication.
 * Matrix prod = matrix.mult(matrix);
 * CsrMatrix sparseProd = matrix.mult2Csr(matrix);
 *
 * // Compute matrix-vector multiplication.
 * Vector denseVector = new Vector(matrix.numCols, 5.0);
 * Matrix matrixVectorProd = matrix.mult(denseVector);
 * }</pre>
 *
 * @see CMatrix
 * @see CooCMatrix
 * @see CVector
 * @see CooCVector
 */
public class CsrCMatrix extends AbstractCsrFieldMatrix<CsrCMatrix, CMatrix, CooCVector, Complex128> {

    private static final long serialVersionUID = 1L;
    // TODO: Implement coalesce and and drop zero methods for all CSR classes.

    /**
     * Creates a complex sparse CSR matrix with the specified {@code shape}, non-zero data, row pointers, and non-zero column
     * indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     */
    public CsrCMatrix(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries, rowPointers, colIndices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex sparse CSR matrix with the specified {@code shape}, non-zero data, row pointers, and non-zero column
     * indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     */
    public CsrCMatrix(Shape shape, List<Complex128> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        super(shape, entries.toArray(new Complex128[0]),
                ArrayConversions.fromIntegerList(rowPointers),
                ArrayConversions.fromIntegerList(colIndices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param shape Shape of the zero matrix.
     */
    public CsrCMatrix(Shape shape) {
        super(shape, new Complex128[0], new int[shape.getSize(0) + 1], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex sparse CSR matrix with the specified shape, non-zero data, row pointers, and non-zero column
     * indices.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     */
    public CsrCMatrix(int rows, int cols, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowPointers, colIndices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex sparse CSR matrix with the specified {@code shape}, non-zero data, row pointers, and non-zero column
     * indices.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     */
    public CsrCMatrix(int rows, int cols, List<Complex128> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        super(new Shape(rows, cols), entries.toArray(new Complex128[0]),
                ArrayConversions.fromIntegerList(rowPointers),
                ArrayConversions.fromIntegerList(colIndices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    public CsrCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[0], new int[rows+1], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing CSR matrices.
     * @param shape The shape of the matrix to construct.
     * @param data The non-zero data of this COO matrix.
     * @param rowPointers The non-zero row pointers of the CSR matrix.
     * @param colIndices The non-zero column indices of the CSR matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CsrCMatrix(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices, Object dummy) {
        super(shape, entries, rowPointers, colIndices, dummy);
    }


    /**
     * <p>Factory to construct a CSR matrix which bypasses any validation checks on the data and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param shape The full size of the COO matrix.
     * @param data The non-zero data of the COO matrix.
     * @param rowPointers The non-zero row pointers of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @return A COO matrix constructed from the provided parameters.
     */
    public static CsrCMatrix unsafeMake(Shape shape, Complex128[] data, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, data, rowPointers, colIndices, null);
    }



    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
    }


    /**
     * Constructs a sparse CSR tensor of the same type as this tensor with the specified non-zero data and indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A sparse CSR tensor of the same type as this tensor with the specified non-zero data and indices.
     */
    @Override
    public CsrCMatrix makeLikeTensor(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a CSR matrix with the specified shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A CSR matrix with the specified shape, non-zero data, and non-zero indices.
     */
    @Override
    public CsrCMatrix makeLikeTensor(Shape shape, List<Complex128> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a dense matrix which is of a similar type to this sparse CSR matrix.
     *
     * @param shape Shape of the dense matrix.
     * @param entries Entries of the dense matrix.
     *
     * @return A dense matrix which is of a similar type to this sparse CSR matrix with the specified {@code shape}
     * and {@code data}.
     */
    @Override
    public CMatrix makeLikeDenseTensor(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * <p>Constructs a sparse COO matrix of a similar type to this sparse CSR matrix.
     * <p>Note: this method constructs a new COO matrix with the specified data and indices. It does <em>not</em> convert this matrix
     * to a CSR matrix. To convert this matrix to a sparse COO matrix use {@link #toCoo()}.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the sparse COO matrix.
     * @param colIndices Non-zero column indices of the Sparse COO matrix.
     *
     * @return A sparse COO matrix of a similar type to this sparse CSR matrix.
     */
    @Override
    public CooCMatrix makeLikeCooMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooCMatrix toCoo() {
        Complex128[] cooEntries = new Complex128[nnz];
        int[] cooRowIndices = new int[nnz];
        int[] cooColIndices = new int[nnz];
        CsrConversions.toCoo(shape, data, rowPointers, colIndices, cooEntries, cooRowIndices, cooColIndices);
        return CooCMatrix.unsafeMake(shape, cooEntries, cooRowIndices, cooColIndices);
    }


    /**
     * Converts this CSR matrix to an equivalent sparse COO tensor.
     *
     * @return An sparse COO tensor equivalent to this CSR matrix.
     */
    @Override
    public CooCTensor toTensor() {
        return toCoo().toTensor();
    }


    /**
     * Converts this CSR matrix to an equivalent COO tensor with the specified shape.
     *
     * @param shape@return A COO tensor equivalent to this CSR matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooCTensor toTensor(Shape shape) {
        return toTensor(shape).reshape(shape);
    }


    /**
     * Converts this matrix to an equivalent real matrix. This is done by ignoring the imaginary components of this matrix.
     * @return A real matrix which is equivalent to this matrix.
     */
    public CsrMatrix toReal() {
        return new CsrMatrix(shape, Complex128Ops.toReal(data),
                rowPointers.clone(), colIndices.clone());
    }


    /**
     * Checks if all data of this matrix are real.
     * @return {@code true} if all data of this matrix are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this matrix has non-zero imaginary component.
     * @return {@code true} if any entry of this matrix has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CsrCMatrix makeLikeNDArray(Shape shape, Complex128[] data) {
        return new CsrCMatrix(shape, data, rowPointers.clone(), colIndices.clone());
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
    public CTensor tensorDot(CsrCMatrix src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     * @param b Second matrix in the matrix multiplication.
     * @return The result of multiplying this matrix with the matrix {@code b}.
     */
    public CMatrix mult(Matrix b) {
        return (CMatrix) RealFieldDenseCsrMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     * @param b Second matrix in the matrix multiplication.
     * @return The result of multiplying this matrix with the matrix {@code b}.
     */
    public CMatrix mult(CMatrix b) {
        return (CMatrix) DenseCsrSemiringMatMult.standard(this, b);
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of {@code b}.
     */
    @Override
    public CVector mult(CooCVector b) {
        Complex128[] dest = new Complex128[b.size];
        SemiringCsrMatMult.standardVector(shape, data, rowPointers, colIndices,
                b.size, b.data, b.indices,
                dest, getZeroElement());
        return new CVector(dest);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     * @param b Second matrix in the matrix multiplication.
     * @return The result of multiplying this matrix with the matrix {@code b}.
     */
    public CMatrix mult(CsrMatrix b) {
        return RealComplexCsrMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     * @param b Second matrix in the matrix multiplication.
     * @return The result of multiplying this matrix with the matrix {@code b}.
     */
    public CsrCMatrix mult2Csr(CsrMatrix b) {
        return RealComplexCsrMatMult.standardAsSparse(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     * @param b Second matrix in the matrix multiplication.
     * @return The result of multiplying this matrix with the matrix {@code b}.
     */
    public CsrCMatrix mult2Csr(CsrCMatrix b) {
        SparseMatrixData<Complex128> destData = SemiringCsrMatMult.standardToSparse(
                shape, data, rowPointers, colIndices,
                b.shape, b.data, b.rowPointers, b.colIndices);
        return new CsrCMatrix(destData.shape(), destData.data(), destData.rowData(), destData.colData());
    }


    /**
     * Gets a range of rows in this matrix.
     *
     * @param rowIdx The index of the row to get.
     * @param start The staring column of the row range to get (inclusive).
     * @param stop The ending column of the row range to get (exclusive).
     *
     * @return A vector containing the elements of the specified row over the range [start, stop).
     *
     * @throws IllegalArgumentException If {@code rowIdx < 0 || rowIdx >= this.numRows()} or {@code start < 0 || start >= numCols} or
     *                                  {@code stop < start || stop > numCols}.
     */
    @Override
    public CooCVector getRow(int rowIdx, int start, int stop) {
        return toCoo().getRow(rowIdx, start, stop);
    }


    /**
     * Gets a range of columns in this matrix.
     *
     * @param colIdx The index of the column to get.
     * @param start The staring row of the column range to get (inclusive).
     * @param stop The ending row of the column range to get (exclusive).
     *
     * @return A vector containing the elements of the specified column over the range [start, stop).
     *
     * @throws IllegalArgumentException If {@code colIdx < 0 || colIdx >= this.numCols()} or {@code start < 0 || start >= numRows} or
     *                                  {@code stop < start || stop > numRows}.
     */
    @Override
    public CooCVector getCol(int colIdx, int start, int stop) {
        return toCoo().getCol(colIdx, start, stop);
    }


    /**
     * Gets the elements of this matrix along the specified diagonal.
     *
     * @param diagOffset The diagonal to get within this matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     *
     * @return The elements of the specified diagonal as a vector.
     */
    @Override
    public CooCVector getDiag(int diagOffset) {
        return toCoo().getDiag(diagOffset);
    }


    /**
     * Sets the specified index of this matrix to the provided value. This is <em>not</em> done in-place as the number of non-zero
     * data in a sparse tensor is fixed.
     * @param value Value to set within matrix.
     * @param rowIdx Row index to set.
     * @param colIdx Column index to set.
     * @return A new CSR matrix with the specified
     */
    public CsrCMatrix set(double value, int rowIdx, int colIdx) {
        return set(new Complex128(value), rowIdx, colIdx);
    }


    /**
     * Rounds all data within this matrix to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new matrix containing the data of this matrix rounded to the specified precision.
     */
    public CsrCMatrix round(int precision) {
        return new CsrCMatrix(shape, Complex128Ops.round(data, precision), rowPointers.clone(), colIndices.clone());
    }


    /**
     * Sets all elements of this matrix to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this matrix with all data within {@code tol} of zero set to zero.
     */
    public CsrCMatrix roundToZero(double tolerance) {
        return toCoo().roundToZero(tolerance).toCsr();
    }


    /**
     * Drops any explicit zeros in this sparse COO matrix.
     * @return A copy of this Csr matrix with any explicitly stored zeros removed.
     */
    public CsrCMatrix dropZeros() {
        SparseMatrixData<Complex128> dropData = SparseUtils.dropZerosCsr(shape, this.data, rowPointers, colIndices);
        return new CsrCMatrix(dropData.shape(), dropData.data(), dropData.rowData(), dropData.colData());
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
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CsrCMatrix b = (CsrCMatrix) object;

        return SparseUtils.CSREquals(this, b);
    }


    @Override
    public int hashCode() {
        if(nnz == 0) return 0;

        int result = 17;

        // Hash calculation ignores explicit zeros in the matrix. This upholds the contract with the equals(Object) method.
        for(int row = 0; row<numRows; row++) {
            for(int idx = rowPointers[row], rowStop = rowPointers[row + 1]; idx < rowStop; idx++) {
                if(!data[idx].isZero()) {
                    result = 31*result + data[idx].hashCode();
                    result = 31*result + Integer.hashCode(colIndices[idx]);
                    result = 31*result + Integer.hashCode(row);
                }
            }
        }

        return result;
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("nnz: ").append(nnz).append("\n");
        result.append("Non-zero data: [");

        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();

        int stopIndex = Math.min(maxCols -1, size-1);
        int width;
        String value;

        if(data.length > 0) {
            // Get data up until the stopping point.
            for(int i=0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(data[i], PrintOptions.getPrecision());
                width = padding + value.length();
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = padding + 3;
                value = "...";
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(data[size-1], PrintOptions.getPrecision());
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Pointers: ")
                .append(PrettyPrint.abbreviatedArray(rowPointers, maxCols, padding, centering))
                .append("\n");
        result.append("Col Indices: ")
                .append(PrettyPrint.abbreviatedArray(colIndices, maxCols, padding, centering));

        return result.toString();
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of {@code b}.
     */
    public CVector mult(CVector b) {
        return (CVector) DenseCsrSemiringMatMult.standardVector(this, b);
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of {@code b}.
     */
    public CVector mult(Vector b) {
        return (CVector) RealFieldDenseCsrMatMult.standardVector(this, b);
    }
}
