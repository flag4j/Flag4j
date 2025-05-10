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
import org.flag4j.arrays.backend.semiring_arrays.AbstractCsrSemiringMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.SemiringMatrix;
import org.flag4j.arrays.dense.SemiringTensor;
import org.flag4j.arrays.dense.SemiringVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.csr.semiring_ops.SemiringCsrMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Field;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;

/**
 * <p>Instances of this class represent a sparse matrix using the compressed sparse row (CSR) format where
 * all data elements belonging to a specified {@link Semiring} type.
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
 * RealFloat32[] data = {
 *      new RealFloat32(1), new RealFloat32(2),
 *      new RealFloat32(3), new RealFloat32(4)
 * };
 * int[] rowPointers = {0, 1, 1, 1, 1, 3, 3, 3, 4}
 * int[] colIndices = {0, 0, 5, 2};
 *
 * // Create CSR matrix.
 * CsrSemiringMatrix<RealFloat32> matrix = new CsrSemiringMatrix<>(shape, data, rowPointers, colIndices);
 *
 * // Add matrices.
 * CsrSemiringMatrix<RealFloat32> sum = matrix.add(matrix);
 *
 * // Compute matrix-matrix multiplication.
 * Matrix prod = matrix.mult(matrix);
 * CsrSemiringMatrix<RealFloat32> sparseProd = matrix.mult2Csr(matrix);
 *
 * // Compute matrix-vector multiplication.
 * SemiringVector<RealFloat32> denseVector = new SemiringVector(matrix.numCols, new RealFloat32(5));
 * SemiringMatrix<RealFloat32> matrixVectorProd = matrix.mult(denseVector);
 * }</pre>
 *
 * @param <T> The type of elements stored in this matrix, constrained by the {@link Semiring} interface.
 * @see SemiringMatrix
 * @see CooSemiringMatrix
 * @see org.flag4j.arrays.dense.SemiringVector
 * @see CooSemiringVector
 */
public class CsrSemiringMatrix<T extends Semiring<T>> extends AbstractCsrSemiringMatrix<
        CsrSemiringMatrix<T>, SemiringMatrix<T>, CooSemiringVector<T>, T> {

    private static final long serialVersionUID = 1L;


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
    public CsrSemiringMatrix(Shape shape, T[] data, int[] rowPointers, int[] colIndices) {
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
     */
    public CsrSemiringMatrix(Shape shape, List<T> data, List<Integer> rowPointers, List<Integer> colIndices) {
        super(shape, (T[]) data.toArray(new Field[data.size()]),
                ArrayConversions.fromIntegerList(rowPointers),
                ArrayConversions.fromIntegerList(colIndices));
    }


    /**
     * Constructs a sparse CSR matrix representing the zero matrix for the field which {@code semiringElement} belongs to.
     * @param shape Shape of the CSR matrix to construct.
     * @param semiringElement Element of the field which the entries of this matrix belong to.
     */
    public CsrSemiringMatrix(Shape shape, T semiringElement) {
        super(shape, (T[]) new Field[0], new int[0], new int[0]);
        setZeroElement(semiringElement.getZero());
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing CSR matrices.
     * @param shape The shape of the matrix to construct.
     * @param data The non-zero data of this COO matrix.
     * @param rowPointers The non-zero row pointers of the CSR matrix.
     * @param colIndices The non-zero column indices of the CSR matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CsrSemiringMatrix(Shape shape, T[] data, int[] rowPointers, int[] colIndices, Object dummy) {
        super(shape, data, rowPointers, colIndices, dummy);
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
    public static <T extends Semiring<T>> CsrSemiringMatrix<T> unsafeMake(
            Shape shape, Complex128[] data, int[] rowPointers, int[] colIndices) {
        return new CsrSemiringMatrix(shape, data, rowPointers, colIndices, null);
    }


    /**
     * Constructs a sparse CSR tensor of the same type as this tensor with the specified non-zero data and indices.
     *
     * @param shape Shape of the matrix.
     * @param data Non-zero data of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A sparse CSR tensor of the same type as this tensor with the specified non-zero data and indices.
     */
    @Override
    public CsrSemiringMatrix<T> makeLikeTensor(Shape shape, T[] data, int[] rowPointers, int[] colIndices) {
        return new CsrSemiringMatrix<>(shape, data, rowPointers, colIndices);
    }


    /**
     * Constructs a CSR matrix with the specified shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param data Non-zero values of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A CSR matrix with the specified shape, non-zero data, and non-zero indices.
     */
    @Override
    public CsrSemiringMatrix<T> makeLikeTensor(Shape shape, List<T> data, List<Integer> rowPointers, List<Integer> colIndices) {
        return new CsrSemiringMatrix<>(shape, data, rowPointers, colIndices);
    }


    /**
     * Constructs a dense matrix which is of a similar type to this sparse CSR matrix.
     *
     * @param shape Shape of the dense matrix.
     * @param data Entries of the dense matrix.
     *
     * @return A dense matrix which is of a similar type to this sparse CSR matrix with the specified {@code shape}
     * and {@code data}.
     */
    @Override
    public SemiringMatrix<T> makeLikeDenseTensor(Shape shape, T[] data) {
        return new SemiringMatrix<>(shape, data);
    }


    /**
     * <p>Constructs a sparse COO matrix of a similar type to this sparse CSR matrix.
     * <p>Note: this method constructs a new COO matrix with the specified data and indices. It does <em>not</em> convert this matrix
     * to a CSR matrix. To convert this matrix to a sparse COO matrix use {@link #toCoo()}.
     *
     * @param shape Shape of the COO matrix.
     * @param data Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the sparse COO matrix.
     * @param colIndices Non-zero column indices of the Sparse COO matrix.
     *
     * @return A sparse COO matrix of a similar type to this sparse CSR matrix.
     */
    @Override
    public CooSemiringMatrix<T> makeLikeCooMatrix(Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        return new CooSemiringMatrix<>(shape, data, rowIndices, colIndices);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooSemiringMatrix<T> toCoo() {
        int[] cooRowIdx = new int[data.length];

        for(int i=0; i<numRows; i++) {
            int stop = rowPointers[i + 1];

            for(int j=rowPointers[i]; j<stop; j++)
                cooRowIdx[j] = i;
        }

        return new CooSemiringMatrix<T>(shape, data.clone(), cooRowIdx, colIndices.clone());
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
    public CsrSemiringMatrix<T> makeLikeNDArray(Shape shape, T[] data) {
        return new CsrSemiringMatrix<>(shape, data, rowPointers.clone(), colIndices.clone());
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
    public SemiringTensor<T> tensorDot(CsrSemiringMatrix<T> src2, int[] aAxes, int[] bAxes) {
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
        T[] dest = (T[]) new Semiring[b.size];
        SemiringCsrMatMult.standardVector(shape, data, rowPointers, colIndices,
                b.size, b.data, b.indices,
                dest, getZeroElement());
        return new SemiringVector<>(dest);
    }


    /**
     * Gets a range of rows in this matrix.
     *
     * @param rowIdx The index of the row to get.
     * @param colStart The staring column of the row range to get (inclusive).
     * @param colEnd The ending column of the row range to get (exclusive).
     *
     * @return A vector containing the elements of the specified row over the range [colStart, colEnd).
     *
     * @throws IllegalArgumentException If {@code rowIdx < 0 || rowIdx >= this.numRows()} or {@code colStart < 0 || colStart >= numCols} or
     *                                  {@code colEnd < colStart || colEnd > numCols}.
     */
    @Override
    public CooSemiringVector<T> getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.validateArrayIndices(numRows, rowIdx);
        ValidateParameters.validateArrayIndices(numCols, colStart, colEnd-1);
        int start = rowPointers[rowIdx];
        int end = rowPointers[rowIdx+1];

        List<T> row = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int j=start; j<end; j++) {
            int col = colIndices[j];

            if(col >= colStart && col < colEnd) {
                row.add(data[j]);
                indices.add(col-colStart);
            }
        }

        return new CooSemiringVector<T>(colEnd-colStart, row, indices);
    }


    /**
     * Gets a range of columns in this matrix.
     *
     * @param colIdx The index of the column to get.
     * @param rowStart The staring row of the column range to get (inclusive).
     * @param rowEnd The ending row of the column range to get (exclusive).
     *
     * @return A vector containing the elements of the specified column over the range [rowStart, rowEnd).
     *
     * @throws IllegalArgumentException If {@code colIdx < 0 || colIdx >= this.numCols()} or {@code rowStart < 0 || rowStart >= numRows} or
     *                                  {@code rowEnd < rowStart || rowEnd > numRows}.
     */
    @Override
    public CooSemiringVector<T> getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.validateArrayIndices(numCols, colIdx);
        ValidateParameters.validateArrayIndices(numRows, rowStart, rowEnd-1);

        List<T> destData = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=rowStart; i<rowEnd; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i + 1];

            for(int j=start; j<stop; j++) {
                if(colIndices[j]==colIdx) {
                    destData.add(data[j]);
                    destIndices.add(i);
                    break; // Should only be a single entry with this row and column index.
                }
            }
        }

        return new CooSemiringVector<T>(numRows, destData, destIndices);
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
    public CooSemiringVector<T> getDiag(int diagOffset) {
        List<T> destData = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<numRows; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];
            int loc = Arrays.binarySearch(colIndices, start, stop, i); // Search for matching column index within row.

            if(loc >= 0) {
                destData.add(data[loc]);
                destIndices.add(i);
            }
        }

        return new CooSemiringVector<T>(Math.min(numRows, numCols), destData, destIndices);
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
     * Converts this CSR matrix to an equivalent sparse COO tensor.
     *
     * @return An sparse COO tensor equivalent to this CSR matrix.
     */
    @Override
    public CooSemiringTensor<T> toTensor() {
        return (CooSemiringTensor<T>) toCoo().toTensor();
    }


    /**
     * Converts this CSR matrix to an equivalent COO tensor with the specified shape.
     *
     * @param shape@return A COO tensor equivalent to this CSR matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooSemiringTensor<T> toTensor(Shape shape) {
        return (CooSemiringTensor<T>) toCoo().toTensor(shape);
    }


    /**
     * Drops any explicit zeros in this sparse COO matrix.
     * @return A copy of this Csr matrix with any explicitly stored zeros removed.
     */
    public CsrSemiringMatrix<T> dropZeros() {
        SparseMatrixData<T> dest = SparseUtils.dropZerosCsr(shape, data, rowPointers, colIndices);
        return new CooSemiringMatrix<>(dest.shape(), dest.data(), dest.rowData(), dest.colData()).toCsr();
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("nnz: ").append(nnz).append("\n");

        int maxCols = PrintOptions.getMaxColumns();
        boolean centering = PrintOptions.useCentering();
        int precision = PrintOptions.getPrecision();
        int padding = PrintOptions.getPadding();

        result.append("Non-zero data: ")
                .append(PrettyPrint.abbreviatedArray(data, maxCols, padding, precision, centering))
                .append("\n");
        result.append("Row Pointers: ")
                .append(PrettyPrint.abbreviatedArray(rowPointers, maxCols, padding, centering))
                .append("\n");
        result.append("Col Indices: ")
                .append(PrettyPrint.abbreviatedArray(colIndices, maxCols, padding, centering));

        return result.toString();
    }
}
