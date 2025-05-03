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
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldTensor;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatMult;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.List;
import java.util.function.BinaryOperator;

/**
 * <p>Instances of this class represent a sparse matrix whose non-zero-elements are stored in Coordinate List (COO) format, with all
 * data elements belonging to a specified {@link Field} type.
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
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // shape, data, and indices for COO matrix.
 * Shape shape = new Shape(512, 1024);
 * Complex128[] data = {
 *      new Complex128(1, 2), new Complex128(3, 4), new Complex128(5, 6)
 *      new Complex128(7, 8), new Complex128(9, 10), new Complex128(11, 12)
 * };
 * int[] rowIndices = {0, 4, 128, 128, 128, 256};
 * int[] colIndices = {16, 2, 5, 512, 1000, 28};
 *
 * // Create COO matrix.
 * CooFieldMatrix<Complex128> matrix = new CooFieldMatrix<>(
 *      shape, data, rowIndices, colIndices
 * );
 *
 * // Sum matrices.
 * CooFieldMatrix<Complex128> sum = matrix.add(matrix);
 *
 * // Multiply matrix to it's Hermitian transpose.
 * FieldMatrix<Complex128> prod = matrix.mult(matrix.H());
 * prod = matrix.multTranspose(matrix);
 * }</pre>
 *
 * @param <T> The type of elements stored in this matrix, constrained by the {@link Field} interface.
 * @see CooFieldTensor
 * @see CooFieldVector
 * @see FieldMatrix
 * @see Field
 */
public class CooFieldMatrix<T extends Field<T>> extends AbstractCooFieldMatrix<CooFieldMatrix<T>,
        FieldMatrix<T>, CooFieldVector<T>, T> {

    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooFieldMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
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
    public CooFieldMatrix(Shape shape, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape,
                (T[]) entries.toArray(new Field[entries.size()]),
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
    public CooFieldMatrix(int rows, int cols, T[] entries, int[] rowIndices, int[] colIndices) {
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
    public CooFieldMatrix(int rows, int cols, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(new Shape(rows, cols),
                (T[]) entries.toArray(new Field[entries.size()]),
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
    private CooFieldMatrix(Shape shape, T[] data, int[] rowIndices, int[] colIndices, Object dummy) {
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
    public static <T extends Field<T>> CooFieldMatrix<T> unsafeMake(
            Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix(shape, data, rowIndices, colIndices, null);
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
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices, colIndices);
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
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooFieldMatrix<>(shape, entries, rowIndices, colIndices);
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
    public CooFieldVector<T> makeLikeVector(Shape shape, T[] entries, int[] indices) {
        return new CooFieldVector<>(shape, entries, indices);
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
    public FieldMatrix<T> makeLikeDenseTensor(Shape shape, T[] entries) {
        return new FieldMatrix<>(shape, entries);
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
    public CsrFieldMatrix<T> makeLikeCsrMatrix(Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrFieldMatrix<>(shape, entries, rowPointers, colIndices);
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
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooFieldMatrix<>(shape, entries, rowIndices.clone(), colIndices.clone());
    }


    /**
     * <p>Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * <p>For matrices, calling {@code this.tensorDot(src2, new int[]{1}, new int[]{0})} is equivalent to matrix multiplication.
     * However, it is highly recommended to use {@link #mult(CooFieldVector)} instead.
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
    public FieldTensor<T> tensorDot(CooFieldMatrix<T> src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * <p>Converts this sparse COO matrix to an equivalent compressed sparse row (CSR) matrix.
     * <p>It is often easier and more efficient to construct a matrix in COO format first then convert to a CSR matrix for efficient
     * computations.
     *
     * @return A CSR matrix equivalent to this COO matrix.
     */
    @Override
    public CsrFieldMatrix<T> toCsr() {
        int[] rowPointers = new int[numRows + 1];

        // Count number of data per row.
        for(int i=0; i<nnz; i++)
            rowPointers[rowIndices[i] + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<numRows; i++)
            rowPointers[i+1] += rowPointers[i];

        return new CsrFieldMatrix<T>(shape, data.clone(), rowPointers, colIndices.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooFieldTensor<T> toTensor() {
        int[][] tIndices = RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices});
        return new CooFieldTensor<>(shape, data.clone(), tIndices);
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooFieldTensor<T> toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
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
    public FieldVector<T> mult(CooFieldVector<T> b) {
        T[] dest = makeEmptyDataArray(numRows);
        CooSemiringMatMult.standardVector(data, rowIndices, colIndices, shape, b.data, b.indices, dest);
        return new FieldVector<T>(dest);
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
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooFieldMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooFieldMatrix<T> src2 = (CooFieldMatrix<T>) object;

        return CooSemiringEquals.cooMatrixEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < data.length; i++) {
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Integer.hashCode(rowIndices[i]);
                result = 31*result + Integer.hashCode(colIndices[i]);
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

        boolean centering = PrintOptions.useCentering();
        int precision = PrintOptions.getPrecision();
        int padding = PrintOptions.getPadding();
        int maxCols = PrintOptions.getMaxColumns();

        int stopIndex = Math.min(maxCols -1, size-1);
        int width;
        String value;

        if(data.length > 0) {
            // Get data up until the stopping point.
            for(int i = 0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(data[i], precision);
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
            value = StringUtils.ValueOfRound(data[size-1], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Indices: ")
                .append(PrettyPrint.abbreviatedArray(rowIndices, maxCols, padding, centering))
                .append("\n");

        result.append("Col Indices: ")
                .append(PrettyPrint.abbreviatedArray(colIndices, maxCols, padding, centering));

        return result.toString();
    }
}
