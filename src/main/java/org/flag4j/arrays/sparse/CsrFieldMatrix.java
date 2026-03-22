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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.ArrayMask;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.field_arrays.AbstractCsrFieldMatrix;
import org.flag4j.arrays.backend.semiring_arrays.TensorOverSemiring;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.csr.semiring_ops.SemiringCsrMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * <p>Instances of this class represent a sparse matrix using the compressed sparse row (CSR) format where
 * all data elements belonging to a specified {@link Field} type.
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
 * Complex128[] data = {
 *      new Complex128(1, 2), new Complex128(3, 4),
 *      new Complex128(5, 6), new Complex128(7, 8)
 * };
 * int[] rowPointers = {0, 1, 1, 1, 1, 3, 3, 3, 4}
 * int[] colIndices = {0, 0, 5, 2};
 *
 * // Create CSR matrix.
 * CsrFieldMatrix<Complex128> matrix = new CsrFieldMatrix<>(shape, data, rowPointers, colIndices);
 *
 * // Add matrices.
 * CsrFieldMatrix<Complex128> sum = matrix.add(matrix);
 *
 * // Compute matrix-matrix multiplication.
 * Matrix prod = matrix.mult(matrix);
 * CsrFieldMatrix<Complex128> sparseProd = matrix.mult2Csr(matrix);
 *
 * // Compute matrix-vector multiplication.
 * FieldVector<Complex128> denseVector = new FieldVector(matrix.numCols, new Complex128(5, 6));
 * FieldMatrix<Complex128> matrixVectorProd = matrix.mult(denseVector);
 * }</pre>
 *
 * @param <T> The type of elements stored in this matrix, constrained by the {@link Field} interface.
 * @see FieldMatrix
 * @see CooFieldMatrix
 * @see FieldVector
 * @see CooFieldVector
 */
public class CsrFieldMatrix<T extends Field<T>> extends AbstractCsrFieldMatrix<CsrFieldMatrix<T>,
        FieldMatrix<T>, CooFieldVector<T>, T> {

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
    public CsrFieldMatrix(Shape shape, T[] data, int[] rowPointers, int[] colIndices) {
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
    public CsrFieldMatrix(Shape shape, List<T> data, List<Integer> rowPointers, List<Integer> colIndices) {
        super(shape, (T[]) data.toArray(new Field[data.size()]),
                ArrayConversions.fromIntegerList(rowPointers),
                ArrayConversions.fromIntegerList(colIndices));
    }


    /**
     * Constructs a sparse CSR matrix representing the zero-matrix for the field which {@code fieldElement} belongs to.
     * @param shape Shape of the CSR matrix to construct.
     * @param fieldElement Element of the field which the entries of this
     */
    public CsrFieldMatrix(Shape shape, T fieldElement) {
        super(shape, (T[]) new Field[0], new int[0], new int[0]);
        setZeroElement(fieldElement.getZero());
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing CSR matrices.
     * @param shape The shape of the matrix to construct.
     * @param data The non-zero data of this COO matrix.
     * @param rowPointers The non-zero row pointers of the CSR matrix.
     * @param colIndices The non-zero column indices of the CSR matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CsrFieldMatrix(Shape shape, T[] data, int[] rowPointers, int[] colIndices, Object dummy) {
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
    public static <T extends Field<T>> CsrFieldMatrix<T> unsafeMake(
            Shape shape, Complex128[] data, int[] rowPointers, int[] colIndices) {
        return new CsrFieldMatrix(shape, data, rowPointers, colIndices, null);
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
    public CsrFieldMatrix<T> makeLikeTensor(Shape shape, T[] data, int[] rowPointers, int[] colIndices) {
        return new CsrFieldMatrix<>(shape, data, rowPointers, colIndices);
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
    public CsrFieldMatrix<T> makeLikeTensor(Shape shape, List<T> data, List<Integer> rowPointers, List<Integer> colIndices) {
        return new CsrFieldMatrix<T>(shape, (List<T>) data, rowPointers, colIndices);
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
    public FieldMatrix<T> makeLikeDenseTensor(Shape shape, T[] data) {
        return new FieldMatrix<>(shape, data);
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
    public CooFieldMatrix<T> makeLikeCooMatrix(Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<>(shape, data, rowIndices, colIndices);
    }


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     *
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     *
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     *
     * @throws ArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    @Override
    public AbstractNDArray<?, ?, T> get(ArrayMask mask) {
        // TODO: Implement this method
        return null;
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
    public CsrFieldMatrix<T> makeLikeNDArray(Shape shape, T[] data) {
        return new CsrFieldMatrix<>(shape, data, rowPointers.clone(), colIndices.clone());
    }


    /**
     * Applies a map to each item in this nD array. This operation is done in-place.
     * If this nD array is sparse, the {@code mapper} operation will only be applied to the non-zero
     * elements in this nD array.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A reference to this nD array.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     */
    @Override
    public CsrFieldMatrix<T> map(UnaryOperator<T> mapper) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the
     * starting value of the
     * reduction.
     * @param accumulator A binary operator that combines the current accumulated
     * result with the next array element and returns the updated result.
     *
     * @return The final accumulated scalar of type {@code V}. If this nD array is empty, {@code identity} will be returned.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator, int...)
     */
    @Override
    public T reduce(T identity, BinaryOperator<T> accumulator) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the starting value of the
     * reduction.
     * @param accumulator The binary operator used to accumulate elements of this nD array.
     * For the results to be well-defined, the accumulator must be associative and communitive.
     * @param axes The axes along which reduce this nD array.
     *
     * @return An nD array of the same shape as this nD array but with the specified {@code axes} removed.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator)
     */
    @Override
    public AbstractNDArray<?, ?, T> reduce(T identity, BinaryOperator<T> accumulator, int... axes) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Checks if each entry in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each entry in this nD array against.
     *
     * @return An {@link ArrayMask} of the same shape as this nD array containing the boolean results from evaluating each
     * entry in the nD array against the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #filter(Function)
     */
    @Override
    public ArrayMask where(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Extracts elements of this nD array that satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return A flat 1D array containing the elements of this nD array that satisfy the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #where(Function)
     */
    @Override
    public AbstractNDArray<?, ?, T> filter(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Checks if <em>any</em> element in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return {@code true} if <em>any</em> element in this nD array satisfies the {@code predicate}; otherwise {@code false}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #all(Function)
     */
    @Override
    public boolean any(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return false;
    }


    /**
     * Checks if <em>all</em> elements in this nD array satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return {@code true} if <em>all</em> elements in this nD array satisfy the {@code predicate}; otherwise {@code false}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #any(Function)
     */
    @Override
    public boolean all(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return false;
    }


    /**
     * Counts the number of elements in this nD array which satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return The number of elements in this nD array which satisfy the specified {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     */
    @Override
    public int countTrue(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return 0;
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooFieldMatrix<T> toCoo() {
        int[] cooRowIdx = new int[data.length];

        for(int i=0; i<numRows; i++) {
            int stop = rowPointers[i+1];

            for(int j=rowPointers[i]; j<stop; j++)
                cooRowIdx[j] = i;
        }

        return new CooFieldMatrix<T>(shape, data.clone(), cooRowIdx, colIndices.clone());
    }


    /**
     * Converts this CSR matrix to an equivalent sparse COO tensor.
     *
     * @return An sparse COO tensor equivalent to this CSR matrix.
     */
    @Override
    public CooFieldTensor<T> toTensor() {
        return toCoo().toTensor();
    }


    /**
     * Converts this CSR matrix to an equivalent COO tensor with the specified shape.
     *
     * @param shape@return A COO tensor equivalent to this CSR matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooFieldTensor<T> toTensor(Shape shape) {
        return toCoo().toTensor(shape);
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

        CsrFieldMatrix<T> b = (CsrFieldMatrix<T>) object;

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
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix does not equal the size of
     *                                {@code b}.
     */
    @Override
    public FieldVector<T> mult(CooFieldVector<T> b) {
        T[] dest = (T[]) new Field[b.size];
        SemiringCsrMatMult.standardVector(shape, data, rowPointers, colIndices,
                b.size, b.data, b.indices,
                dest, getZeroElement());
        return new FieldVector<>(dest);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooFieldVector<T> toVector() {
        int type = vectorType();
        int[] indices = new int[data.length];

        if(type == -1) {
            // Not a vector.
            for(int i=0; i<numRows; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i+1];
                int rowOffset = i*numCols;

                for(int j=start; j<stop; j++)
                    indices[j] = rowOffset + colIndices[j];
            }

        } else if(type <= 1) {
            // Row vector.
            System.arraycopy(colIndices, 0, indices, 0, colIndices.length);
        } else {
            // Column vector.
            for(int i=0; i<numRows; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i+1];

                for(int j=start; j<stop; j++)
                    indices[j] = i;
            }
        }

        return new CooFieldVector<T>(shape.totalEntriesIntValueExact(), data.clone(), indices);
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     *
     * @return The specified row of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     *                                        the number of rows in this matrix.
     */
    @Override
    public CooFieldVector<T> getRow(int rowIdx) {
        ValidateParameters.validateArrayIndices(numRows, rowIdx);
        int start = rowPointers[rowIdx];
        T[] destData = (T[]) new Field[rowPointers[rowIdx + 1]-start];
        int[] destIndices = new int[destData.length];

        System.arraycopy(data, start, destData, 0, destData.length);
        System.arraycopy(colIndices, start, destIndices, 0, destData.length);

        return new CooFieldVector<T>(numCols, destData, destIndices);
    }


    /**
     * Gets a specified row of this matrix between {@code colStart} (inclusive) and {@code colEnd} (exclusive).
     *
     * @param rowIdx Index of the row in this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The row at index {@code rowIdx} of this matrix between the {@code colStart} and {@code colEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code colEnd} is less than {@code colStart}.
     */
    public CooFieldVector<T> getRow(int rowIdx, int colStart, int colEnd) {
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

        return new CooFieldVector<T>(colEnd-colStart, row, indices);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     *
     * @return The specified column of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     *                                        the number of columns in this matrix.
     */
    public CooFieldVector<T> getCol(int colIdx) {
        return getCol(colIdx, 0, numRows);
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param colIdx Index of the column in this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     *
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    public CooFieldVector<T> getCol(int colIdx, int rowStart, int rowEnd) {
        // TODO: This method (and others returning a vector) could easily be used for complex csr matrices as well.
        //  Just need to pass a factory so the correct type of vector is returned.
        //  e.g. getCol(AbstractCsrSemiringMatrix<?, ?, ?, ?, T> mat, int colIdx, int rowStart, int rowEnd, CsrVectorFactory factory)
        ValidateParameters.validateArrayIndices(numCols, colIdx);
        ValidateParameters.validateArrayIndices(numRows, rowStart, rowEnd-1);

        List<T> destData = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=rowStart; i<rowEnd; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                if(colIndices[j]==colIdx) {
                    destData.add(data[j]);
                    destIndices.add(i);
                    break; // Should only be a single entry with this row and column index.
                }
            }
        }

        return new CooFieldVector<T>(numRows, destData, destIndices);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal data of this matrix.
     */
    public CooFieldVector<T> getDiag() {
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

        return new CooFieldVector<T>(Math.min(numRows, numCols), destData, destIndices);
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
    public CooFieldVector<T> getDiag(int diagOffset) {
        return toCoo().getDiag(diagOffset);
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
     * @throws IllegalArgumentException If the two tensor's shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public AbstractNDArray<?, T[], T> tensorDot(CsrFieldMatrix<T> src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * Computes the sum of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the sum. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     *
     * @see #sum()
     */
    @Override
    public TensorOverSemiring<?, ?, ?, T> sum(int... axes) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Computes the product of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the product. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     *
     * @see #prod()
     */
    @Override
    public TensorOverSemiring<?, ?, ?, T> prod(int... axes) {
        // TODO: Implement this method
        return null;
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
     * Drops any explicit zeros in this sparse COO matrix.
     * @return A copy of this Csr matrix with any explicitly stored zeros removed.
     */
    public CsrFieldMatrix<T> dropZeros() {
        SparseMatrixData<T> dest = SparseUtils.dropZerosCsr(shape, data, rowPointers, colIndices);
        return new CooFieldMatrix<>(dest.shape(), dest.data(), dest.rowData(), dest.colData()).toCsr();
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
