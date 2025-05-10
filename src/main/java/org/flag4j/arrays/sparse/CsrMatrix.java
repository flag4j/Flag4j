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
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.primitive_arrays.AbstractDoubleTensor;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real.RealCsrDenseMatMult;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrMatMult;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.csr.real.*;
import org.flag4j.linalg.ops.sparse.csr.real_complex.RealComplexCsrMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;

import static org.flag4j.linalg.ops.sparse.SparseUtils.sortCsrMatrix;

/**
 * <p>Instances of this class represent a real sparse matrix using the compressed sparse row (CSR) format.
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
 * @see Matrix
 * @see CooMatrix
 * @see Vector
 * @see CooVector
 */
public class CsrMatrix extends AbstractDoubleTensor<CsrMatrix>
        implements MatrixMixin<CsrMatrix, Matrix, CooVector, Double> {

    // TODO: Implement sparse-matrix dense-vector multiplication.

    private static final long serialVersionUID = 1L;

    /**
     * <p>Pointers indicating starting index of each row within the {@link #colIndices} and {@link #data} arrays.
     * Has length {@link #numRows numRows + 1}.
     *
     * <p>The range {@code [data[rowPointers[i]], data[rowPointers[i+1]])} contains all {@link #data non-zero data} within
     * row {@code i}.
     *
     * <p>Similarly, {@code [colData[rowPointers[i]], colData[rowPointers[i+1]])} contains all {@link #colIndices column indices}
     * for the data in row {@code i}.
     */
    public final int[] rowPointers;
    /**
     * Column indices for non-zero values of this sparse CSR matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero data in this CSR matrix.
     */
    public final int nnz;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;
    /**
     * The sparsity of this matrix.
     */
    private double sparsity = -1;


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
     * @throws ArrayShapeException If {@code shape.getRank() != 2}.
     */
    public CsrMatrix(Shape shape, double[] data, int[] rowPointers, int[] colIndices) {
        super(shape, data);
        SparseValidation.validateCsr(shape, data.length, rowPointers, colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.nnz = data.length;
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero data, row pointers, and non-zero column indices.
     *
     * @param numRows The number of rows in this matrix.
     * @param numCols The number of columns in this matrix.
     * @param data The non-zero data of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code data} and {@code colData} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code data.length == colData.length}.
     */
    public CsrMatrix(int numRows, int numCols, double[] data, int[] rowPointers, int[] colIndices) {
        super(new Shape(numRows, numCols), data);
        SparseValidation.validateCsr(shape, data.length, rowPointers, colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.nnz = data.length;
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Constructs a zero matrix with the specified shape.
     * @param numRows Number of rows in the zero matrix to construct.
     * @param numCols Number of columns in the zero matrix to construct.
     */
    public CsrMatrix(int numRows, int numCols) {
        super(new Shape(numRows, numCols), new double[0]);

        this.rowPointers = new int[numRows + 1];
        this.colIndices = new int[0];
        this.nnz = 0;
        this.numRows = numRows;
        this.numCols = numCols;
    }


    /**
     * Constructs zero matrix with the specified {@code shape}.
     * @param shape Shape of the zero matrix to construct. Must be rank 2.
     * @throws ArrayShapeException If {@code shape.getRank() != 2}.
     */
    public CsrMatrix(Shape shape) {
        super(shape, new double[0]);
        ValidateParameters.ensureRank(shape, 2);

        this.rowPointers = new int[0];
        this.colIndices = new int[0];
        this.nnz = 0;
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing CSR matrices.
     * @param shape The shape of the matrix to construct.
     * @param data The non-zero data of this COO matrix.
     * @param rowPointers The non-zero row pointers of the CSR matrix.
     * @param colIndices The non-zero column indices of the CSR matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CsrMatrix(Shape shape, double[] data, int[] rowPointers, int[] colIndices, Object dummy) {
        super(shape, data);

        this.rowPointers = new int[0];
        this.colIndices = new int[0];
        this.nnz = 0;
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
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
    public static CsrMatrix unsafeMake(Shape shape, double[] data, int[] rowPointers, int[] colIndices) {
        return new CsrMatrix(shape, data, rowPointers, colIndices, null);
    }


    /**
     * Gets the length of the data array which backs this matrix.
     *
     * @return The length of the data array which backs this matrix.
     */
    @Override
    public int dataLength() {
        return data.length;
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
    public Matrix tensorDot(CsrMatrix src2, int[] aAxes, int[] bAxes) {
        return RealCsrMatrixTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
    public CooTensor tensorTr(int axis1, int axis2) {
        return toTensor().tensorTr(axis1, axis2);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CsrMatrix set(Double value, int... indices) {
        // Ensure indices are in bounds.
        ValidateParameters.validateTensorIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];

        double[] newEntries;
        int[] newRowPointers = rowPointers.clone();
        int[] newColIndices;
        boolean found = false; // Flag indicating an element already exists in this matrix at the specified row and col.
        int loc = -1;

        if(rowPointers[row] < rowPointers[row+1]) {
            int start = rowPointers[row];
            int stop = rowPointers[row+1];

            loc = Arrays.binarySearch(colIndices, start, stop, col);
            found = loc >= 0;
        }

        if(found) {
            newEntries = data.clone();
            newEntries[loc] = value;
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch
            newEntries = new double[data.length + 1];
            newColIndices = new int[data.length + 1];

            // Copy old data and insert new one.
            System.arraycopy(data, 0, newEntries, 0, loc);
            newEntries[loc] = value;
            System.arraycopy(data, loc, newEntries, loc+1, data.length-loc);

            // Copy old column indices and insert new one.
            System.arraycopy(colIndices, 0, newColIndices, 0, loc);
            newColIndices[loc] = col;
            System.arraycopy(colIndices, loc, newColIndices, loc+1, data.length-loc);

            // Increment row pointers.
            for(int i=row+1; i<rowPointers.length; i++) {
                newRowPointers[i]++;
            }
        }

        return new CsrMatrix(shape, newEntries, newRowPointers, newColIndices);
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CsrMatrix flatten() {
        return toCoo().flatten().toCsr();
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public CsrMatrix flatten(int axis) {
        return toCoo().flatten(axis).toCsr();
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws ArrayShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    @Override
    public CsrMatrix reshape(Shape newShape) {
        return toCoo().reshape().toCsr();
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public Double get(int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];
        return get(row, col);
    }


    /**
     * Constructs a CSR matrix of the same type as this matrix with the given the {@code shape} and {@code data} and the same
     * row pointers and column indices as this matrix.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A CSR matrix of the same type as this matrix with the given the {@code shape} and {@code data} and the same
     * row pointers and column indices as this matrix.
     */
    @Override
    public CsrMatrix makeLikeNDArray(Shape shape, double[] data) {
        return new CsrMatrix(shape, data, rowPointers, colIndices);
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
    public CsrMatrix T(int axis1, int axis2) {
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return copy();

        return RealCsrOps.transpose(this);
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
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public CsrMatrix T(int... axes) {
        if(axes.length != 2 || !((axes[0] == 0 && axes[1] == 1) || (axes[0] == 1 && axes[1] == 0))) {
            throw new LinearAlgebraException("Cannot transpose axes: "  + Arrays.toString(axes) + " for tensor of rank 2.");
        }

        return RealCsrOps.transpose(this);
    }


    /**
     * The sparsity of this sparse CSR matrix. That is, the decimal percentage of elements in this matrix which are zero.
     *
     * @return The density of this sparse matrix.
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
        return 1.0 - getSparsity();
    }


    /**
     * Converts this sparse CSR matrix to an equivalent dense matrix.
     *
     * @return A dense matrix equivalent to this sparse CSR matrix.
     */
    public Matrix toDense() {
        double[] dest = new double[shape.totalEntries().intValueExact()];

        for(int i=0; i<rowPointers.length-1; i++) {
            int rowOffset = i*numCols;

            for(int j=rowPointers[i]; j<rowPointers[i+1]; j++) {
                dest[rowOffset + colIndices[j]] = data[j];
            }
        }

        return new Matrix(shape, dest);
    }


    /**
     * Converts this CSR matrix to an equivalent {@link CooMatrix COO matrix}.
     * @return A {@link CooMatrix COO matrix} equivalent to this matrix.
     */
    public CooMatrix toCoo() {
        int[] destRowIdx = new int[data.length];

        for(int i=0; i<numRows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++)
                destRowIdx[j] = i;
        }

        return CooMatrix.unsafeMake(shape, data.clone(), destRowIdx, colIndices.clone());
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        sortCsrMatrix(data, rowPointers, colIndices);
    }


    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    @Override
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     *
     * @return The number of columns in this matrix.
     */
    @Override
    public int numCols() {
        return numCols;
    }


    /**
     * Gets the element of this matrix at this specified {@code row} and {@code col}.
     *
     * @param row Row index of the item to get from this matrix.
     * @param col Column index of the item to get from this matrix.
     *
     * @return The element of this matrix at the specified index.
     */
    @Override
    public Double get(int row, int col) {
        ValidateParameters.validateTensorIndex(shape, row, col);
        int loc = Arrays.binarySearch(colIndices, rowPointers[row], rowPointers[row+1], col);

        if(loc >= 0) return data[loc];
        else return 0.0;
    }


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     *
     * <p>Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double tr() {
        ValidateParameters.ensureSquareMatrix(shape);
        double trace = 0;

        for(int i=0; i<numRows; i++) {
            int rowPtr = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowPtr; j<stop; j++) {
                if(i==colIndices[j]) {
                    trace += data[j];
                }
            }
        }

        return trace;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return {@code true} is this matrix is upper triangular; {@code false} otherwise.
     *
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        if(!isSquare()) return false;

        for(int i=1; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowStart; j<stop; j++) {
                if(colIndices[j] >= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(data[j] != 0) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point then the matrix must be upper triangular.
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return {@code true} is this matrix is lower triangular; {@code false} otherwise.
     *
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        if(!isSquare()) return false;

        for(int i=0; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int rowStop = rowPointers[i+1];

            for(int j=rowStop-1; j>=rowStart; j--) {
                if(colIndices[j] <= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(data[j] != 0) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point then the matrix must be lower-triangular.
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return {@code true} if this matrix is the identity matrix; {@code false} otherwise.
     *
     * @see #isCloseToI()
     */
    @Override
    public boolean isI() {
        return RealCsrProperties.isIdentity(this);
    }


    /**
     * Checks that this matrix is close to the identity matrix.
     *
     * @return True if this matrix is approximately the identity matrix.
     *
     * @see #isI()
     */
    public boolean isCloseToI() {
        return RealCsrProperties.isCloseToIdentity(this);
    }


    /**
     * <p>Computes the determinant of a square matrix.
     * <p><b>WARNING:</b> This method will convert the matrix to a dense matrix in order to compute the determinant.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    public Double det() {
        return toDense().det();
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols() != b.numRows()}.
     */
    @Override
    public Matrix mult(CsrMatrix b) {
        return RealCsrMatMult.standard(this, b);
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     *
     * <p>Warning: This method will likely be slower than {@link #mult(CsrMatrix)} if the result of multiplying this matrix
     * with {@code b} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     *
     * @param b Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code b} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2Csr(CsrCMatrix b) {
        return RealComplexCsrMatMult.standardAsSparse(this, b);
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     *
     * <p>Warning: This method will likely be slower than {@link #mult(CsrMatrix)} if the result of multiplying this matrix
     * with {@code b} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     *
     * @param b Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code b} as a sparse CSR matrix.
     */
    public CsrMatrix mult2Csr(CsrMatrix b) {
        return RealCsrMatMult.standardAsSparse(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols() != b.numRows()}.
     */
    public CMatrix mult(CsrCMatrix b) {
        return RealComplexCsrMatMult.standard(this, b);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public Matrix multTranspose(CsrMatrix b) {
        return RealCsrMatMult.standard(this, b.T());
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix b.
     *
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    @Override
    public Double fib(CsrMatrix b) {
        return this.T().mult(b).tr();
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(MatrixMixin, int)
     * @see #augment(CsrMatrix) 
     */
    @Override
    public CsrMatrix stack(CsrMatrix b) {
        return toCoo().stack(b.toCoo()).toCsr();
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(CsrMatrix) 
     * @see #stack(MatrixMixin, int)
     */
    @Override
    public CsrMatrix augment(CsrMatrix b) {
        return toCoo().augment(b.toCoo()).toCsr();
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public CsrMatrix augment(CooVector b) {
        return toCoo().augment(b).toCsr();
    }


    /**
     * Swaps specified rows in the matrix. This is done in-place.
     *
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public CsrMatrix swapRows(int rowIndex1, int rowIndex2) {
        RealCsrManipulations.swapRows(this, rowIndex1, rowIndex2);
        return this;
    }


    /**
     * Swaps specified columns in the matrix. This is done in-place.
     *
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public CsrMatrix swapCols(int colIndex1, int colIndex2) {
        RealCsrManipulations.swapCols(this, colIndex1, colIndex2);
        return this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return {@code true} if this matrix is symmetric; {@code false} otherwise.
     *
     * @see #isAntiSymmetric()
     */
    @Override
    public boolean isSymmetric() {
        return RealCsrProperties.isSymmetric(this);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    @Override
    public boolean isHermitian() {
        return RealCsrProperties.isSymmetric(this);
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return {@code true} if this matrix is anti-symmetric; {@code false} otherwise.
     *
     * @see #isSymmetric()
     */
    public boolean isAntiSymmetric() {
        return RealCsrProperties.isAntiSymmetric(this);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return {@code true} if this matrix it is orthogonal; {@code false} otherwise.
     */
    @Override
    public boolean isOrthogonal() {
        return isSquare() && mult(T()).isCloseToI();
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public CsrMatrix removeRow(int rowIndex) {
        return toCoo().removeRow(rowIndex).toCsr();
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeRows(int... rowIndices) {
        return toCoo().removeRows(rowIndices).toCsr();
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeCol(int colIndex) {
        return toCoo().removeCol(colIndex).toCsr();
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeCols(int... colIndices) {
        return toCoo().removeCols(colIndices).toCsr();
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A copy of this matrix with the given slice set to the specified values.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the {@code values} slice, with the upper-left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSliceCopy(CsrMatrix values, int rowStart, int colStart) {
        return toCoo().setSliceCopy(values.toCoo(), rowStart, colStart).toCsr();
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     *
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public CsrMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        return RealCsrOps.getSlice(this, rowStart, rowEnd, colStart, colEnd);
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     *
     * @return A reference to this matrix.
     */
    @Override
    public CsrMatrix set(Double value, int row, int col) {
        // Ensure indices are in bounds.
        ValidateParameters.validateTensorIndex(shape, row, col);
        double[] newEntries;
        int[] newRowPointers = rowPointers.clone();
        int[] newColIndices;
        boolean found = false; // Flag indicating an element already exists in this matrix at the specified row and col.
        int loc = -1;

        if(rowPointers[row] < rowPointers[row+1]) {
            int start = rowPointers[row];
            int stop = rowPointers[row+1];

            loc = Arrays.binarySearch(colIndices, start, stop, col);
            found = loc >= 0;
        }

        if(found) {
            newEntries = data.clone();
            newEntries[loc] = value;
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch
            newEntries = new double[data.length + 1];
            newColIndices = new int[data.length + 1];

            // Copy old data and insert new one.
            System.arraycopy(data, 0, newEntries, 0, loc);
            newEntries[loc] = value;
            System.arraycopy(data, loc, newEntries, loc+1, data.length-loc);

            // Copy old column indices and insert new one.
            System.arraycopy(colIndices, 0, newColIndices, 0, loc);
            newColIndices[loc] = col;
            System.arraycopy(colIndices, loc, newColIndices, loc+1, data.length-loc);

            // Increment row pointers.
            for(int i=row+1; i<rowPointers.length; i++) {
                newRowPointers[i]++;
            }
        }

        return new CsrMatrix(shape, newEntries, newRowPointers, newColIndices);
    }


    /**
     * Extracts the upper-triangular portion of this matrix with a specified diagonal offset. All other data of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for the upper-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all data at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all data at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all data at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The upper-triangular portion of this matrix with a specified diagonal offset. All other data of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public CsrMatrix getTriU(int diagOffset) {
        return toCoo().getTriU(diagOffset).toCsr();
    }


    /**
     * Extracts the lower-triangular portion of this matrix with a specified diagonal offset. All other data of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for the lower-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all data at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all data at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all data at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The lower-triangular portion of this matrix with a specified diagonal offset. All other data of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public CsrMatrix getTriL(int diagOffset) {
        return toCoo().getTriL(diagOffset).toCsr();
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws IllegalArgumentException If {@code this.numCols != b.size}.
     */
    @Override
    public Vector mult(CooVector b) {
        return RealCsrMatMult.standardVector(this, b);
    }


    /**
     * Multiplies this sparse CSR matrix with a real dense vector.
     * @param b The real dense matrix in the matrix-vector product.
     * @return Computes the matrix-vector product of this matrix and {@code b}.
     * @throws IllegalArgumentException If {@code this.numCols != b.size}.
     */
    public Vector mult(Vector b) {
        return RealCsrDenseMatMult.standardVector(this, b);
    }


    /**
     * Multiplies this sparse CSR matrix with a complex dense vector.
     * @param b The real dense matrix in the matrix-vector product.
     * @return Computes the matrix-vector product of this matrix and {@code b}.
     * @throws IllegalArgumentException If {@code this.numCols != b.size}.
     */
    public CVector mult(CVector b) {
        return (CVector) RealFieldDenseCsrMatMult.standardVector(this, b);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooVector toVector() {
        int type = vectorType();
        int[] indices = new int[data.length];

        if(type == -1) {
            // Not a vector.
            for(int i=0; i<numRows; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i+1];
                int rowOffset = i*numCols;

                for(int j=start; j<stop; j++) {
                    indices[j] = rowOffset + colIndices[j];
                }
            }

        } else if(type <= 1) {
            // Row vector.
            System.arraycopy(colIndices, 0, indices, 0, colIndices.length);
        } else {
            // Column vector.
            for(int i=0; i<numRows; i++) {
                for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++)
                    indices[j] = i;
            }
        }

        return CooVector.unsafeMake(shape.totalEntries().intValueExact(), data.clone(), indices);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO tensor.
     * @return
     */
    public CooTensor toTensor() {
        return toCoo().toTensor();
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
    public CooVector getRow(int rowIdx) {
        ValidateParameters.validateArrayIndices(numRows, rowIdx);
        int start = rowPointers[rowIdx];

        double[] destEntries = new double[rowPointers[rowIdx + 1]-start];
        int[] destIndices = new int[destEntries.length];

        System.arraycopy(data, start, destEntries, 0, destEntries.length);
        System.arraycopy(colIndices, start, destIndices, 0, destEntries.length);

        return CooVector.unsafeMake(this.numCols, destEntries, destIndices);
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
    public CooVector getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.validateArrayIndices(numRows, rowIdx);
        ValidateParameters.validateArrayIndices(numCols, colStart, colEnd-1);
        int start = rowPointers[rowIdx];
        int end = rowPointers[rowIdx+1];

        List<Double> row = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int j=start; j<end; j++) {
            int col = colIndices[j];

            if(col >= colStart && col < colEnd) {
                row.add(data[j]);
                indices.add(col-colStart);
            }
        }

        return new CooVector(this.numCols-colStart, row, indices);
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
    public CooVector getCol(int colIdx) {
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
     * @throws @throws                  IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the
     *                                  shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    public CooVector getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.validateArrayIndices(numCols, colIdx);
        ValidateParameters.validateArrayIndices(numRows, rowStart, rowEnd-1);

        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=rowStart; i<rowEnd; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                if(colIndices[j]==colIdx) {
                    destEntries.add(data[j]);
                    destIndices.add(i);
                    break; // Should only be a single entry with this row and column index.
                }
            }
        }

        return new CooVector(numRows, destEntries, destIndices);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal data of this matrix.
     */
    public CooVector getDiag() {
        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<numRows; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            int loc = Arrays.binarySearch(colIndices, start, stop, i); // Search for matching column index within row.

            if(loc >= 0) {
                destEntries.add(data[loc]);
                destIndices.add(i);
            }
        }

        return new CooVector(Math.min(numRows, numCols), destEntries, destIndices);
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
    public CooVector getDiag(int diagOffset) {
        return toCoo().getDiag();
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows in this matrix.
     */
    public CsrMatrix setCol(CooVector values, int colIndex) {
        // Convert to COO first for more efficient modification.
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows in this matrix.
     */
    public CsrMatrix setRow(CooVector values, int rowIndex) {
        // Convert to COO first for more efficient modification.
        return toCoo().setRow(values, rowIndex).toCsr();
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
    public CsrMatrix T() {
        return RealCsrOps.transpose(this);
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

        CsrMatrix b = (CsrMatrix) object;

        return SparseUtils.CSREquals(this, b);
    }


    @Override
    public int hashCode() {
        if(nnz == 0) return 0;

        int result = 17;
        result = 31*result + shape.hashCode();

        // Hash calculation ignores explicit zeros in the matrix. This upholds the contract with the equals(Object) method.
        for(int row = 0; row<numRows; row++) {
            for(int idx = rowPointers[row], rowStop = rowPointers[row + 1]; idx < rowStop; idx++) {
                if (data[idx] != 0.0) {
                    result = 31 * result + Double.hashCode(data[idx]);
                    result = 31 * result + Integer.hashCode(colIndices[idx]);
                    result = 31 * result + Integer.hashCode(row);
                }
            }
        }

        return result;
    }


    /**
     * Multiplies this sparse CSR matrix with a real dense matrix.
     * @param b The real dense matrix in the matrix-matrix product.
     * @return Computes the matrix product of this matrix and {@code b}.
     * @throws IllegalArgumentException If {@code this.numCols != b.numRows}.
     */
    public Matrix mult(Matrix b) {
        return RealCsrDenseMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix {@code b}.
     */
    public CMatrix mult(CMatrix B) {
        return (CMatrix) RealFieldDenseCsrMatMult.standard(this, B);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CsrMatrix add(CsrMatrix b) {
        return RealCsrOps.applyBinOpp(this, b, Double::sum, null);
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
    public CsrMatrix elemMult(CsrMatrix b) {
        return RealCsrOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CsrMatrix sub(CsrMatrix b) {
        return RealCsrOps.applyBinOpp(this, b, (Double x, Double y)->x-y, (Double x) -> -x);
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public CsrMatrix H() {
        return T();
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        return shape.getNdIndices(RealProperties.argmin(data));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return shape.getNdIndices(RealProperties.argmax(data));
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return shape.getNdIndices(RealProperties.argminAbs(data));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return shape.getNdIndices(RealProperties.argmaxAbs(data));
    }


    /**
     * Adds a complex-valued scalar to all non-zero data of this sparse matrix.
     * @param b scalar to add.
     * @return The result of adding this matrix to {@code b}.
     */
    public CsrCMatrix add(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(data, b, dest);
        return new CsrCMatrix(shape, dest, rowPointers.clone(), colIndices.clone());
    }


    /**
     * Subtracts a complex-valued scalar from all non-zero data of this sparse matrix.
     * @param b scalar to subtract.
     * @return The result of subtracting {@code b} from this matrix's non-zero data.
     */
    public CsrCMatrix sub(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(data, b, dest);
        return new CsrCMatrix(shape, dest, rowPointers.clone(), colIndices.clone());
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     *
     * <p><b>Warning</b>: This method is not supported for sparse matrices. If called on a sparse matrix,
     * an {@link UnsupportedOperationException} will be thrown as the operation would almost certainly
     * result in a division by zero.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    @Override
    public CsrMatrix div(CsrMatrix b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse matrices.");
    }


    /**
     * Drops any explicit zeros in this sparse COO matrix.
     * @return A copy of this Csr matrix with any explicitly stored zeros removed.
     */
    public CsrMatrix dropZeros() {
        return SparseUtils.dropZerosCsr(this);
    }


    /**
     * Coalesces this sparse CSR matrix. An uncoalesced matrix is a sparse matrix with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by summing duplicated data. If another form of aggregation other
     * than summation is desired, use {@link #coalesce(BinaryOperator)}.
     * @return A new coalesced sparse CSR matrix which is equivalent to this CSR matrix.
     * @see #coalesce(BinaryOperator)
     */
    public CsrMatrix coalesce() {
        return toCoo().coalesce().toCsr();
    }


    /**
     * Coalesces this sparse COO matrix. An uncoalesced matrix is a sparse matrix with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data using {@code aggregator}.
     * @param aggregator Custom aggregation function to combine multiple.
     * @return A new coalesced sparse COO matrix which is equivalent to this COO matrix.
     * @see #coalesce()
     */
    public CsrMatrix coalesce(BinaryOperator<Double> aggregator) {
        return toCoo().coalesce(aggregator).toCsr();
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

        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();
        int precision = PrintOptions.getPrecision();
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

        result.append("Row Pointers: ")
                .append(PrettyPrint.abbreviatedArray(rowPointers, maxCols, padding, centering))
                .append("\n");
        result.append("Col Indices: ")
                .append(PrettyPrint.abbreviatedArray(colIndices, maxCols, padding, centering));

        return result.toString();
    }
}
