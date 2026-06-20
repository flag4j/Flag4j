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

package org.flag4j.arrays.backend.semiring_arrays;


import org.flag4j.arrays.*;
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.sparse.SparseValidation;
import org.flag4j.linalg.ops.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.ops.sparse.SparseElementSearch;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.coo.*;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatMult;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatrixOps;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatrixProperties;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayMapper;
import org.flag4j.util.ArrayReducer;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;

import static org.flag4j.linalg.ops.sparse.SparseUtils.copyRanges;

/**
 * <p>A sparse matrix stored in coordinate list (COO) format. The {@link #data} of this COO vector are
 * elements of a {@link Semiring}.
 *
 * <p>The {@link #data non-zero data} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero data is fixed.
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
 *     <li>The non-zero {@link #data} of the matrix. All other data in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the data of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified, but any ops implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 *
 * @param <T> Type of this sparse COO matrix.
 * @param <U> Type of dense matrix which is similar to {@code T}.
 * @param <V> Type of sparse COO vector which is similar to {@code T}.
 * @param <W> Type of the semiring element in this matrix.
 */
public abstract class AbstractCooSemiringMatrix<T extends AbstractCooSemiringMatrix<T, U, V, W>,
        U extends AbstractDenseSemiringMatrix<U, ?, W>,
        V extends AbstractCooSemiringVector<V, ?, T, U, W>,
        W extends Semiring<W>>
        extends AbstractNDArray<T, W[], W>
        implements SemiringTensorMixin<T, U, W>, MatrixMixin<T, U, V, W> {

    /**
     * The zero-element for the semiring that this tensor's elements belong to.
     */
    protected W zeroElement;
    /**
     * Row indices for the non-zero value of this sparse COO matrix.
     */
    public final int[] rowIndices;
    /**
     * column indices for non-zero value of this sparse COO matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero data in this COO matrix.
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
    private double sparsity = -1.0;


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    protected AbstractCooSemiringMatrix(Shape shape, W[] data, int[] rowIndices, int[] colIndices) {
        super(shape, data);
        SparseValidation.validateCoo(shape, data.length, rowIndices, colIndices);

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = data.length;
        numRows = shape.getSize(0);
        numCols = shape.getSize(1);

        // Attempt to set the zero-element for the semiring.
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
    }


    /**
     * <p>Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     * <p>This constructor will not perform data validation on <em>any</em> of the parameters.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. The value of this object is ignored.
     */
    protected AbstractCooSemiringMatrix(Shape shape, W[] data, int[] rowIndices, int[] colIndices, Object dummy) {
        super(shape, data);

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = data.length;
        numRows = shape.getSize(0);
        numCols = shape.getSize(1);

        // Attempt to set the zero-element for the semiring.
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
    }


    /**
     * Constructs a sparse COO tensor of the same type as this tensor with the specified non-zero data and indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return A sparse COO tensor of the same type as this tensor with the specified non-zero data and indices.
     */
    public abstract T makeLikeNDArray(Shape shape, W[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and non-zero indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return A COO matrix with the specified shape, non-zero data, and non-zero indices.
     */
    public abstract T makeLikeNDArray(Shape shape, List<W> entries, List<Integer> rowIndices, List<Integer> colIndices);


    /**
     * Constructs a sparse COO vector of a similar type to this COO matrix.
     * @param shape Shape of the vector. Must be rank 1.
     * @param entries Non-zero data of the COO vector.
     * @param indices Non-zero indices of the COO vector.
     * @return A sparse COO vector of a similar type to this COO matrix.
     */
    public abstract V makeLikeVector(Shape shape, W[] entries, int[] indices);
    

    /**
     * Constructs a dense tensor with the specified {@code shape} and {@code data} which is a similar type to this sparse tensor.
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     * @return A dense tensor with the specified {@code shape} and {@code data} which is a similar type to this sparse tensor.
     */
    public abstract U makeLikeDenseNDArray(Shape shape, W[] entries);


    /**
     * Constructs a sparse CSR matrix of a similar type to this sparse COO matrix.
     * @param shape Shape of the CSR matrix to construct.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A CSR matrix of a similar type to this sparse COO matrix.
     */
    public abstract AbstractCsrSemiringMatrix<?, U, V, W> makeLikeCsrMatrix(
            Shape shape, W[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Gets the zero-element for the semiring of this tensor.
     * @return The zero-element for the semiring of this tensor. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Semiring)} then {@code null} will be returned.
     *
     * @see #setZeroElement(Semiring)
     */
    public W getZeroElement() {
        return zeroElement;
    }


    /**
     * Gets the sparsity of this matrix as a decimal percentage.
     * That is, the percentage of data in this matrix that is zero.
     * @return The sparsity of this matrix as a decimal percentage.
     * @see #getDensity()
     */
    public double getSparsity() {
        // Check if the sparsity has already been computed.
        if (this.sparsity < 0)
            this.sparsity = SparseUtils.computeSparsity(shape, nnz);

        return sparsity;
    }


    /**
     * Gets the density of this matrix as a decimal percentage.
     * That is, the percentage of data in this matrix that are non-zero.
     * @return The density of this matrix as a decimal percentage.
     * @see #getSparsity()
     */
    public double getDensity() {
        return 1.0 - sparsity;
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
     * Sets the zero-element for the semiring of this tensor.
     * @param zeroElement The zero-element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the semiring.
     *
     * @see #getZeroElement()
     */
    public void setZeroElement(W zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }


    /**
     * Gets the element of this tensor at the specified index.
     *
     * @param index Indices of the element to get.
     *
     * @return The element of this tensor at the specified index. If there is a non-zero value with the specified index, that value
     * will be returned. If there is no non-zero value at the specified index, then the zero-element will attempt to be
     * returned (i.e., the additive identity of the semiring). However, if the zero-element could not be determined during
     * construction or if it was not set with {@link #setZeroElement(Semiring)} then
     * {@code null} will be returned.
     *
     * @throws ArrayIndexOutOfBoundsException If any, index is not within this tensor.
     */
    @Override
    public W get(int... index) {
        ValidateParameters.validateTensorIndex(shape, index);
        return get(index[0], index[1]);
    }


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     *
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     *
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     *
     * @throws NDArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    @Override
    public AbstractNDArray<?, ?, W> get(ArrayMask mask) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(W value, int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        return set(value, indices[0], indices[1]);
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
    public T set(W value, int row, int col) {
        // Find the position of row index within the row indices if it exits.
        int idx = SparseElementSearch.matrixBinarySearch(rowIndices, colIndices, row, col);
        W[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
            idx = -idx - 1;

            // No non-zero-element with these indices exists. Insert new value.
            destEntries = makeEmptyDataArray(data.length + 1);
            destRowIndices = new int[data.length + 1];
            destColIndices = new int[data.length + 1];

            CooGetSet.cooInsertNewValue(
                    value, row, col,
                    data, rowIndices, colIndices, idx,
                    destEntries, destRowIndices, destColIndices);
        } else {
            // Value with these indices exists. Simply update the value.
            destEntries = Arrays.copyOf(data, data.length);
            destEntries[idx] = value;
            destRowIndices = rowIndices.clone();
            destColIndices = colIndices.clone();
        }

        return makeLikeNDArray(shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a specified row of this matrix to a vector.
     *
     * @param row Vector containing data to replace the specified row in this matrix.
     * @param rowIdx Index of the row to set.
     *
     * @return If this matrix is dense, the row set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse, a copy will be created with the new row and returned.
     */
    @Override
    public T setRow(V row, int rowIdx) {
        SparseMatrixData<W> dest = CooGetSet.setRow(
                shape, data, rowIndices, colIndices,
                rowIdx, row.size, row.data, row.indices);
        return makeLikeNDArray(dest.shape(), dest.data(), dest.rowData(), dest.colData());
    }


    /**
     * Sets a column of this matrix at the given index to the specified vector.
     *
     * @param col Vector containing new column data.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A copy of this matrix with the specified column set to {@code col}.
     *
     * @throws IllegalArgumentException If the {@code col} vector has a different length than the number of rows in this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex < 0 || colIndex >= this.numCols}.
     */
    public T setCol(V col, int colIndex) {
        SparseMatrixData<W> dest = CooGetSet.setCol(
                shape, data, rowIndices, colIndices,
                colIndex, col.size, col.data, col.indices);
        CooDataSorter sorter = new CooDataSorter(dest.data(), dest.rowData(), dest.colData()).sparseSort();
        return makeLikeNDArray(dest.shape(), dest.data(), dest.rowData(), dest.colData());
    }


    /**
     * Flattens this matrix to a single row.
     *
     * @return The flattened matrix.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return flatten(1);
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
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = {1, 1};
        dims[axis] = shape.numelIntValueExact();
        Shape flatShape = new Shape(dims);

        int[] destIndices = new int[data.length];

        for(int i = 0; i < data.length; i++)
            destIndices[i] = shape.get1DIndex(rowIndices[i], colIndices[i]);

        return (axis == 0)
                ? makeLikeNDArray(flatShape, data.clone(), destIndices, new int[data.length])
                : makeLikeNDArray(flatShape, data.clone(), new int[data.length], destIndices);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws NDArrayShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        ValidateParameters.ensureTotalEntriesEqual(shape, newShape);
        int oldColCount = shape.getSize(1);
        int newColCount = newShape.getSize(1);

        // Initialize new COO structures with the same size as the original.
        int[] newRowIndices = new int[rowIndices.length];
        int[] newColIndices = new int[colIndices.length];

        for (int i = 0; i < rowIndices.length; i++) {
            int flatIndex = rowIndices[i]*oldColCount + colIndices[i];
            newRowIndices[i] = flatIndex / newColCount;
            newColIndices[i] = flatIndex % newColCount;
        }

        return makeLikeNDArray(newShape, data.clone(), newRowIndices, newColIndices);
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
        T transpose = makeLikeNDArray(shape.swapAxes(0, 1), data.clone(), colIndices.clone(), rowIndices.clone());
        transpose.sortIndices(); // Ensure the indices are sorted correctly.

        return transpose;
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
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return copy();
        else return T();
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
    public T T(int... axes) {
        if(axes.length != 2)
            throw new IllegalArgumentException("Expecting two axes in transpose but got " + axes.length + ".");
        return T(axes[0], axes[1]);
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
     * @param row The row index of the item to get from this matrix.
     * @param col Column index of the item to get from this matrix.
     *
     * @return The element of this matrix at the specified index.
     */
    @Override
    public W get(int row, int col) {
        W value = (W) CooGetSet.getCoo(data, rowIndices, colIndices, row, col);
        return (value == null) ? getZeroElement() : value;
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
    public W tr() {
        W trace = getZeroElement();

        for(int i = 0; i< data.length; i++)
            if(rowIndices[i]==colIndices[i]) trace = trace.add(data[i]); // Then entry is on the diagonal.

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
        for(int i = 0; i< data.length; i++)
            if(rowIndices[i] > colIndices[i] && !data[i].isZero())
                return false; // Then a non-zero entry is not in the upper triangle.

        return true;
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
        for(int i = 0; i< data.length; i++)
            if(rowIndices[i] < colIndices[i] && !data[i].isZero())
                return false; // Then the non-zero entry is not in the lower triangle.

        return true;
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return {@code true} if this matrix is the identity matrix; {@code false} otherwise.
     */
    @Override
    public boolean isI() {
        return CooSemiringMatrixProperties.isIdentity(shape, data, rowIndices, colIndices);
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
    public U mult(T b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        W[] dest = makeEmptyDataArray(numRows*b.numCols);
        CooSemiringMatMult.standard(
                data, rowIndices, colIndices, shape,
                b.data, b.rowIndices, b.colIndices, b.shape, dest);

        return makeLikeDenseNDArray(new Shape(numRows, b.numCols), dest);
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
    public U multTranspose(T b) {
        ValidateParameters.ensureAllEqual(numCols, b.numCols);
        return mult(b.H());
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
     * @see #augment(T)
     */
    @Override
    public T stack(T b) {
        ValidateParameters.ensureAllEqual(numCols, b.numCols);

        Shape destShape = new Shape(numRows+b.numRows, numCols);
        W[] destEntries = makeEmptyDataArray(data.length + b.data.length);
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];
        CooConcat.stack(data, rowIndices, colIndices, numRows,
                b.data, b.rowIndices, b.colIndices,
                destEntries, destRowIndices, destColIndices);

        return makeLikeNDArray(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(T)
     * @see #stack(MatrixMixin, int) 
     */
    @Override
    public T augment(T b) {
        ValidateParameters.ensureAllEqual(numRows, b.numRows);

        Shape destShape = new Shape(numRows, numCols + b.numCols);
        W[] destEntries = makeEmptyDataArray(data.length + b.data.length);
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];
        CooConcat.augment(data, rowIndices, colIndices, numCols,
                b.data, b.rowIndices, b.colIndices,
                destEntries, destRowIndices, destColIndices);

        return makeLikeNDArray(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public T augment(V b) {
        ValidateParameters.ensureAllEqual(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        W[] destEntries = makeEmptyDataArray(nnz + b.data.length);
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];
        CooConcat.augmentVector(
                data, rowIndices, colIndices, numCols,
                b.data, b.indices,
                destEntries, destRowIndices, destColIndices);

        return makeLikeNDArray(destShape, destEntries, destRowIndices, destColIndices);
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
    public T swapRows(int rowIndex1, int rowIndex2) {
        CooManipulations.swapRows(shape, data, rowIndices, colIndices, rowIndex1, rowIndex2);
        return (T) this;
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
    public T swapCols(int colIndex1, int colIndex2) {
        CooManipulations.swapCols(shape, data, rowIndices, colIndices, colIndex1, colIndex2);
        return (T) this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return {@code true} if this matrix is symmetric; {@code false} otherwise.
     */
    @Override
    public boolean isSymmetric() {
        return CooProperties.isSymmetric(shape, data, rowIndices, colIndices, zeroElement);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    @Override
    public boolean isHermitian() {
        return isSymmetric();
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return {@code true} if this matrix it is orthogonal; {@code false} otherwise.
     */
    @Override
    public boolean isOrthogonal() {
        if(isSquare()) return mult(T()).isI();
        else return false;
    }


    /**
     * Gets a range of a row in this matrix.
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
    public V getRow(int rowIdx, int start, int stop) {
        SparseVectorData<W> data = CooGetSet.getRow(shape, this.data, rowIndices, colIndices, rowIdx, start, stop);
        W[] dest = makeEmptyDataArray(data.data().size());
        data.data().toArray(dest);

        return makeLikeVector(data.shape(), dest, data.indicesToArray());
    }


    /**
     * Gets a range of a column in this matrix.
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
    public V getCol(int colIdx, int start, int stop) {
        SparseVectorData<W> data = CooGetSet.getCol(shape, this.data, rowIndices, colIndices, colIdx, start, stop);
        W[] dest = makeEmptyDataArray(data.data().size());
        data.data().toArray(dest);
        return makeLikeVector(data.shape(), dest, data.indicesToArray());
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
    public V getDiag(int diagOffset) {
        SparseVectorData<W> data = CooGetSet.getDiag(shape, this.data, rowIndices, colIndices, diagOffset);
        W[] dest = makeEmptyDataArray(data.data().size());
        data.data().toArray(dest);
        return makeLikeVector(data.shape(),
                dest,
                data.indicesToArray());
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public T removeRow(int rowIndex) {
        ValidateParameters.validateArrayIndices(numRows, rowIndex);
        Shape shape = new Shape(numRows-1, numCols);

        // Find the start and end index within the data array which have the given row index.
        IntPair startEnd = SparseElementSearch.matrixFindRowStartEnd(rowIndices, rowIndex);
        int start = startEnd.first();
        int end = startEnd.second();

        int size = data.length - (end - start);

        // Initialize arrays.
        W[] entries = makeEmptyDataArray(size);
        int[] rowIndices = new int[size];
        int[] colIndices = new int[size];
        copyRanges(this.data, this.rowIndices, this.colIndices, entries, rowIndices, colIndices, startEnd);

        // Shift all row indices occurring after removed row.
        if (start > 0) {
            for(int i=start, length=rowIndices.length; i<rowIndices.length; i++)
                rowIndices[i]--;
        } else {
            for(int i=0, length=rowIndices.length; i<rowIndices.length; i++) {
                if(rowIndices[i] > rowIndex)
                    rowIndices[i]--;
            }
        }

        return makeLikeNDArray(shape, entries, rowIndices, colIndices);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIdxs The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIdxs) {
        ValidateParameters.validateArrayIndices(numRows, rowIdxs);
        // Ensure the indices are sorted.
        Arrays.sort(rowIdxs);

        Shape shape = new Shape(numRows - rowIdxs.length, numCols);
        List<W> entries = new ArrayList<>(nnz);
        List<Integer> newRowIndices = new ArrayList<>(nnz);
        List<Integer> newColIndices = new ArrayList<>(nnz);

        int j = 0; // Points into the rowIdxs array
        int removeCount = 0; // Tracks the number of removed rows.

        for (int i = 0; i < nnz; i++) {
            int oldRow = rowIndices[i];

            // Advance j while rowIdxs[j] < oldRow, updating removeCount
            while (j < rowIdxs.length && rowIdxs[j] < oldRow) {
                removeCount++;
                j++;
            }

            // If oldRow is one of the removed rows, skip this entry.
            if (j < rowIdxs.length && rowIdxs[j] == oldRow)
                continue;

            // Otherwise, shift oldRow by however many removed rows lie below it.
            int newRow = oldRow - removeCount;

            // Keep the entry
            entries.add(data[i]);
            newRowIndices.add(newRow);
            newColIndices.add(colIndices[i]);
        }

        return makeLikeNDArray(shape, entries, newRowIndices, newColIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCol(int colIndex) {
        ValidateParameters.validateArrayIndices(numRows, colIndex);

        Shape shape = new Shape(numRows, numCols-1);
        List<W> destEntries = new ArrayList<>(data.length);
        List<Integer> destRowIndices = new ArrayList<>(data.length);
        List<Integer> destColIndices = new ArrayList<>(data.length);

        for(int i = 0; i< data.length; i++) {
            if(colIndices[i] != colIndex) {
                // Then the entry is not in the specified column, so remove it.
                destEntries.add(data[i]);
                destRowIndices.add(rowIndices[i]);

                if(colIndices[i] < colIndex) destColIndices.add(colIndices[i]);
                else destColIndices.add(colIndices[i]-1);
            }
        }

        return makeLikeNDArray(shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIdxs Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCols(int... colIdxs) {
        ValidateParameters.validateArrayIndices(numRows, colIdxs);

        // Ensure the indices are sorted.
        Arrays.sort(colIdxs);

        Shape shape = new Shape(numRows, numCols - colIdxs.length);
        List<W> destEntries = new ArrayList<>(data.length);
        List<Integer> destRowIdx = new ArrayList<>(data.length);
        List<Integer> destColIdx = new ArrayList<>(data.length);

        for (int i = 0; i < data.length; i++) {
            int oldCol = colIndices[i];

            // Check if oldCol is being removed.
            int idx = Arrays.binarySearch(colIdxs, oldCol);

            // If idx >= 0, oldCol is in colIdxs then skip this entry.
            if (idx >= 0) continue;

            // Otherwise, shift column index.
            int insertionPoint = -idx - 1;
            int newCol = oldCol - insertionPoint;

            destEntries.add(data[i]);
            destRowIdx.add(rowIndices[i]);
            destColIdx.add(newCol);
        }

        return makeLikeNDArray(shape, destEntries, destRowIdx, destColIdx);
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
    public T setSliceCopy(T values, int rowStart, int colStart) {
        SparseMatrixData<W> sliceData = CooGetSet.setSlice(
                shape, data, rowIndices, colIndices,
                values.shape, values.data, values.rowIndices, values.colIndices,
                rowStart, colStart);
        return makeLikeNDArray(sliceData.shape(), sliceData.data(), sliceData.rowData(), sliceData.colData());
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
    public T getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        SparseMatrixData<W> sliceData = CooGetSet.getSlice(
                shape, data, rowIndices, colIndices,
                rowStart, rowEnd, colStart, colEnd);
        return makeLikeNDArray(sliceData.shape(), sliceData.data(), sliceData.rowData(), sliceData.colData());
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
    public T getTriU(int diagOffset) {
        SparseMatrixData<W> data = CooGetSet.getTriU(diagOffset, shape, this.data, rowIndices, colIndices);
        return makeLikeNDArray(data.shape(),  data.data(), data.rowData(), data.colData());
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
    public T getTriL(int diagOffset) {
        SparseMatrixData<W> data = CooGetSet.getTriL(diagOffset, shape, this.data, rowIndices, colIndices);
        return makeLikeNDArray(data.shape(),  data.data(), data.rowData(), data.colData());
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeNDArray(shape, data.clone());
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
    public T map(UnaryOperator<W> mapper) {
        ArrayMapper.map(data, mapper);
        return (T) this;
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
    public W reduce(W identity, BinaryOperator<W> accumulator) {
        return ArrayReducer.reduce(data, identity, accumulator);
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
    public ArrayMask where(Function<W, Boolean> predicate) {
        BitSet mask = new BitSet(data.length);

        for(int i=0, size=data.length; i<size; i++)
            mask.set(i, predicate.apply(data[i]));

        return new ArrayMask(shape, mask);
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
    public boolean any(Function<W, Boolean> predicate) {
        for(W v : data)
            if (predicate.apply(v)) return true;

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
    public boolean all(Function<W, Boolean> predicate) {
        for(W v : data)
            if (!predicate.apply(v)) return false;

        return true;
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
    public int countTrue(Function<W, Boolean> predicate) {
        int count = 0;

        for (W v : data)
            if (predicate.apply(v)) count++;

        return count;
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
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex-valued tensor) in this tensor.
     */
    @Override
    public W min() {
        return CompareSemiring.min(data);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex-valued tensor) in this tensor.
     */
    @Override
    public W max() {
        return CompareSemiring.max(data);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        int idx = CompareSemiring.argmin(data);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        int idx = CompareSemiring.argmax(data);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws NDArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        SparseMatrixData<W> data = CooSemiringMatrixOps.add(
                shape, this.data, rowIndices, colIndices,
                b.shape, b.data, b.rowIndices, b.colIndices);

        return makeLikeNDArray(data.shape(), data.data(), data.rowData(), data.colData());
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
        SparseMatrixData<W> data = CooSemiringMatrixOps.elemMult(
                shape, this.data, rowIndices, colIndices,
                b.shape, b.data, b.rowIndices, b.colIndices);

        return makeLikeNDArray(data.shape(), data.data(), data.rowData(), data.colData());
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
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensor's rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e., the axes are equal, or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);

        return makeLikeNDArray(new Shape(1, 1), (W[]) new Semiring[]{tr()}, new int[]{0}, new int[]{0});
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public T sortIndices() {
        CooDataSorter.wrap(data, rowIndices, colIndices).sparseSort().unwrap(data, rowIndices, colIndices);
        return (T) this;
    }


    /**
     * Converts this sparse COO matrix to an equivalent dense matrix.
     * @return A dense matrix equivalent to this sparse COO matrix.
     */
    public U toDense() {
        W[] entries = makeEmptyDataArray(shape.numelIntValueExact());
        Arrays.fill(entries, getZeroElement());

        for(int i = 0; i< nnz; i++)
            entries[rowIndices[i]*numCols + colIndices[i]] = data[i];

        return makeLikeDenseNDArray(shape, entries);
    }


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    public AbstractCsrSemiringMatrix<?, U, V, W> toCsr() {
        W[] csrEntries = makeEmptyDataArray(data.length);
        int[] csrRowPointers = new int[numRows + 1];
        int[] csrColPointers = new int[colIndices.length];
        CooConversions.toCsr(shape, data, rowIndices, colIndices, csrEntries, csrRowPointers, csrColPointers);
        return makeLikeCsrMatrix(shape, csrEntries, csrRowPointers, csrColPointers);
    }


    /**
     * Converts this matrix to an equivalent rank 2 tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public abstract AbstractCooSemiringTensor<?, ?, W> toTensor();


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    public abstract AbstractCooSemiringTensor<?, ?, W> toTensor(Shape newShape);


    /**
     * Converts this sparse CSR matrix to an equivalent vector.
     * If this matrix is not a row or column vector, it will be flattened before conversion.
     * @return A vector equivalent to this CSR matrix.
     */
    public V toVector() {
        int[] destIndices = new int[data.length];
        for(int i = 0; i< data.length; i++)
            destIndices[i] = rowIndices[i]*numCols + colIndices[i];

        return makeLikeVector(new Shape(numRows*numCols), data.clone(), destIndices);
    }


    /**
     * Coalesces this sparse COO matrix. An uncoalesced matrix is a sparse matrix with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by summing up duplicated data. If another form of aggregation other
     * than summation is desired, use {@link #coalesce(BinaryOperator)}.
     * @return A new coalesced sparse COO matrix which is equivalent to this COO matrix.
     * @see #coalesce(BinaryOperator) 
     */
    public T coalesce() {
        return coalesce(Semiring::add);
    }


    /**
     * Coalesces this sparse COO matrix. An uncoalesced matrix is a sparse matrix with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data using {@code aggregator}.
     * @param aggregator Custom aggregation function to combine multiple.
     * @return A new coalesced sparse COO matrix which is equivalent to this COO matrix.
     * @see #coalesce() 
     */
    public T coalesce(BinaryOperator<W> aggregator) {
        SparseMatrixData<W> mat = SparseUtils.coalesce(aggregator, shape, data, rowIndices, colIndices);
        return makeLikeNDArray(mat.shape(), mat.data(), mat.rowData(), mat.colData());
    }


    /**
     * Drops any explicit zeros in this sparse COO matrix.
     * @return A copy of this COO matrix with any explicitly stored zeros removed.
     */
    public T dropZeros() {
        SparseMatrixData<W> mat = SparseUtils.dropZeros(shape, data, rowIndices, colIndices);
        return makeLikeNDArray(mat.shape(), mat.data(), mat.rowData(), mat.colData());
    }
}
