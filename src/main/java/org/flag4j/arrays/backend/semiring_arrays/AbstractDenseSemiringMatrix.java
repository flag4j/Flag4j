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


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.dense.DenseOps;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringConversions;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMultDispatcher;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;


/**
 * The base class for all dense matrices whose elements are members of a {@link Semiring}.
 * @param <T> The type of this matrix.
 * @param <U> The type of the vector which is of similar type to {@code T}.
 * @param <V> The type of the semiring the data of the matrix belongs to.
 */
public abstract class AbstractDenseSemiringMatrix<T extends AbstractDenseSemiringMatrix<T, U, V>,
        U extends AbstractDenseSemiringVector<U, T, V>, V extends Semiring<V>>
        extends AbstractDenseSemiringTensor<T, V> implements MatrixMixin<T, T, U, V> {

    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseSemiringMatrix(Shape shape, V[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        numRows = shape.getSize(0);
        numCols = shape.getSize(1);
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     * @param shape Shape of the vector to construct. Must be rank 1.
     * @param entries Entries of the vector.
     * @return A vector of a similar type as this matrix.
     */
    protected abstract U makeLikeVector(Shape shape, V[] entries);


    /**
     * Constructs a vector of a similar type as this matrix.
     * @param entries Entries of the vector.
     * @return A vector of a similar type as this matrix.
     */
    protected abstract U makeLikeVector(V[] entries);


    /**
     * Constructs a sparse COO matrix, which is of a similar type as this dense matrix.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @return A sparse COO matrix which is of a similar type as this dense matrix.
     */
    protected abstract AbstractCooSemiringMatrix<?, T, ?, V> makeLikeCooMatrix(
            Shape shape, V[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Constructs a sparse CSR matrix, which is of a similar type as this dense matrix.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A sparse CSR matrix which is of a similar type as this dense matrix.
     */
    public abstract AbstractCsrSemiringMatrix<?, T, ?, V> makeLikeCsrMatrix(
            Shape shape, V[] entries, int[] rowPointers, int[] colIndices);


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
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatch(data, shape, dest);
        return makeLikeNDArray(shape.swapAxes(0, 1), dest);
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
     * Gets the length of the data array which backs this matrix.
     *
     * @return The length of the data array which backs this matrix.
     */
    @Override
    public int dataLength() {
        return data.length;
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
    public V get(int row, int col) {
        return (V) data[row*numCols + col];
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
    public V tr() {
        ValidateParameters.ensureSquareMatrix(shape);
        V sum = data[0];
        int colsOffset = this.numCols + 1;

        for(int i=1; i<numRows; i++)
            sum = sum.add((V) data[i*colsOffset]);

        return (V) sum;
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

        // Ensure the lower half is zeros.
        for(int i=1; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<i; j++)
                if(!data[rowOffset + j].isZero()) return false; // No need to continue.
        }

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
        if(!isSquare()) return false;

        // Ensure upper half is zeros.
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=i+1; j<numCols; j++)
                if(!data[rowOffset + j].isZero()) return false; // No need to continue.
        }

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
        int pos = 0;

        if(isSquare()) {
            for(int i=0; i<numRows; i++) {
                for(int j=0; j<numCols; j++) {
                    if((i==j && !data[pos].isOne()) || i!=j && !data[pos].isZero()) {
                        return false; // No need to continue
                    }

                    pos++;
                }
            }
        } else {
            return false; // An identity matrix must be square.
        }

        // If we make it to this point, this matrix must be an identity matrix.
        return true;
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
    public U mult(U b) {
        V[] dest = makeEmptyDataArray(numRows);
        DenseSemiringMatMultDispatcher.dispatchVector(data, shape, b.data, b.shape, dest);
        return makeLikeVector(new Shape(numRows), dest);
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
    public T mult(T b) {
        V[]  dest = makeEmptyDataArray(numRows*b.numCols);
        DenseSemiringMatMultDispatcher.dispatch(data, shape, b.data, b.shape, dest);
        return makeLikeNDArray(new Shape(numRows, b.numCols), dest);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method <em>may</em>, be noticeably faster than directly computing the transpose followed by the
     * multiplication as {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public T multTranspose(T b) {
        V[]  dest = makeEmptyDataArray(numRows*b.numRows);
        DenseSemiringMatMultDispatcher.dispatchTranspose(data, shape, b.data, b.shape, dest);
        return makeLikeNDArray(new Shape(numRows, b.numRows), dest);
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
        ValidateParameters.ensureArrayLengthsEq(this.numCols, b.numCols);
        Shape stackedShape = new Shape(this.numRows + b.numRows, this.numCols);
        V[]  stackedEntries = makeEmptyDataArray(stackedShape.numel().intValueExact());

        System.arraycopy(this.data, 0, stackedEntries, 0, this.data.length);
        System.arraycopy(b.data, 0, stackedEntries, this.data.length, b.data.length);

        return makeLikeNDArray(stackedShape, stackedEntries);
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
     * @see #stack(AbstractDenseSemiringMatrix)
     */
    @Override
    public T augment(T b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.numRows);

        int augNumCols = numCols + b.numCols;
        Shape augShape = new Shape(numRows, augNumCols);
        V[]  augEntries = makeEmptyDataArray(numRows*augNumCols);

        // Copy data from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(data, i*numCols, augEntries, i*augNumCols, numCols);
            int augOffset = i*augNumCols + numCols;
            int bOffset = i*b.numCols;

            for(int j=0; j<b.numCols; j++)
                augEntries[augOffset + j] = b.data[bOffset + j];
        }

        return makeLikeNDArray(augShape, augEntries);
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public T augment(U b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.size);
        V[]  augmented = makeEmptyDataArray(numRows*(numCols + 1));

        // Copy data from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(data, i*numCols, augmented, i*(numCols+1), numCols);
            augmented[i*(numCols+1) + numCols] = b.data[i];
        }

        return makeLikeNDArray(new Shape(numRows, numCols+1), augmented);
    }


    /**
     * Swaps specified rows in the matrix. This is done in-place.
     *
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public T swapRows(int rowIndex1, int rowIndex2) {
        DenseOps.swapRows(shape, data, rowIndex1, rowIndex2);
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
        ValidateParameters.validateArrayIndices(numCols, colIndex1, colIndex2);

        if(colIndex1 != colIndex2) {
            V temp;

            for(int i=0; i<numRows; i++) {
                // Swap elements.
                int idx = i*numCols;
                ArrayUtils.swap(data, idx + colIndex1, idx + colIndex2);
            }
        }

        return (T) this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return {@code true} if this matrix is symmetric; {@code false} otherwise.
     */
    @Override
    public boolean isSymmetric() {
        if(this==null) return false;
        if(this.data.length==0) return true;

        return numRows==numCols && this.equals(this.T());
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
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is approximately equal to its transpose.
     *
     * @return {@code true} if this matrix it is orthogonal; {@code false} otherwise.
     */
    @Override
    public boolean isOrthogonal() {
        return numRows == numCols && mult(T()).isI();
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
        Shape copyShape = new Shape(numRows-1, numCols);
        V[]  copyEntries = makeEmptyDataArray((numRows-1)*numCols);
        int row = 0;

        for(int i=0; i<numRows; i++) {
            if(i!=rowIndex) {
                System.arraycopy(data, i*numCols, copyEntries, row*numCols, numCols);
                row++;
            }
        }

        return makeLikeNDArray(new Shape(numRows-1, numCols), copyEntries);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIndices) {
        Shape copyShape = new Shape(numRows-rowIndices.length, numCols);
        V[]  copyEntries = makeEmptyDataArray((numRows-rowIndices.length)*numCols);
        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(ArrayUtils.notContains(rowIndices, i)) {
                System.arraycopy(this.data, i*numCols, copyEntries, row*numCols, numCols);
                row++;
            }
        }

        return makeLikeNDArray(new Shape(numRows-rowIndices.length, numCols), copyEntries);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCol(int colIndex) {
        int copyNumCols = numCols-1;
        Shape copyShape = new Shape(numRows, copyNumCols);
        V[]  copyEntries = makeEmptyDataArray(numRows*copyNumCols);

        for(int i=0; i<this.numRows; i++) {
            int rowOffset = i*numCols;
            int copyOffset = i*copyNumCols;
            int  col = 0;

            for(int j=0; j<this.numCols; j++) {
                if(j!=colIndex) {
                    copyEntries[copyOffset + col] = this.data[rowOffset + j];
                    col++;
                }
            }
        }

        return makeLikeNDArray(new Shape(numRows, numCols-1), copyEntries);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCols(int... colIndices) {
        int copyNumCols = this.numCols-colIndices.length;
        Shape copyShape = new Shape(numRows, copyNumCols);
        V[]  copyEntries = makeEmptyDataArray(numRows*copyNumCols);

        for(int i=0; i<this.numRows; i++) {
            int rowOffset = i*numCols;
            int copyOffset = i*copyNumCols;
            int col = 0;

            for(int j=0; j<this.numCols; j++) {
                if(ArrayUtils.notContains(colIndices, j)) {
                    copyEntries[copyOffset + col] = this.data[rowOffset + j];
                    col++;
                }
            }
        }

        return makeLikeNDArray(new Shape(numRows, numCols-colIndices.length), copyEntries);
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this vector,
     * or if {@code values} does not fit within this vector when its first entry is placed at {@code startIdx}.
     */
    public T setSlice(T values, int rowStart, int colStart) {
        ValidateParameters.validateArrayIndices(numRows, rowStart);
        ValidateParameters.validateArrayIndices(numCols, colStart);

        for(int i=0; i<values.numRows; i++) {
            int src1Offset = (i+rowStart)*numCols + colStart;
            int src2RowOffset = i*values.numCols;

            for(int j=0; j<values.numCols; j++) {
                this.data[src1Offset + j] = values.data[src2RowOffset + j];
            }
        }

        return (T) this;
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
        return copy().setSlice(values, rowStart, colStart);
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
        ValidateParameters.validateArrayIndices(numRows, rowStart, rowEnd);
        ValidateParameters.validateArrayIndices(numCols, colStart, colEnd);

        int sliceRows = rowEnd-rowStart;
        int sliceCols = colEnd-colStart;
        int destPos = 0;
        V[]  slice = makeEmptyDataArray(sliceRows*sliceCols);

        for(int i=rowStart; i<rowEnd; i++) {
            int srcPos = i*numCols + colStart;
            int end = srcPos + colEnd - colStart;

            while(srcPos < end)
                slice[destPos++] = data[srcPos++];
        }

        return makeLikeNDArray(new Shape(sliceRows, sliceCols), slice);
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
    public T set(V value, int row, int col) {
        return super.set(value, row, col);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different shape then this matrix.
     */
    public T setValues(V[][] values) {
        ValidateParameters.ensureAllEqual(numRows, values.length);
        ValidateParameters.ensureAllEqual(numCols, values[0].length);

        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<numCols; j++)
                data[rowOffset + j] = values[i][j];
        }

        return (T) this;
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
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        V[]  copyEntries = makeEmptyDataArray(data.length);
        Arrays.fill(copyEntries, (data.length > 0) ? data[0].getZero() : null);
        T result = makeLikeNDArray(shape, copyEntries);

        // Extract the upper triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=Math.max(0, i + diagOffset); j<numCols; j++) {
                if (j >= i + diagOffset)
                    result.data[rowOffset + j] = data[rowOffset + j];
            }
        }

        return result;
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
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        V[]  copyEntries = makeEmptyDataArray(data.length);
        Arrays.fill(copyEntries, (data.length > 0) ? data[0].getZero() : null);
        T result = makeLikeNDArray(shape, copyEntries);

        // Extract the lower triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j <= Math.min(numCols - 1, i + diagOffset); j++) {
                if(j <= i + diagOffset) {
                    result.data[rowOffset + j] = this.data[rowOffset + j];
                }
            }
        }

        return result;
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
    public U getDiag(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -(numRows-1), numCols-1, "diagOffset");


        // Check for some quick returns.
        if(numRows == 1 && diagOffset > 0) {
            V[] dest = makeEmptyDataArray(1);
            dest[0] = data[diagOffset];
            return makeLikeVector(shape, dest);
        }
        if(numCols == 1 && diagOffset < 0) {
            V[] dest = makeEmptyDataArray(1);
            dest[0] = data[-diagOffset];
            return makeLikeVector(shape, dest);
        }

        // Compute the length of the diagonal.
        int newSize = Math.min(numRows, numCols);
        int idx = 0;

        if(diagOffset > 0) {
            newSize = Math.min(newSize, numCols - diagOffset);
            idx = diagOffset;
        }
        else if(diagOffset < 0) {
            newSize = Math.min(newSize, numRows + diagOffset);
            idx = -diagOffset*numCols;
        }

        V[]  diag = makeEmptyDataArray(newSize);

        for(int i=0; i<newSize; i++) {
            diag[i] = this.data[idx];
            idx += numCols + 1;
        }

        return makeLikeVector(shape, diag);
    }


    /**
     * Sets a specified row of this matrix to a vector.
     *
     * @param row Vector to replace the specified row in this matrix.
     * @param rowIdx Index of the row to set.
     *
     * @return If this matrix is dense, the row set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse, a copy will be created with the new row and returned.
     */
    @Override
    public T setRow(U row, int rowIdx) {
        return setRow(row.data, rowIdx);
    }


    /**
     * Sets a specified row of this matrix to an array.
     *
     * @param row Array containing values to replace the specified row in this matrix.
     * @param rowIdx Index of the row to set.
     *
     * @return If this matrix is dense, the row set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse, a copy will be created with the new row and returned.
     */
    public T setRow(V[] row, int rowIdx) {
        ValidateParameters.ensureArrayLengthsEq(row.length, this.numCols);

        for(int i=0, size=row.length, rowOffset=rowIdx*numCols; i<size; i++)
            super.data[rowOffset + i] = row[i];

        return (T) this;
    }


    /**
     * Sets a specified column of this matrix to a vector.
     *
     * @param col Vector to replace the specified column in this matrix.
     * @param colIdx Index of the column to set.
     *
     * @return If this matrix is dense, the column set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse, a copy will be created with the new column and returned.
     */
    @Override
    public T setCol(U col, int colIdx) {
        return setCol(col.data, colIdx);
    }


    /**
     * Sets a specified column of this matrix to an array.
     *
     * @param col Vector to replace the specified column in this matrix.
     * @param colIdx Index of the column to set.
     *
     * @return If this matrix is dense, the column set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse, a copy will be created with the new column and returned.
     */
    public T setCol(V[] col, int colIdx) {
        ValidateParameters.ensureArrayLengthsEq(col.length, this.numRows);

        int rowOffset = 0;
        for(int i=0, size=col.length; i<size; i++) {
            super.data[rowOffset + colIdx] = col[i];
            rowOffset += numCols;
        }

        return (T) this;
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
    @Override
    public U getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.validateArrayIndices(numCols, colStart, colEnd-1);
        ValidateParameters.ensureGreaterEq(colStart, colEnd);
        int start = rowIdx*numCols + colStart;
        int stop = rowIdx*numCols + colEnd;

        V[] row = Arrays.copyOfRange(data, start, stop);

        return makeLikeVector(new Shape(colEnd-colStart), row);
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
    @Override
    public U getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.validateArrayIndices(numRows, rowStart, rowEnd-1);
        ValidateParameters.ensureGreaterEq(rowStart, rowEnd);
        V[] col = makeEmptyDataArray(rowEnd-rowStart);

        for(int i=rowStart; i<rowEnd; i++)
            col[i] = data[i*numCols + colIdx];

        return makeLikeVector(new Shape(rowEnd-rowStart), col);
    }


    /**
     * Flattens this matrix to a row vector.
     *
     * @return The flattened matrix.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return flatten(0);
    }


    /**
     * Flattens this matrix along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @return If {@code axis == 0} a matrix with the shape {@code (1, this.numRows*this.numCols)} is returned.
     * If {@code axis == 1} a matrix with the shape {@code (this.numRows*this.numCols, 1)} is returned.
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        return (axis == 0)
                ? makeLikeNDArray(new Shape(1, data.length), data.clone())
                : makeLikeNDArray(new Shape(data.length, 1), data.clone());
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        // TODO: This should probably be removed from semiring tensors.
        return T(); // Conjugation is not defined on a general semiring. Fall back to standard transpose.
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     * @see #toCoo(double) 
     */
    public AbstractCooSemiringMatrix<?, ?, ?, V> toCoo() {
        return toCoo(0.9);
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the primary concern, it is better to underestimate the sparsity.
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     * @see #toCoo() 
     */
    public AbstractCooSemiringMatrix<?, ?, ?, V> toCoo(double estimatedSparsity) {
        SparseMatrixData<V> data = DenseSemiringConversions.toCoo(shape, this.data, estimatedSparsity);
        V[] cooEntries = makeEmptyDataArray(data.data().size());
        data.data().toArray(cooEntries);
        int[] rowIndices = ArrayConversions.fromIntegerList(data.rowData());
        int[] colIndices = ArrayConversions.fromIntegerList(data.colData());

        return makeLikeCooMatrix(data.shape(), cooEntries, rowIndices, colIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     * @see #toCsr(double)
     */
    public AbstractCsrSemiringMatrix<?, ?, ?, V> toCsr() {
        return toCoo(0.9).toCsr();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the primary concern, it is better to underestimate the sparsity.
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     * @see #toCsr()
     */
    public AbstractCsrSemiringMatrix<?, ?, ?, V> toCsr(double estimatedSparsity) {
        return toCoo(estimatedSparsity).toCsr();
    }


    /**
     * Converts this matrix to an equivalent vector.
     * If this matrix is not a row or column vector, it will first be flattened, then converted to a vector.
     *
     * @return A vector which contains the same data as this matrix.
     */
    @Override
    public U toVector() {
        return makeLikeVector(new Shape(numRows*numCols), data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor.
     * @return A tensor with the same shape and data as this matrix.
     */
    public abstract AbstractDenseSemiringTensor<?, V> toTensor();


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     * @param newShape Shape of the tensor. Can be any rank but must have the same number of total entries as the shape of this matrix.
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    public abstract AbstractDenseSemiringTensor<?, V> toTensor(Shape newShape);
}
