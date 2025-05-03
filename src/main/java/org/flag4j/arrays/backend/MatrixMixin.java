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

package org.flag4j.arrays.backend;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * <p>The {@code MatrixMixin} interface defines methods that any matrix implementation must support.
 * This interface is designed to ensure a consistent API for various matrix operations, regardless of the underlying matrix
 * type (dense or sparse). It includes methods for accessing matrix properties, performing mathematical operations,
 * and manipulating matrix elements and substructures.
 *
 * <p>This interface is intended to be used as a mixin for a class which also extends {@link AbstractTensor}.
 *
 * @param <T> The type of the implementing matrix class.
 * @param <U> The type of a dense matrix that is analogous to {@code T}.
 * If {@code T} represents a dense matrix, then {@code T} and {@code U} should be the same type.
 * @param <V> The type of vector analogous to {@code T}.
 * @param <W> The type of an individual matrix element, which may be a primitive type, wrapper class, or a custom object.
 */
public interface MatrixMixin<T extends MatrixMixin<T, U, V, W>,
        U extends MatrixMixin<U, U, ?, W>,
        V extends VectorMixin<V, ?, U, W>, W> {

    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    int numRows();


    /**
     * Gets the number of columns in this matrix.
     * @return The number of columns in this matrix.
     */
    int numCols();


    /**
     * Gets the length of the data array which backs this matrix.
     * @return The length of the data array which backs this matrix.
     */
    int dataLength();


    /**
     * Gets the element of this matrix at this specified {@code row} and {@code col}.
     * @param row Row index of the item to get from this matrix.
     * @param col Column index of the item to get from this matrix.
     * @return The element of this matrix at the specified index.
     */
    public W get(int row, int col);


    /**
     * Gets the shape of this matrix.
     * @return The shape of this matrix.
     */
    Shape getShape();


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     *
     * <p>Same as {@link #tr()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    default W trace() {return tr();}


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     *
     * <p>Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    W tr();


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e., the number of rows equals the number of columns). Otherwise, returns false.
     */
    default boolean isSquare() {
        return numRows()==numCols();
    }


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     *
     * @return True if this matrix can be represented as either a row or column vector.
     */
    default boolean isVector() {
        return numRows()<=1 || numCols()<=1;
    }


    /**
     * Checks what type of vector this matrix is. i.e., not a vector, a 1x1 matrix, a row vector, or a column vector.
     *
     * @return An int corresponding to the type of vector this matrix represents:
     * <ul>
     *     <li>If this matrix can not be represented as a vector, then returns -1.</li>
     *     <li>If this matrix is a 1x1 matrix, then returns 0.</li>
     *     <li>If this matrix is a row vector, then returns 1.</li>
     *     <li>If this matrix is a column vector, then returns 2.</li>
     * </ul>
     */
    default int vectorType() {
        int type;
        int rows = numRows();
        int cols = numCols();

        if(rows==1 || cols==1) {
            if(rows==1 && cols==1) type = 0;
            else if(rows==1) type = 1;
            else type = 2; // Then this matrix is equivalent to a column vector.
        } else {
            type = -1; // Then this matrix is not equivalent to any vector.
        }

        return type;
    }


    /**
     * Checks if this matrix is triangular (i.e., upper triangular, diagonal, lower triangular).
     * @return {@code true} is this matrix is triangular; {@code false} otherwise.
     */
    default boolean isTri() {
        return isTriL() || isTriU();
    }


    /**
     * Checks if this matrix is diagonal.
     * @return {@code true} is this matrix is diagonal; {@code false} otherwise.
     */
    default boolean isDiag() {
        return isTriL() && isTriU();
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return {@code true} is this matrix is upper triangular; {@code false} otherwise.
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    boolean isTriU();


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return {@code true} is this matrix is lower triangular; {@code false} otherwise.
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    boolean isTriL();


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return {@code true} if this matrix is the identity matrix; {@code false} otherwise.
     */
    boolean isI();


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of multiplying this matrix with {@code b}.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If the number of columns in this matrix do not equal the size of
     * {@code b}.
     */
    VectorMixin<?, ?, ?, W> mult(V b);


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If the number of columns in this matrix do not equal the number
     * of rows in matrix {@code b}.
     */
    U mult(T b);


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    U multTranspose(T b);


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix b.
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    default W fib(T b) {
        ValidateParameters.ensureEqualShape(getShape(), b.getShape());
        return this.H().mult(b).trace();
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(T, int)
     * @see #augment(T)
     */
    T stack(T b);


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     * @return The result of stacking {@code b} to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(T)
     * @see #stack(T, int)
     */
    T augment(T b);


    /**
     * Augments a vector to this matrix.
     * @param b The vector to augment to this matrix.
     * @return The result of augmenting {@code b} to this matrix.
     */
    T augment(V b);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(T)} and {@link #augment(T)}.
     *
     * @param b Matrix to stack to this matrix.
     * @param axis Axis along which to stack:
     *      <ul>
     *          <li>If axis=0, then stacks along rows and is equivalent to {@link #augment(T)}.</li>
     *          <li>If axis=1, then stacks along columns and is equivalent to {@link #stack(T)}.</li>
     *      </ul>
     *
     * @return The result of stacking this matrix and {@code b} along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     * @see #augment(T)
     * @see #stack(T)
     */
    default T stack(T b, int axis){
        if(axis == 0)
            return this.augment(b);
        else if(axis == 1)
            return this.stack(b);
        else
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
    }


    /**
     * Swaps specified rows in the matrix. This is done in-place.
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     * @return A reference to this matrix.
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    T swapRows(int rowIndex1, int rowIndex2);


    /**
     * Swaps specified columns in the matrix. This is done in-place.
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     * @return A reference to this matrix.
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    T swapCols(int colIndex1, int colIndex2);


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     * @return {@code true} if this matrix is symmetric; {@code false} otherwise.
     */
    boolean isSymmetric();


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    boolean isHermitian();


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return {@code true} if this matrix it is orthogonal; {@code false} otherwise.
     */
    boolean isOrthogonal();


    /**
     * Gets the row of this matrix at the specified index.
     * @param rowIdx Index of the row to get.
     * @return The row of this matrix at index {@code rowIdx} as a vector.
     * @throws IllegalArgumentException If {@code rowIdx < 0 || rowIdx >= this.numRows()}.
     */
    default V getRow(int rowIdx) {
        return getRow(rowIdx, 0, numCols());
    }


    /**
     * Gets a range of a row of this matrix.
     * @param rowIdx The index of the row to get.
     * @param start The staring column of the row range to get (inclusive).
     * @param stop The ending column of the row range to get (exclusive).
     * @return A vector containing the elements of the specified row over the range [start, stop).
     * @throws IllegalArgumentException If {@code rowIdx < 0 || rowIdx >= this.numRows()} or {@code start < 0 || start >= numCols} or
     * {@code stop < start || stop > numCols}.
     */
    V getRow(int rowIdx, int start, int stop);


    /**
     * Gets the column of this matrix at the specified index.
     * @param colIdx Index of the column to get.
     * @return The column of this matrix at index {@code colIdx} as a vector.
     * @throws IllegalArgumentException If {@code colIdx < 0 || colIdx >= this.numCols()}.
     */
    default V getCol(int colIdx) {
        return getCol(colIdx, 0, numRows());
    }


    /**
     * Gets a range of a column of this matrix.
     * @param colIdx The index of the column to get.
     * @param start The staring row of the column range to get (inclusive).
     * @param stop The ending row of the column range to get (exclusive).
     * @return A vector containing the elements of the specified column over the range [start, stop).
     * @throws IllegalArgumentException If {@code colIdx < 0 || colIdx >= this.numCols()} or {@code start < 0 || start >= numRows} or
     * {@code stop < start || stop > numRows}.
     */
    V getCol(int colIdx, int start, int stop);


    /**
     * Gets the diagonal elements of this matrix.
     * @return Collects the elements of this matrix along the principle diagonal and returns as a vector. Will have length equal to
     * {@code Math.min(this.numRows(), this.numCols())}.
     */
    default V getDiag() {
        return getDiag(0);
    }


    /**
     * Gets the elements of this matrix along the specified diagonal.
     * @param diagOffset The diagonal to get within this matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     * @return The elements of the specified diagonal as a vector.
     */
    V getDiag(int diagOffset);


    /**
     * Sets a specified row of this matrix to a vector.
     * @param row Vector to replace specified row in this matrix.
     * @param rowIdx Index of the row to set.
     * @return If this matrix is dense, the row set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse a copy will be created with the new row and returned.
     */
    T setRow(V row, int rowIdx);


    /**
     * Sets a specified column of this matrix to a vector.
     * @param col Vector to replace specified column in this matrix.
     * @param colIdx Index of the column to set.
     * @return If this matrix is dense, the column set operation is done in-place and a reference to this matrix is returned.
     * If this matrix is sparse a copy will be created with the new column and returned.
     */
    T setCol(V col, int colIdx);


    /**
     * Removes a specified row from this matrix.
     * @param rowIndex Index of the row to remove from this matrix.
     * @return A copy of this matrix with the specified row removed.
     */
    T removeRow(int rowIndex);


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     * @return A copy of this matrix with the specified column removed.
     */
    T removeRows(int... rowIndices);


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return A copy of this matrix with the specified column removed.
     */
    T removeCol(int colIndex);


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     * @return A copy of this matrix with the specified column removed.
     */
    T removeCols(int... colIndices);


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    T setSliceCopy(T values, int rowStart, int colStart);


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    T getSlice(int rowStart, int rowEnd, int colStart, int colEnd);


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A reference to this matrix.
     */
    T set(W value, int row, int col);


    /**
     * Extracts the upper-triangular portion of this matrix with a specified diagonal offset. All other data of the resulting
     * matrix will be zero.
     * @param diagOffset Diagonal offset for upper-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all data at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all data at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all data at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     * @return The upper-triangular portion of this matrix with a specified diagonal offset. All other data of the returned
     * matrix will be zero.
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    T getTriU(int diagOffset);


    /**
     * Extracts the upper-triangular portion of this matrix. All other data in the resulting matrix will be zero.
     * @return The upper-triangular portion of this matrix. with all other data in the resulting matrix will be zero.
     */
    default T getTriU() {
        return getTriU(0);
    }


    /**
     * Extracts the lower-triangular portion of this matrix with a specified diagonal offset. All other data of the resulting
     * matrix will be zero.
     * @param diagOffset Diagonal offset for lower-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all data at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all data at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all data at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     * @return The lower-triangular portion of this matrix with a specified diagonal offset. All other data of the returned
     * matrix will be zero.
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    T getTriL(int diagOffset);


    /**
     * Extracts the lower-triangular portion of this matrix. All other data in the resulting matrix will be zero.
     * @return The lower-triangular portion of this matrix. with all other data in the resulting matrix will be zero.
     */
    default T getTriL() {
        return getTriL(0);
    }


    /**
     * Creates a deep copy of this matrix.
     * @return A deep copy of this matrix.
     */
    T copy();


    /**
     * Computes the Hermitian transpose of this matrix.
     * @return The Hermitian transpose of this matrix.
     */
    T H();


    /**
     * Computes the transpose of this matrix.
     * @return The transpose of this matrix.
     */
    T T();


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not a row or column vector it will first be flattened then
     * converted to a vector.
     * @return A vector which
     */
    V toVector();


    /**
     * Accepts a visitor that implements the {@link MatrixVisitor} interface.
     * This method is part of the "Visitor Pattern" and allows operations to be performed
     * on the matrix without modifying the matrix's class directly.
     *
     * @param <R> The return type of the visitor's operation.
     * @param visitor The visitor implementing the operation to be performed.
     * @return The result of the visitor's operation, typically another matrix or a scalar value.
     * @throws NullPointerException if the visitor is {@code null}.
     */
    <R> R accept(MatrixVisitor<R> visitor);


    /**
     * Computes the element-wise sum of two matrices.
     * @param b Second matrix in the element-wise sum.
     * @return The element-wise sum of this matrix and {@code b}.
     */
    T add(T b);


    /**
     * Computes the element-wise difference of two matrices.
     * @param b Second matrix in the element-wise difference.
     * @return The element-wise difference of this matrix and {@code b}.
     */
    T sub(T b);


    /**
     * Computes the element-wise product of two matrices.
     * @param b Second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    T elemMult(T b);


    /**
     * Computes the element-wise quotient of two matrices.
     * @param b Second matrix in the element-wise quotient.
     * @return The element-wise quotient of this matrix and {@code b}.
     */
    T div(T b);
}
