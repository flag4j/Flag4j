/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense;

import org.flag4j.arrays.Shape;
import org.flag4j.util.ValidateParameters;

public final class DenseOps {

    private DenseOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Swaps specified rows in the matrix. This is done in-place.
     *
     * @param shape Shape of the matrix.
     * @param data Data of the matrix (modified).
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code shape.getRank() != 2}.
     * @throws IndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    public static <T> void swapRows(Shape shape, T[] data, int rowIdx1, int rowIdx2) {
        ValidateParameters.ensureRank(shape, 2);
        int numRows = shape.get(0);
        int numCols = shape.get(1);
        ValidateParameters.validateArrayIndices(numRows, rowIdx1, rowIdx2);

        swapRowsUnsafe(shape, data, rowIdx1, rowIdx2, 0, numCols);
    }


    /**
     * <p>Swaps two rows, over a specified range of columns, within a matrix. Specifically, all elements in the matrix within rows
     * {@code rowIdx1}
     * and {@code rowIdx2} and between columns {@code start} (inclusive) and {@code stop} (exclusive).
     * This operation is done in-place.
     *
     * <p>No bounds checking is done within this method to ensure that the indices provided are valid. As such, it is
     * <i>highly</i> recommended to us {@link #swapRows(Shape, Object[], int, int)} in most cases.
     *
     * @param shape Shape of the matrix.
     * @param data Data of the matrix (modified).
     * @param rowIdx1 Index of the first row to swap.
     * @param rowIdx2 Index of the second row to swap.
     * @param start Index of the column specifying the start of the range for the row swap (inclusive).
     * @param stop Index of the column specifying the end of the range for the row swap (exclusive).
     */
    public static <T> void swapRowsUnsafe(Shape shape, T[] data, int rowIdx1, int rowIdx2, int start, int stop) {
        // Quick return when indices are equal.
        if(rowIdx1 == rowIdx2) return;

        final int cols = shape.get(1);
        final int rowOffset1 = rowIdx1*cols;
        final int rowOffset2 = rowIdx2*cols;
        T temp;

        for(int j=start; j<stop; j++) {
            temp = data[rowOffset1 + j];
            data[rowOffset1 + j] = data[rowOffset2 + j];
            data[rowOffset2 + j] = temp;
        }
    }


    /**
     * Swaps specified columns in the matrix. This is done in-place.
     *
     * @param shape Shape of the matrix.
     * @param data Data of the matrix (modified).
     * @param rowIndex1 Index of the first column to swap.
     * @param rowIndex2 Index of the second column to swap.
     *
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code shape.getRank() != 2}.
     * @throws IndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    public static <T> void swapCols(Shape shape, T[] data, int colIdx1, int colIdx2) {
        ValidateParameters.ensureRank(shape, 2);
        int numRows = shape.get(0);
        int numCols = shape.get(1);
        ValidateParameters.validateArrayIndices(numCols, colIdx1, colIdx2);

        swapColsUnsafe(shape, data, colIdx1, colIdx2, 0, numRows);
    }


    /**
     * <p>Swaps two columns, over a specified range of rows, within a matrix. Specifically, all elements in the matrix within columns
     * {@code colIdx1} and {@code colIdx2} and between rows {@code start} (inclusive) and {@code stop} (exclusive). This operation
     * is done in-place.
     *
     * <p>No bounds checking is done within this method to ensure that the indices provided are valid. As such, it is
     * <i>highly</i> recommended to us {@link #swapCols(Shape, Object[], int, int)} in most cases.
     *
     * @param shape Shape of the matrix.
     * @param data Data of the matrix (modified).
     * @param colIdx1 Index of the first column to swap.
     * @param colIdx2 Index of the second column to swap.
     * @param start Index of the row specifying the start of the range for the row swap (inclusive).
     * @param stop Index of the row specifying the end of the range for the row swap (exclusive).
     */
    public static <T> void swapColsUnsafe(Shape shape, T[] data, int colIdx1, int colIdx2, int start, int stop) {
        if(colIdx1 == colIdx2) return;

        final int cols = shape.get(1);
        int rowOffset = start*cols;
        T temp;

        for(int i=start; i<stop; i++) {
            temp = data[rowOffset + colIdx1];
            data[rowOffset + colIdx1] = data[rowOffset + colIdx2];
            data[rowOffset + colIdx2] = temp;
            rowOffset += cols;
        }
    }
}
