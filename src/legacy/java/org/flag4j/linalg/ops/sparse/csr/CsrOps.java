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

package org.flag4j.linalg.ops.sparse.csr;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Utility class for computing ops on sparse CSR matrices.
 */
public final class CsrOps {

    private CsrOps() {
        // Hide default constructor for utility class.
    }

    /**
     * Computes transpose of complex CSR matrix.
     * @param src The matrix to transpose.
     * @return The transpose of the {@code src} matrix.
     */
    public static <T> void transpose(T[] srcEntries, int[] srcRowPointers, int[] srcColIndices,
                                     T[] destEntries, int[] destRowPointers, int[] destColIndices) {
        // Count number of items in each row.
        for(int i=0, size=srcColIndices.length; i<size; i++)
            destRowPointers[srcColIndices[i] + 1]++;

        // Accumulate the row counts.
        for(int i=1; i<destRowPointers.length; i++)
            destRowPointers[i] += destRowPointers[i-1];

        int[] tempPos = java.util.Arrays.copyOf(destRowPointers, destRowPointers.length);

        // Fill in the values for the transposed matrix
        for (int row = 0, size=srcRowPointers.length-1; row < size; row++) {
            for (int i = srcRowPointers[row], rowLength=srcRowPointers[row+1]; i < rowLength; i++) {
                int col = srcColIndices[i];
                int pos = tempPos[col];
                destEntries[pos] = srcEntries[i];
                destColIndices[pos] = row;
                tempPos[col]++;
            }
        }
    }


    /**
     * Computes herniation transpose of complex CSR matrix.
     * @param src The matrix to transpose.
     * @return The transpose of the {@code src} matrix.
     */
    public static <T extends FieldElement<T>> void hermTranspose(
            T[] srcEntries, int[] srcRowPointers, int[] srcColIndices,
            T[] destEntries, int[] destRowPointers, int[] destColIndices) {
        // Count number of items in each row.
        for(int i=0, size=srcColIndices.length; i<size; i++)
            destRowPointers[srcColIndices[i] + 1]++;

        // Accumulate the row counts.
        for(int i=1; i<destRowPointers.length; i++)
            destRowPointers[i] += destRowPointers[i-1];

        int[] tempPos = java.util.Arrays.copyOf(destRowPointers, destRowPointers.length);

        // Fill in the values for the transposed matrix
        for (int row = 0, size=srcRowPointers.length-1; row < size; row++) {
            for (int i = srcRowPointers[row], rowLength=srcRowPointers[row+1]; i < rowLength; i++) {
                int col = srcColIndices[i];
                int pos = tempPos[col];
                destEntries[pos] = srcEntries[i].conj();
                destColIndices[pos] = row;
                tempPos[col]++;
            }
        }
    }


    /**
     * <p>Applies an element-wise binary operation to two csr matrices.
     *
     * <p>Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to dene matrices and using a dense matrix addition algorithm.
     *
     * @param shape1 Shape of the first CSR matrix.
     * @param src1Entries Non-zero items of the first CSR matrix.
     * @param src1RowPointers Non-zero row pointers of the first CSR matrix.
     * @param src1ColIndices Non-zero column indices of the first CSR matrix.
     * @param shape2 Shape of the second CSR matrix.
     * @param src2Entries Non-zero items of the second CSR matrix.
     * @param src2RowPointers Non-zero row pointers of the second CSR matrix.
     * @param src2ColIndices Non-zero column indices of the second CSR matrix.
     * @param opp Binary operator to apply element-wise to {@code src1} and {@code src2}.
     * @param uOpp Unary operator for use with binary ops which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return The result of applying the specified binary operation to {@code src1} and {@code src2}
     * element-wise.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static <T> SparseMatrixData<T> applyBinOpp(Shape shape1, T[] src1Entries, int[] src1RowPointers, int[] src1ColIndices,
                                                      Shape shape2, T[] src2Entries, int[] src2RowPointers, int[] src2ColIndices,
                                                      BinaryOperator<T> opp, UnaryOperator<T> uOpp) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        final int rows1 = shape1.getSize(0);

        List<T> dest = new ArrayList<>();
        int[] rowPointers = new int[src1RowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows1; i++) {
            int rowPtr1 = src1RowPointers[i];
            int rowPtr2 = src2RowPointers[i];

            while(rowPtr1 < src1RowPointers[i+1] && rowPtr2 < src2RowPointers[i+1]) {
                int col1 = src1ColIndices[rowPtr1];
                int col2 = src2ColIndices[rowPtr2];

                if(col1 == col2) {
                    if(uOpp != null) dest.add(opp.apply(src1Entries[rowPtr1], uOpp.apply(src2Entries[rowPtr2])));
                    else dest.add(opp.apply(src1Entries[rowPtr1], src2Entries[rowPtr2]));

                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                } else if(col1 < col2) {
                    dest.add(src1Entries[rowPtr1]);
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    if(uOpp!=null) dest.add(uOpp.apply(src2Entries[rowPtr2]));
                    else dest.add(src2Entries[rowPtr2]);
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1RowPointers[i+1]) {
                dest.add(src1Entries[rowPtr1]);
                colIndices.add(src1ColIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2RowPointers[i+1]) {
                if(uOpp!=null) dest.add(uOpp.apply(src2Entries[rowPtr2]));
                else dest.add(src2Entries[rowPtr2]);
                colIndices.add(src2ColIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++)
            rowPointers[i] += rowPointers[i-1];

        return new SparseMatrixData<T>(shape1, dest, ArrayConversions.toArrayList(rowPointers), colIndices);
    }


    /**
     * Copies and inserts a new value into a sparse CSR matrix. This method assumes that a non-zero value is not already present at
     * the specified indices.
     * @param srcEntries Non-zero items of the original CSR matrix.
     * @param srcRowPointers Non-zero row pointers of the original CSR matrix.
     * @param srcColIndices Non-zero column indices of the original CSR matrix.
     * @param destEntries Array to store the new items in.
     * @param destRowPointers Array to store the new row pointers in.
     * @param destColIndices Array to store the new column indices in.
     * @param row Row index of the point to insert.
     * @param col Column index of the point to insert.
     * @param insertionPoint Index within {@code srcEntries} that the new value should be inserted at.
     * @param value New value to insert into the CSR matrix.
     */
    public static <T> void insertNewValue(T[] srcEntries, int[] srcRowPointers, int[] srcColIndices,
                                          T[] destEntries, int[] destRowPointers, int[] destColIndices,
                                          int row, int col, int insertionPoint, T value) {
        // Copy old items and insert new one.
        System.arraycopy(srcEntries, 0, destEntries, 0, insertionPoint);
        destEntries[insertionPoint] = value;
        System.arraycopy(srcEntries, insertionPoint, destEntries, insertionPoint+1,
                srcEntries.length-insertionPoint);

        // Copy old column indices and insert new one.
        System.arraycopy(srcColIndices, 0, destColIndices, 0, insertionPoint);
        destColIndices[insertionPoint] = col;
        System.arraycopy(srcColIndices, insertionPoint, destColIndices, insertionPoint+1,
                srcEntries.length-insertionPoint);

        // Increment row pointers.
        for(int i=row+1; i<srcRowPointers.length; i++)
            destRowPointers[i]++;
    }


    /**
     * Swaps two rows in a sparse CSR matrix. This is done in-place.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param rowIdx1 Index of the first row to swap.
     * @param rowIdx2 Index of the second row to swap.
     * @throws IndexOutOfBoundsException If either {@code rowIdx1} or {@code rowIdx2} is out of bounds of the rows of this matrix.
     */
    public static <T> void swapRows(T[] entries, int[] rowPointers, int[] colIndices,
                                    int rowIdx1, int rowIdx2) {
        if(rowIdx1 == rowIdx2) return;
        else if(rowIdx1 > rowIdx2) {
            // ensure the second index is larger than the first.
            int temp = rowIdx1;
            rowIdx1 = rowIdx2;
            rowIdx2 = temp;
        }

        // Get range for values in the given rows.
        int start1 = rowPointers[rowIdx1];
        int end1 = rowPointers[rowIdx1+1];
        int nnz1 = end1-start1;  // Number of non-zero items in the first row to swap.

        int start2 = rowPointers[rowIdx2];
        int end2 = rowPointers[rowIdx2+1];
        int nnz2 = end2-start2;  // Number of non-zero items in the second row to swap.

        int destPos = 0;

        T[] updatedEntries = (T[]) new FieldElement[end2-start1];
        int[] updatedColIndices = new int[end2-start1];

        // Copy items from second row in swap.
        System.arraycopy(entries, start2, updatedEntries, destPos, nnz2);
        System.arraycopy(colIndices, start2, updatedColIndices, destPos, nnz2);

        // Copy items between rows.
        destPos += nnz2;
        System.arraycopy(entries, end1, updatedEntries, destPos, start2-end1);
        System.arraycopy(colIndices, end1, updatedColIndices, destPos, start2-end1);

        // Copy items from first row in swap.
        destPos += (start2-end1);
        System.arraycopy(entries, start1, updatedEntries, destPos, nnz1);
        System.arraycopy(colIndices, start1, updatedColIndices, destPos, nnz1);

        // Update the row pointers.
        int diff = nnz2-nnz1;  // Difference in number of items.
        for(int i=rowIdx1+1; i<=rowIdx2; i++)
            rowPointers[i] += diff;

        // Copy updated ndarrays to this tensor's storage.
        System.arraycopy(updatedEntries, 0, entries, start1, updatedEntries.length);
        System.arraycopy(updatedColIndices, 0, colIndices, start1, updatedEntries.length);
    }


    /**
     * Swaps two columns in a sparse CSR matrix. This is done in-place.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param colIdx1 Index of the first column to swap.
     * @param colIdx2 Index of the second column to swap.
     * @throws IndexOutOfBoundsException If either {@code colIndex1} or {@code colIndex2} is out of bounds of the columns of this
     * matrix.
     */
    public static <T> void swapCols(T[] entries, int[] rowPointers, int[] colIndices,
                                    int colIdx1, int colIdx2) {
        if(colIdx1 == colIdx2) return;

        int numRows = rowPointers.length - 1;

        // Ensure colIndex1 < colIndex2 for simplicity
        if(colIdx1 > colIdx2) {
            int temp = colIdx1;
            colIdx1 = colIdx2;
            colIdx2 = temp;
        }

        // Traverse each row to find and swap the columns.
        for(int i=0; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int rowEnd = rowPointers[i+1];

            int pos1 = Arrays.binarySearch(colIndices, rowStart, rowEnd, colIdx1);
            int pos2 = Arrays.binarySearch(colIndices, rowStart, rowEnd, colIdx2);

            if(pos1 >= 0 && pos2 >= 0) {
                // Both columns contain a non-zero value in this row.
                ArrayUtils.swap(entries, pos1, pos2);
            } else if(pos1 >= 0) {
                // Only first column contains non-zero value.
                int newPos = -pos2-1;
                if(newPos==rowEnd) newPos--;
                moveAndShiftLeft(entries, rowPointers, colIndices, colIdx2, pos1, newPos);
            } else if(pos2 >= 0) {
                // Only second column contains non-zero value.
                int newPos = -pos1-1;
                if(newPos==rowEnd) newPos--;
                moveAndShiftRight(entries, rowPointers, colIndices, colIdx1, pos2, newPos);
            }
        }
    }


    /**
     * Moves a non-zero value in a row of a CSR matrix to a new column to the left of its current column. The new column is assumed
     * to contain a zero. To accommodate this move, all items between the columns are shifted right within the non-zero items array.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param newColIdx New column for the value to be moved to within the row.
     * @param currPos Current index of the value within the non-zero items of {@code src} (assumed to be in the same row as {@code
     * newPos}).
     * @param newPos New index for the value to be moved to within the non-zero items of {@code src} (assumed to be in the same
     * row as {@code currPos}).
     */
    private static <T> void moveAndShiftRight(T[] entries, int[] rowPointers, int[] colIndices,
                                              int newColIdx, int currPos, int newPos) {
        T value = entries[currPos];  // Extract the non-zero value.

        // Shift items in row to right.
        for(int j=currPos; j>newPos; j--) {
            entries[j] = entries[j - 1];
            colIndices[j] = colIndices[j - 1];
        }

        entries[newPos] = value;  // Move non-zero value to new location.
        colIndices[newPos] = newColIdx;  // Update column index for the value.
    }


    /**
     * Moves a non-zero value in a row of a CSR matrix to a new column to the right of its current column. The new column is assumed
     * to contain a zero. To accommodate this move, all items between the columns are shifted left within the non-zero items array.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param newColIdx New column for the value to be moved to within the row.
     * @param currPos Current index of the value within the non-zero items of {@code src} (assumed to be in the same row as {@code
     * newPos}).
     * @param newPos New index for the value to be moved to within the non-zero items of {@code src} (assumed to be in the same
     * row as {@code currPos}).
     */
    private static <T> void moveAndShiftLeft(T[] entries, int[] rowPointers, int[] colIndices,
                                             int newColIdx, int currPos, int newPos) {
        T value = entries[currPos];  // Extract the non-zero value.

        // Shift items in row to left.
        for(int j=currPos; j<newPos; j++) {
            entries[j] = entries[j+1];
            colIndices[j] = colIndices[j+1];
        }

        entries[newPos] = value;  // Move non-zero value to new location.
        colIndices[newPos] = newColIdx;  // Update column index for the value.
    }


    /**
     * Gets a specified slice of a CSR matrix.
     *
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd   Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd   Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart}
     * or if {@code colEnd} is not greater than {@code colStart}.
     */
    public static <T> SparseMatrixData<T> getSlice(Shape shape, T[] entries, int[] rowPointers, int[] colIndices,
                                                   int rowStart, int rowEnd,
                                                   int colStart, int colEnd) {
        SparseUtils.validateSlice(shape, rowStart, rowEnd, colStart, colEnd);
        List<T> slice = new ArrayList<>();
        List<Integer> sliceRowIndices = new ArrayList<>();
        List<Integer> sliceColIndices = new ArrayList<>();

        // Efficiently construct COO matrix then convert to a CSR matrix.
        int rowStop = rowEnd-1;
        for(int i=rowStart; i<rowStop; i++) {
            // Beginning and ending indices for the row.
            int begin = rowPointers[i];
            int end = rowPointers[i+1];

            for(int j=begin; j<end; j++) {
                int col = colIndices[j];

                // Add value if it is within the slice.
                if(col >= colStart && col < colEnd) {
                    slice.add(entries[j]);
                    sliceRowIndices.add(i-rowStart);
                    sliceColIndices.add(col-colStart);
                }
            }
        }

        // Matrix has been constructed as COO matrix. Now must be converted to CSR matrix.
        int size = rowEnd-rowStart + 1;
        List<Integer> sliceRowPointers = new ArrayList<>(size);
        for(int i=0; i<size; i++)
            sliceRowPointers.add(0);

        // Copy row indices.
        for(int i=0, stop=sliceRowIndices.size(); i<stop; i++) {
            Integer curr = sliceRowPointers.get(sliceRowIndices.get(i) + 1);
            sliceRowPointers.set(sliceRowIndices.get(i) + 1, ++curr);
        }

        Integer prev = sliceRowPointers.get(0);
        //Accumulate row pointers.
        for(int i=1; i<size; i++) {
            Integer curr = sliceRowPointers.get(i);
            sliceRowPointers.set(i, prev + curr);
            prev += curr;
        }

        return new SparseMatrixData<T>(new Shape(rowEnd-rowStart, colEnd-colStart),
                slice, sliceRowPointers, sliceColIndices);
    }
}
