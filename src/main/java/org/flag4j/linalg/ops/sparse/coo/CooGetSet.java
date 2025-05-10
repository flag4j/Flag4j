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

package org.flag4j.linalg.ops.sparse.coo;

import org.flag4j.arrays.IntPair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.SparseVectorData;
import org.flag4j.linalg.ops.sparse.SparseElementSearch;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>A utility class that aids in getting or setting specified elements of a sparse COO tensor, matrix, or vector.
 * <p>All methods in this class guarantee all results will be properly lexicographically sorted by indices.
 */
public final class CooGetSet {

    private CooGetSet() {
        // Hide default constructor for utility class.
    }


    /**
     * Sets a specified row of a real sparse COO matrix to the values in a sparse COO vector.
     * @param srcShape Shape of the matrix to set row in.
     * @param srcEntries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param rowIdx Index of the row to set.
     * @param size Full size of the COO vector.
     * @param row Non-zero data of the COO vector containing new row values.
     * @param indices Non-zero indices of the COO vector containing new row values.
     * @return Sparse matrix data containing the data for the COO matrix resulting from setting the specified row of the provided
     * COO matrix to the provided COO vector.
     * @throws IllegalArgumentException If {@code
     * srcShape.get(1) != size}.
     */
    public static <T> SparseMatrixData<T> setRow(Shape srcShape, T[] srcEntries, int[] rowIndices, int[] colIndices,
                                                 int rowIdx,
                                                 int size, T[] row, int[] indices) {
        if(srcShape.getSize(1) != size) {
            throw new IllegalArgumentException("Cannot set row of matrix with shape " + srcShape
                    + " with a vector of size " + size + ".");
        }

        int[] rowArray = new int[row.length];
        Arrays.fill(rowArray, rowIdx);

        List<T> entries = new ArrayList<>(Arrays.asList(row));
        List<Integer> destRowIndices = ArrayConversions.toArrayList(rowArray);
        List<Integer> destColIndices = ArrayConversions.toArrayList(indices);

        for(int i=0, nnz=srcEntries.length; i<nnz; i++) {
            int srcRow = rowIndices[i];

            if(srcRow != rowIdx) {
                entries.add(srcEntries[i]);
                destRowIndices.add(srcRow);
                destColIndices.add(colIndices[i]);
            }
        }

        // Ensure the data is properly sorted.
        new CooDataSorter<>(entries, destRowIndices, destColIndices).sparseSort();

        return new SparseMatrixData<>(srcShape, entries, destRowIndices, destColIndices);
    }


    /**
     * Sets a column of a sparse matrix to the values in a sparse tensor.
     * @param srcShape Shape of the matrix to set column in.
     * @param srcEntries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param colIdx Index of the column to set.
     * @param size Full size of the COO vector.
     * @param col Non-zero data of the COO vector containing new column values.
     * @param indices Non-zero indices of the COO vector containing new column values.
     * @return A copy of the {@code src} matrix with the specified column set to the {@code col} sparse vector.
     * @throws IllegalArgumentException If the {@code src} matrix does not have the same number of rows as total data
     * in the {@code col} vector.
     */
    public static <T> SparseMatrixData<T> setCol(Shape srcShape, T[] srcEntries, int[] rowIndices, int[] colIndices,
                                                 int colIdx,
                                                 int size, T[] col, int[] indices) {
        ValidateParameters.validateArrayIndices(srcShape.getSize(1), colIdx);
        ValidateParameters.ensureAllEqual(srcShape.getSize(0), size);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<T> destEntries = new ArrayList<>(Arrays.asList(col));
        List<Integer> destRowIndices = ArrayConversions.toArrayList(indices);
        List<Integer> destColIndices = ArrayConversions.toArrayList(ArrayBuilder.filledArray(col.length, colIdx));

        addNotInCol(destEntries, destRowIndices, destColIndices,
                srcEntries, rowIndices, colIndices, colIdx);

        return new SparseMatrixData<T>(srcShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Adds values from a sparse matrix to specified lists if the value is not within a specified column.
     * @param destEntries List to add non-zero data from sparse matrix to.
     * @param destRowIndices List to add non-zero row indices from sparse matrix to.
     * @param destColIndices List to add non-zero column indices from sparse matrix to.
     * @param src The sparse matrix to get non-zero values and indices from.
     * @param colIdx Specified column to not add data to the lists from.
     */
    private static <T> void addNotInCol(List<T> destEntries, List<Integer> destRowIndices,
                                        List<Integer> destColIndices,
                                        T[] srcEntries, int[] rowIndices, int[] colIndices,
                                        int colIdx) {
        for(int i=0, size=srcEntries.length; i<size; i++) {
            // Add all data which are not in the specified column.
            if(colIndices[i]!=colIdx) {
                destEntries.add(srcEntries[i]);
                destRowIndices.add(rowIndices[i]);
                destColIndices.add(colIndices[i]);
            }
        }
    }


    /**
     * Gets the specified element from a sparse COO vector.
     * @param entries Non-zero values of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     * @param index Index of the value to get from the vector.
     * @return The value in the sparse COO vector at the specified index if it exists. If the value is not found
     * within the non-zero data, {@code null} will be returned.
     */
    public static <V> V getCoo(V[] entries, int[] indices, int index) {
        int idx = Arrays.binarySearch(indices, index);
        return idx<0 ? null : entries[idx];
    }


    /**
     * Gets the specified element from a sparse COO matrix.
     * @param entries Non-zero values of the sparse COO matrix from which to get element.
     * @param rowIndices Non-zero row indices for the sparse COO matrix.
     * @param colIndices Non-zero column indices for the sparse COO matrix.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return The value in the sparse COO matrix at the specified row and column indices if it exists. If the value is not found
     * within the non-zero data, {@code null} will be returned.
     */
    public static <V> V getCoo(V[] entries, int[] rowIndices, int[] colIndices, int row, int col) {
        int idx = SparseElementSearch.matrixBinarySearch(rowIndices, colIndices, row, col);
        return idx<0 ? null : entries[idx];
    }


    /**
     * Gets an element of a sparse COO tensor at the specified {@cide target} index. If no non-zero value exists, then {@code null}
     * is returned.
     *
     * @param entries Non-zero data of the COO tensor.
     * @param indices Non-zero indices o the COO tensor.
     * @param target Target index to search for in {@code indices}.
     * @return The value in {@code data} which has an index matching the target. That is, if some {@code idx} is found such that
     * {@code Arrays.equals(indices[idx], target)}, then {@code data[idx]} is returned. If no such {@code idx} id found, then
     * {@code null} is returned.
     */
    public static <T> T getCoo(T[] entries, int[][] indices, int[] target) {
        int idx = SparseElementSearch.binarySearchCoo(indices, target);
        return (idx >= 0) ? entries[idx] : null;
    }


    /**
     * Inserts a new value into a sparse COO tensor. This assumes there is no non-zero value already at the specified index.
     * @param value Value to insert into the tensor.
     * @param index Non-zero index for new value.
     * @param srcEntries Non-zero data of the source tensor. Unmodified.
     * @param srcIndices Non-zero indices of the source tensor. Assumed to be rectangular. Unmodified.
     * @param insertionPoint Index in {@code srcEntries} and {@code srcIndices} to insert {@code value} and {@code index}.
     * @param destEntries Destination for storing the result of inserting the {@code value} into {@code srcEntries}.
     * @param destIndices Destination for storing the result of inserting the {@code index} into {@code srcIndices}.
     * @throws IllegalArgumentException If {@code destEntries.length != srcEntries.length + 1} or
     * {@code destIndices.length != srcIndices.length + 1}
     * @throws IllegalArgumentException If {@code index.length != srcIndices[0].length}.
     */
    public static <T> void cooInsertNewValue(T value, int[] index, T[] srcEntries, int[][] srcIndices, int insertionPoint,
                                             T[] destEntries, int[][] destIndices) {
        if(destEntries.length != srcEntries.length + 1 || destIndices.length != srcIndices.length + 1) {
            throw new IllegalArgumentException("Destination arrays must have length one larger than source arrays.");
        }
        if(index.length != srcIndices[0].length) {
            throw new IllegalArgumentException("Index does not have the proper number of dimensions for the tensor. " +
                    "Expecting size " + srcIndices[0].length + " but got " + index.length + ".");
        }

        System.arraycopy(srcEntries, 0, destEntries, 0, insertionPoint);
        destEntries[insertionPoint] = value;
        System.arraycopy(srcEntries, insertionPoint, destEntries, insertionPoint+1, srcEntries.length-insertionPoint);

        for(int i=0; i<insertionPoint; i++)
            destIndices[i] = srcIndices[i].clone();

        destIndices[insertionPoint] = index;

        for(int i=insertionPoint, nnz= srcEntries.length; i<nnz; i++)
            destIndices[i+1] = srcIndices[i].clone();
    }


    /**
     * Inserts a new value into a sparse COO matrix. This assumes there is no non-zero value at the specified row and column.
     * @param value Value to insert into the matrix.
     * @param rowIdx index for the value to insert.
     * @param srcEntries Non-zero data of the source matrix. Unmodified.
     * @param srcRowIndices Non-zero row indices of the source matrix. Unmodified.
     * @param insertionPoint Index in {@code srcEntries}, {@code srcRowIndices}, {@code srcColIndices} to insert {@code value},
     * {@code rowIdx}, and {@code colIdx}
     * @param destEntries Destination for storing the result of inserting the {@code value} into {@code srcEntries}.
     * @param destRowIndices Destination for storing the result of inserting the {@code rowIdx} into {@code srcRowIndices}.
     * @param destColIndices Destination for storing the result of inserting the {@code colIdx} into {@code srcColIndices}.
     * @throws IllegalArgumentException If {@code destEntries.length != srcEntries.length + 1} or
     * {@code destIndices.length != srcRowIndices.length + 1} or {@code destColIndices.length != srcColIndices.length + 1}.
     */
    public static <T> void cooInsertNewValue(T value, int rowIdx, int colIdx,
                                             T[] srcEntries, int[] srcRowIndices, int[] srcColIndices,
                                             int insertionPoint,
                                             T[] destEntries, int[] destRowIndices, int[] destColIndices) {
        if(destEntries.length != srcEntries.length + 1
                || destRowIndices.length != srcRowIndices.length + 1
                || destColIndices.length != srcColIndices.length + 1) {
            throw new IllegalArgumentException("Destination arrays must have length one larger than source arrays.");
        }

        System.arraycopy(srcEntries, 0, destEntries, 0, insertionPoint);
        destEntries[insertionPoint] = value;
        System.arraycopy(srcEntries, insertionPoint, destEntries, insertionPoint+1, srcEntries.length - insertionPoint);

        System.arraycopy(srcRowIndices, 0, destRowIndices, 0, insertionPoint);
        destRowIndices[insertionPoint] = rowIdx;
        System.arraycopy(srcRowIndices, insertionPoint, destRowIndices,
                insertionPoint + 1, srcRowIndices.length - insertionPoint);

        System.arraycopy(srcColIndices, 0, destColIndices, 0, insertionPoint);
        destColIndices[insertionPoint] = colIdx;
        System.arraycopy(srcColIndices, insertionPoint, destColIndices,
                insertionPoint+1, srcColIndices.length - insertionPoint);
    }


    /**
     * Inserts a new value into a sparse COO vector. This assumes there is no non-zero value at the specified index.
     * @param value Value to insert into the vector.
     * @param index Index for the value to insert.
     * @param srcEntries Non-zero data of the source vector. Unmodified.
     * @param srcIndices Non-zero indices of the source vector. Unmodified.
     * @param insertionPoint Index in {@code srcEntries} and {@code srcIndices} to insert {@code value} and {@code index}.
     * @param destEntries Destination for storing the result of inserting the {@code value} into {@code srcEntries}.
     * @param destIndices Destination for storing the result of inserting the {@code index} into {@code srcIndices}.
     * @throws IllegalArgumentException If {@code destEntries.length != srcEntries.length + 1} or
     * {@code destIndices.length != srcIndices.length + 1}.
     */
    public static <T> void cooInsertNewValue(T value, int index,
                                             T[] srcEntries, int[] srcIndices,
                                             int insertionPoint,
                                             T[] destEntries, int[] destIndices) {
        if(destEntries.length != srcEntries.length + 1 || destIndices.length != srcIndices.length + 1) {
            throw new IllegalArgumentException("Destination arrays must have length one larger than source arrays.");
        }

        // Copy the elements up to the insertion point
        System.arraycopy(srcEntries, 0, destEntries, 0, insertionPoint);
        destEntries[insertionPoint] = value;
        System.arraycopy(srcEntries, insertionPoint, destEntries, insertionPoint + 1, srcEntries.length - insertionPoint);

        // Insert the new index
        System.arraycopy(srcIndices, 0, destIndices, 0, insertionPoint);
        destIndices[insertionPoint] = index;
        System.arraycopy(srcIndices, insertionPoint, destIndices, insertionPoint + 1, srcIndices.length - insertionPoint);
    }


    /**
     * Gets an element of a sparse COO tensor at the specified {@cide target} index. If no non-zero value exists, then {@code null}
     * is returned.
     *
     * @param entries Non-zero data of the COO tensor.
     * @param indices Non-zero indices o the COO tensor.
     * @param target Target index to search for in {@code indices}.
     * @return The value in {@code data} which has an index matching the target. That is, if some {@code idx} is found such that
     * {@code Arrays.equals(indices[idx], target)}, then {@code data[idx]} is returned. If no such {@code idx} id found, then
     * {@code null} is returned.
     */
    public static double getCoo(double[] entries, int[][] indices, int[] target) {
        int idx = SparseElementSearch.binarySearchCoo(indices, target);
        return (idx >= 0) ? entries[idx] : null;
    }


    /**
     * Gets an element of a sparse COO tensor at the specified {@cide target} index. If no non-zero value exists, then {@code null}
     * is returned.
     *
     * @param entries Non-zero data of the COO tensor.
     * @param indices Non-zero indices o the COO tensor.
     * @param target Target index to search for in {@code indices}.
     * @return The value in {@code data} which has an index matching the target. That is, if some {@code idx} is found such that
     * {@code Arrays.equals(indices[idx], target)}, then {@code data[idx]} is returned. If no such {@code idx} id found, then
     * {@code null} is returned.
     */
    public static float getCoo(float[] entries, int[][] indices, int[] target) {
        int idx = SparseElementSearch.binarySearchCoo(indices, target);
        return (idx >= 0) ? entries[idx] : null;
    }


    /**
     * Gets an element of a sparse COO tensor at the specified {@cide target} index. If no non-zero value exists, then {@code null}
     * is returned.
     *
     * @param entries Non-zero data of the COO tensor.
     * @param indices Non-zero indices o the COO tensor.
     * @param target Target index to search for in {@code indices}.
     * @return The value in {@code data} which has an index matching the target. That is, if some {@code idx} is found such that
     * {@code Arrays.equals(indices[idx], target)}, then {@code data[idx]} is returned. If no such {@code idx} id found, then
     * {@code null} is returned.
     */
    public static int getCoo(int[] entries, int[][] indices, int[] target) {
        int idx = SparseElementSearch.binarySearchCoo(indices, target);
        return (idx >= 0) ? entries[idx] : null;
    }


    /**
     * Gets the upper-triangular portion of a sparse COO matrix with a possible diagonal offset. The remaining values will be zero.
     *
     * @param diagOffset Diagonal offset indicating which diagonal to extract values at or above.
     * <ul>
     *     <li>If {@code diagOffset == 0} then the properly upper-triangular portion of the matrix is extracted.</li>
     *     <li>If {@code diagOffset == k} where {@code k > 0} then the values at and above the k<sup>th</sup> super-diagonal.</li>
     *     <li>If {@code diagOffset == k} where {@code k < 0} then the values at and above the k<sup>th</sup> sub-diagonal.</li>
     * </ul>
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the COO matrix.
     * @param colIndices Column indices of the COO matrix.
     * @return A data container containing the resulting upper-triangular non-zero data, row indices, and column indices.
     */
    public static <T> SparseMatrixData<T> getTriU(int diagOffset, Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        final int nnz = entries.length;
        int sizeEst = nnz / 2; // Estimate the number of non-zero data.
        List<T> triuEntries = new ArrayList<>(sizeEst);
        List<Integer> triuRowIndices = new ArrayList<>(sizeEst);
        List<Integer> triuColIndices = new ArrayList<>(sizeEst);

        for(int i=0; i<nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            if(col - row >= diagOffset) {
                triuEntries.add(entries[i]);
                triuRowIndices.add(row);
                triuColIndices.add(col);
            }
        }

        return new SparseMatrixData<T>(shape, triuEntries, triuRowIndices, triuColIndices);
    }


    /**
     * Gets the lower-triangular portion of a sparse COO matrix with a possible diagonal offset. The remaining values will be zero.
     *
     * @param diagOffset Diagonal offset indicating which diagonal to extract values at or below.
     * <ul>
     *     <li>If {@code diagOffset == 0} then the properly lower-triangular portion of the matrix is extracted.</li>
     *     <li>If {@code diagOffset == k} where {@code k > 0} then the values at and below the k<sup>th</sup> super-diagonal.</li>
     *     <li>If {@code diagOffset == k} where {@code k < 0} then the values at and below the k<sup>th</sup> sub-diagonal.</li>
     * </ul>
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the COO matrix.
     * @param colIndices Column indices of the COO matrix.
     * @return A data container containing the resulting lower-triangular non-zero data, row indices, and column indices.
     */
    public static <T> SparseMatrixData<T> getTriL(int diagOffset, Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        final int nnz = entries.length;
        int sizeEst = nnz / 2; // Estimate the number of non-zero data.
        List<T> trilEntries = new ArrayList<>(sizeEst);
        List<Integer> trilRowIndices = new ArrayList<>(sizeEst);
        List<Integer> trilColIndices = new ArrayList<>(sizeEst);

        for (int i = 0; i < nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            if (row - col >= diagOffset) {
                trilEntries.add(entries[i]);
                trilRowIndices.add(row);
                trilColIndices.add(col);
            }
        }

        return new SparseMatrixData<>(shape, trilEntries, trilRowIndices, trilColIndices);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse COO matrix to the data of another sparse COO matrix.
     *
     * @param shape1 Shape of the first matrix.
     * @param src1Entries Non-zero data of the matrix to set slice within.
     * @param src1RowIndices Row indices of the matrix to set slice within.
     * @param src1ColIndices Column indices of the matrix to set slice within.
     * @param shape2 Shape of the first matrix.
     * @param src2Entries Non-zero data of the matrix to copy into the specified slice.
     * @param src2RowIndices Row indices of the matrix to copy into the specified slice.
     * @param src2ColIndices Column indices of the matrix to copy into the specified slice.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A sparse data container containing the result of setting the slice in the source matrix.
     * @throws IndexOutOfBoundsException If the {@code values} matrix does not fit in the {@code src}
     * matrix given the row and column index.
     */
    public static <T> SparseMatrixData<T> setSlice(
            Shape shape1, T[] src1Entries, int[] src1RowIndices, int[] src1ColIndices,
            Shape shape2, T[] src2Entries, int[] src2RowIndices, int[] src2ColIndices,
            int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(shape1, shape2, row, col);

        // Initialize lists to new values for the specified slice.
        List<T> entries = new ArrayList<>(Arrays.asList(src2Entries));
        List<Integer> rowIndices = ArrayConversions.toArrayList(ArrayUtils.shift(row, src2RowIndices));
        List<Integer> colIndices = ArrayConversions.toArrayList(ArrayUtils.shift(col, src2ColIndices));

        int[] rowRange = ArrayBuilder.intRange(row, shape2.getSize(0) + row);
        int[] colRange = ArrayBuilder.intRange(col, shape2.getSize(1) + col);
        copyValuesNotInSlice(src1Entries, src1RowIndices, src1ColIndices, entries, rowIndices, colIndices, rowRange, colRange);

        // Ensure the data is sorted properly.
        new CooDataSorter(entries, rowIndices, colIndices).sparseSort();

        return new SparseMatrixData<T>(shape1, entries, rowIndices, colIndices);
    }


    /**
     * Extracts a specified slice from a sparse COO matrix.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the COO matrix.
     * @param colIndices Column indices of the COO matrix.
     * @param rowStart Starting row index of the slice (inclusive).
     * @param rowEnd Ending row index of the slice (exclusive).
     * @param colStart Staring column index of a slice (inclusive).
     * @param colEnd Ending column index of the slice (exclusive).
     * @return A sparse data container containing the specified slice extracted from the COO matrix.
     * @throws IndexOutOfBoundsException If the specified slice does not fit into the matrix.
     */
    public static <T> SparseMatrixData<T> getSlice(Shape shape, T[] entries, int[] rowIndices, int[] colIndices,
                                                   int rowStart, int rowEnd, int colStart, int colEnd) {
        ValidateParameters.validateArrayIndices(shape.getSize(0), rowStart, rowEnd-1);
        ValidateParameters.validateArrayIndices(shape.getSize(1), colStart, colEnd-1);

        List<T> destEntries = new ArrayList<>();
        List<Integer> destRowIndices = new ArrayList<>();
        List<Integer> destColIndices = new ArrayList<>();

        int start = SparseElementSearch.matrixBinarySearch(rowIndices, colIndices, rowStart, colStart);

        if(start < 0) {
            // If no item with the specified indices is found, then begin search at the insertion point.
            start = -start - 1;
        }

        for(int i=start; i<entries.length; i++) {
            if(inSlice(rowIndices[i], colIndices[i], rowStart, rowEnd, colStart, colEnd)) {
                destEntries.add(entries[i]);
                destRowIndices.add(rowIndices[i]-rowStart);
                destColIndices.add(colIndices[i]-colStart);
            }
        }

        return new SparseMatrixData<T>(new Shape(rowEnd-rowStart, colEnd-colStart),
                destEntries, destRowIndices, destColIndices);
    }


    /**
     * Gets the elements of a COO matrix along the specified diagonal.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param diagOffset The diagonal to get within the COO matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     *
     * @return A sparse vector data object containing the non-zero data and indices along the specified diagonal of the COO matrix.
     */
    public static <T> SparseVectorData<T> getDiag(Shape shape, T[] entries, int[] rowIndices, int[] colIndices, int diagOffset) {
        int numRows = shape.getSize(0);
        int numCols = shape.getSize(1);

        ValidateParameters.ensureInRange(diagOffset, -(numRows-1),
                numCols-1, "diagOffset");

        // Validate diagOffset is within the valid range.
        if (diagOffset < -(numRows - 1) || diagOffset > numCols - 1)
            throw new IllegalArgumentException("diagOffset out of range");


        // Compute the length of the diagonal.
        int newSize = Math.min(numRows, numCols);

        if(diagOffset > 0)
            newSize = Math.min(newSize, numCols - diagOffset);
        else if(diagOffset < 0)
            newSize = Math.min(newSize, numRows + diagOffset);

        // Determine the starting row index based on diagOffset.
        int startRow = diagOffset >= 0 ? 0 : -diagOffset;

        List<Integer> idxList = new ArrayList<>();
        List<T> entriesList = new ArrayList<>();

        // Iterate over non-zero data in the COO matrix.
        for (int i = 0, nnz=entries.length; i < nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            // Check if the current element is on the specified diagonal.
            if (col - row == diagOffset) {
                int pos = row - startRow;
                idxList.add(pos);
                entriesList.add(entries[i]);
            }
        }

        return new SparseVectorData<T>(new Shape(newSize), entriesList, idxList);
    }


    /**
     * Gets a specified row of a COO matrix between {@code start} (inclusive) and {@code end} (exclusive).
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param rowIdx Index of the row in this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The row at index {@code rowIdx} of this matrix between the {@code start} and {@code end}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code end} are {@code start} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code end} is less than {@code start}.
     */
    public static <T> SparseVectorData<T> getRow(Shape shape, T[] entries,
                                                 int[] rowIndices, int[] colIndices,
                                                 int rowIdx, int start, int end) {
        ValidateParameters.validateArrayIndices(shape.getSize(0), rowIdx);
        ValidateParameters.ensureInRange(start, 0, shape.getSize(1), "start");
        ValidateParameters.ensureInRange(end, start, shape.getSize(1), "end");

        IntPair rowStartEnd = SparseElementSearch.matrixFindRowStartEnd(rowIndices, rowIdx);
        int rowStart = rowStartEnd.first();
        int rowEnd = rowStartEnd.second();

        if (rowStart == rowEnd)
            return new SparseVectorData<>(new Shape(end-start), new ArrayList<>(), new ArrayList<>());

        int colStart = Arrays.binarySearch(colIndices, rowStart, rowEnd, start);
        if (colStart < 0) colStart = -colStart - 1;

        int colEnd = Arrays.binarySearch(colIndices, rowStart, rowEnd, end);
        if (colEnd < 0) colEnd = -colEnd - 1;

        if (colStart >= colEnd)
            return new SparseVectorData<>(new Shape(end - start), new ArrayList<>(), new ArrayList<>());

        int initialCapacity = colEnd - colStart;
        List<T> rowEntries = new ArrayList<>(initialCapacity);
        List<Integer> indices = new ArrayList<>(initialCapacity);

        for(int i=colStart; i<colEnd; i++) {
            rowEntries.add(entries[i]);
            indices.add(colIndices[i] - start);
        }

        return new SparseVectorData<>(new Shape(end - start), rowEntries, indices);
    }


    /**
     * Gets a specified column of a COO matrix between {@code start} (inclusive) and {@code end} (exclusive).
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param colIdx Index of the column in this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The column at index {@code colIdx} of this matrix between the {@code start} and {@code end}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code end} are {@code start} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code end} is less than {@code start}.
     */
    public static <T> SparseVectorData<T> getCol(Shape shape, T[] entries,
                                                 int[] rowIndices, int[] colIndices,
                                                 int colIdx, int start, int end) {
        // Validate parameters.
        ValidateParameters.validateArrayIndices(shape.getSize(1), colIdx);
        ValidateParameters.ensureInRange(start, 0, shape.getSize(0), "start");
        ValidateParameters.ensureInRange(end, start, shape.getSize(0), "end");

        List<T> colEntries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for (int i = 0, nnz=entries.length; i < nnz; i++) {
            if (colIndices[i] == colIdx && rowIndices[i] >= start && rowIndices[i] < end) {
                colEntries.add(entries[i]);
                indices.add(rowIndices[i] - start);
            }
        }

        // The shape of the vector is determined by the number of rows in the range
        return new SparseVectorData<>(new Shape(end - start), colEntries, indices);
    }


    /**
     * Checks that parameters are valued for setting a slice of a matrix.
     * @param srcShape Shape of the matrix to set slice of.
     * @param valuesShape Shape of the matrix representing the slice.
     * @param row Starting row for slice.
     * @param col Ending row for slice.
     */
    private static void setSliceParamCheck(
            Shape srcShape, Shape valuesShape, int row, int col) {
        ValidateParameters.validateArrayIndices(srcShape.getSize(0), row);
        ValidateParameters.validateArrayIndices(srcShape.getSize(0), col);

        if(valuesShape.getSize(0) + row > srcShape.getSize(0) || valuesShape.getSize(1) + col > srcShape.getSize(1)) {
            throw new IndexOutOfBoundsException(
                    String.format("Slice of shape %s starting at (%d, %d) does not fit in matrix of shape %s.",
                            valuesShape, row, col, srcShape));
        }
    }


    /**
     * Checks if an index is in the specified slice.
     * @param row Row index of value.
     * @param col Column index of value.
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending column index of slice (exclusive).
     * @return True if the indices are in the slice. False otherwise.
     */
    private static boolean inSlice(int row, int col, int rowStart, int rowEnd, int colStart, int colEnd) {
        return row >= rowStart && row < rowEnd && col >= colStart && col < colEnd;
    }


    /**
     * Copies values in sparse matrix which do not fall in the specified row and column ranges.
     * @param src Source sparse matrix to copy from.
     * @param destEntries Destination list to add copied values to.
     * @param destRowIndices Destination list to add copied row indices to.
     * @param destColIndices Destination list to add copied column indices to.
     * @param rowRange List of row indices to NOT copy from.
     * @param colRange List of column indices to NOT copy from.
     */
    private static <T> void copyValuesNotInSlice(
            T[] srcEntries,
            int[] srcRowIndices,
            int[] srcColIndices,
            List<T> destEntries,
            List<Integer> destRowIndices,
            List<Integer> destColIndices, int[] rowRange, int[] colRange) {

        // Copy values not in slice.
        for(int i=0, size=srcEntries.length; i<size; i++) {
            if( !(ArrayUtils.contains(rowRange, srcRowIndices[i])
                    && ArrayUtils.contains(colRange, srcColIndices[i])) ) {
                // Then the entry is not in the slice so add it.
                destEntries.add(srcEntries[i]);
                destRowIndices.add(srcRowIndices[i]);
                destColIndices.add(srcColIndices[i]);
            }
        }
    }
}
