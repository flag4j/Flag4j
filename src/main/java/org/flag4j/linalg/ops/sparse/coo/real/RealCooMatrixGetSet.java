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

package org.flag4j.linalg.ops.sparse.coo.real;

import org.flag4j.arrays.IntPair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.sparse.SparseElementSearch;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.util.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * This class provides methods for getting and setting elements and slices from/to a real sparse matrix.
 */
public final class RealCooMatrixGetSet {

    private RealCooMatrixGetSet() {
        // Hide default constructor for utility class.
    }


    /**
     * Gets the specified element from a sparse matrix.
     * @param src Source matrix to get value from.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return The value in the sparse matrix at the specified indices.
     */
    public static double matrixGet(CooMatrix src, int row, int col) {
        int idx = SparseElementSearch.matrixBinarySearch(src.rowIndices, src.colIndices, row, col);
        return idx<0 ? 0 : src.data[idx];
    }


    /**
     * Sets the specified element from a sparse matrix.
     * @param src Sparse matrix to set value in.
     * @param row Row index of the value to set in the sparse matrix.
     * @param col Column index of the value to set in the sparse matrix.
     * @param value Value to set.
     * @return The
     */
    public static CooMatrix matrixSet(CooMatrix src, int row, int col, double value) {
        // Find position of row index within the row indices if it exits.
        int idx = SparseElementSearch.matrixBinarySearch(src.rowIndices, src.colIndices, row, col);
        double[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
            // No non-zero-element with these indices exists. Insert new value.
            destEntries = new double[src.data.length + 1];
            System.arraycopy(src.data, 0, destEntries, 0, -idx-1);
            destEntries[-idx-1] = value;
            System.arraycopy(src.data, -idx-1, destEntries, -idx, src.data.length+idx+1);

            destRowIndices = new int[src.data.length + 1];
            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, -idx-1);
            destRowIndices[-idx-1] = row;
            System.arraycopy(src.rowIndices, -idx-1, destRowIndices, -idx, src.rowIndices.length+idx+1);

            destColIndices = new int[src.data.length + 1];
            System.arraycopy(src.colIndices, 0, destColIndices, 0, -idx-1);
            destColIndices[-idx-1] = col;
            System.arraycopy(src.colIndices, -idx-1, destColIndices, -idx, src.colIndices.length+idx+1);
        } else {
            // Value with these indices exists. Simply update value.
            destEntries = Arrays.copyOf(src.data, src.data.length);
            destEntries[idx] = value;
            destRowIndices = src.rowIndices.clone();
            destColIndices = src.colIndices.clone();
        }

        return CooMatrix.unsafeMake(src.shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a specified row of a real sparse COO matrix to the values in a sparse COO vector.
     * @param src Sparse COO matrix to set row in.
     * @param rowIdx Index of the row to set.
     * @param row Sparse COO vector containing new data for the row.
     * @return A new COO matrix resulting from setting row {@code rowIdx} in {@code src} to {@code row}.
     * @throws IllegalArgumentException If {@code row.length != src.numCols}.
     */
    public static CooMatrix setRow(CooMatrix src, int rowIdx, CooVector row) {
        ValidateParameters.validateArrayIndices(src.numRows, rowIdx);
        ValidateParameters.ensureAllEqual(src.numCols, row.size);

        int[] rowArray = new int[row.nnz];
        Arrays.fill(rowArray, rowIdx);

        List<Double> entries = ArrayConversions.toArrayList(row.data);
        List<Integer> rowIndices = ArrayConversions.toArrayList(rowArray);
        List<Integer> colIndices = ArrayConversions.toArrayList(row.indices);

        for(int i=0, size=src.nnz; i<size; i++) {
            int srcRow = src.rowIndices[i];

            if(srcRow != rowIdx) {
                entries.add(src.data[i]);
                rowIndices.add(srcRow);
                colIndices.add(src.colIndices[i]);
            }
        }

        return new CooMatrix(src.shape, entries, rowIndices, colIndices).sortIndices();
    }


    /**
     * Sets a specified row of a real sparse matrix to the values of a dense array.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the data of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static CooMatrix setRow(CooMatrix src, int rowIdx, double[] row) {
        ValidateParameters.validateArrayIndices(src.numRows, rowIdx);
        ValidateParameters.ensureAllEqual(src.numCols, row.length);

        IntPair startEnd = SparseElementSearch.matrixFindRowStartEnd(src.rowIndices, rowIdx);
        int start = startEnd.first();
        int end = startEnd.second();

        double[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No data with row index found.
            destEntries = new double[src.data.length + row.length];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.data, 0, destEntries, 0, -start-1);
            System.arraycopy(row, 0, destEntries, -start-1, row.length);
            System.arraycopy(
                    src.data, -start-1,
                    destEntries, -start-1+row.length, destEntries.length-(row.length - start - 1)
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, -start-1);
            Arrays.fill(destRowIndices, -start-1, -start-1+row.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, -start-1,
                    destRowIndices, -start-1+row.length, destRowIndices.length-(row.length - start - 1)
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, -start-1);
            System.arraycopy(ArrayBuilder.intRange(0, src.numCols), 0, destColIndices, -start-1, row.length);
            System.arraycopy(
                    src.colIndices, -start-1,
                    destColIndices, -start-1+row.length, destColIndices.length-(row.length - start - 1)
            );

        } else {
            end++; // Make end index exclusive to simplify computations.

            // Entries with row index found.
            destEntries = new double[src.data.length + row.length - (end-start)];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.data, 0, destEntries, 0, start);
            System.arraycopy(row, 0, destEntries, start, row.length);
            System.arraycopy(
                    src.data, end,
                    destEntries, start + row.length, destEntries.length-(start + row.length)
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, start);
            Arrays.fill(destRowIndices, start, start+row.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, end,
                    destRowIndices, start + row.length, destEntries.length-(start + row.length)
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, start);
            System.arraycopy(ArrayBuilder.intRange(0, src.numCols), 0, destColIndices, start, row.length);
            System.arraycopy(
                    src.colIndices, end,
                    destColIndices, start + row.length, destEntries.length-(start + row.length)
            );
        }

        return CooMatrix.unsafeMake(src.shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a column of a sparse matrix to the data of a dense array.
     * @param src Source matrix to set column of.
     * @param colIdx The index of the column to set within the {@code src} matrix.
     * @param col The dense array containing the new column data for the {@code src} array.
     * @return A copy of the {@code src} matrix with the specified column set to the dense array.
     * @throws IllegalArgumentException If the {@code colIdx} is not within the range of the matrix.
     * @throws IllegalArgumentException If the {@code col} array does not have the same length as the number of
     * rows in {@code src} matrix.
     */
    public static CooMatrix setCol(CooMatrix src, int colIdx, double[] col) {
        ValidateParameters.validateArrayIndices(src.numCols, colIdx);
        ValidateParameters.ensureAllEqual(src.numRows, col.length);

        Integer[] colIndices = new Integer[col.length];
        Arrays.fill(colIndices, colIdx);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<Double> destEntries = DoubleStream.of(col).boxed().collect(Collectors.toList());
        List<Integer> destRowIndices = IntStream.of(
                ArrayBuilder.intRange(0, col.length)
        ).boxed().collect(Collectors.toList());
        List<Integer> destColIndices = new ArrayList<>(Arrays.asList(colIndices));

        return addNotInCol(destEntries, destRowIndices, destColIndices, src, colIdx);
    }


    /**
     * Sets a column of a sparse matrix to the values in a sparse tensor.
     * @param src Source matrix to set column of.
     * @param colIdx Index of the column to set.
     * @param col New data for the specified column.
     * @return A copy of the {@code src} matrix with the specified column set to the {@code col} sparse vector.
     * @throws IllegalArgumentException If the {@code src} matrix does not have the same number of rows as total data
     * in the {@code col} vector.
     */
    public static CooMatrix setCol(CooMatrix src, int colIdx, CooVector col) {
        ValidateParameters.validateArrayIndices(src.numCols, colIdx);
        ValidateParameters.ensureAllEqual(src.numRows, col.size);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<Double> destEntries = ArrayConversions.toArrayList(col.data);
        List<Integer> destRowIndices = ArrayConversions.toArrayList(col.indices);
        List<Integer> destColIndices = ArrayConversions.toArrayList(ArrayBuilder.filledArray(col.nnz, colIdx));

        return addNotInCol(destEntries, destRowIndices, destColIndices, src, colIdx);
    }


    /**
     * Adds values from a sparse matrix to specified lists if the value is not within a specified column.
     * @param destEntries List to add non-zero data from sparse matrix to.
     * @param destRowIndices List to add non-zero row indices from sparse matrix to.
     * @param destColIndices List to add non-zero column indices from sparse matrix to.
     * @param src The sparse matrix to get non-zero values and indices from.
     * @param colIdx Specified column to not add data to the lists from.
     * @return A sparse matrix made from the resulting {@code destEntries, destRowIndices} and {@code destColIndices}
     * lists.
     */
    private static CooMatrix addNotInCol(List<Double> destEntries, List<Integer> destRowIndices,
                                         List<Integer> destColIndices, CooMatrix src, int colIdx) {
        for(int i = 0; i<src.data.length; i++) {
            // Add all data which are not in the specified column.
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.data[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        CooMatrix dest = new CooMatrix(
                src.shape,
                destEntries,
                destRowIndices,
                destColIndices
        );

        dest.sortIndices();

        return dest;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the data of another sparse matrix.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} matrix.
     * @throws IllegalArgumentException If the {@code values} matrix does not fit in the {@code src}
     * matrix given the row and
     * column index.
     */
    public static CooMatrix setSlice(CooMatrix src, CooMatrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ValidateParameters.validateArrayIndices(src.numRows, row);
        ValidateParameters.validateArrayIndices(src.numCols, col);

        ValidateParameters.ensureLessEq(src.numRows, values.numRows + row);
        ValidateParameters.ensureLessEq(src.numCols, values.numCols + col);

        // Initialize lists to new values for the specified slice.
        List<Double> entries = ArrayConversions.toArrayList(values.data);
        List<Integer> rowIndices = ArrayConversions.toArrayList(ArrayUtils.shift(row, values.rowIndices));
        List<Integer> colIndices = ArrayConversions.toArrayList(ArrayUtils.shift(col, values.colIndices));

        int[] rowRange = ArrayBuilder.intRange(row, values.numRows + row);
        int[] colRange = ArrayBuilder.intRange(col, values.numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure data are properly sorted.
        CooMatrix mat = new CooMatrix(src.shape, entries, rowIndices, colIndices);
        mat.sortIndices();

        return mat;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the data of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static CooMatrix setSlice(CooMatrix src, double[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ValidateParameters.ensureLessEq(src.numRows, values.length + row);
        ValidateParameters.ensureLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = ArrayUtils.flatten(values);
        int[] sliceRows = ArrayBuilder.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayJoiner.repeat(values.length, ArrayBuilder.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the data of a dense matrix.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense matrix containing values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static CooMatrix setSlice(CooMatrix src, Matrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ValidateParameters.ensureLessEq(src.numRows, values.numRows + row);
        ValidateParameters.ensureLessEq(src.numCols, values.numCols + col);

        int[] sliceRows = ArrayBuilder.intRange(row, values.numRows + row, values.numCols);
        int[] sliceCols = ArrayJoiner.repeat(values.numRows, ArrayBuilder.intRange(col, values.numCols + col));

        return setSlice(src, values.data, values.numRows, values.numCols, sliceRows, sliceCols, row, col);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the data of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static CooMatrix setSlice(CooMatrix src, int[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ValidateParameters.ensureLessEq(src.numRows, values.length + row);
        ValidateParameters.ensureLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = new double[values.length*values[0].length];
        int pos = 0;
        for(int[] vRow : values) {
            for(int d : vRow) {
                flatValues[pos++] = d;
            }
        }
        int[] sliceRows = ArrayBuilder.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayJoiner.repeat(values.length, ArrayBuilder.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
    }


    /**
     * Sets a slice of a sparse matrix to values given in a 1d dense array.
     * @param src Source sparse matrix to copy non-slice from.
     * @param values Dense value for slice.
     * @param numRows Number of rows in the matrix represented by {@code values}.
     * @param numCols Number of columns in the matrix represented by {@code values}.
     * @param sliceRows Row indices for slice.
     * @param sliceCols Column indices for slice.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the specified values.
     */
    private static CooMatrix setSlice(CooMatrix src, double[] values, int numRows, int numCols,
                                      int[] sliceRows, int[] sliceCols, int row, int col) {
        // Copy vales and row/column indices (with appropriate shifting) to destination lists.
        List<Double> entries = ArrayConversions.toArrayList(values);
        List<Integer> rowIndices = ArrayConversions.toArrayList(sliceRows);
        List<Integer> colIndices = ArrayConversions.toArrayList(sliceCols);

        int[] rowRange = ArrayBuilder.intRange(row, numRows + row);
        int[] colRange = ArrayBuilder.intRange(col, numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure data are properly sorted.
        CooMatrix mat = new CooMatrix(src.shape, entries, rowIndices, colIndices);
        mat.sortIndices();

        return mat;
    }


    /**
     * Gets a specified row from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @return Returns the specified row from this sparse matrix.
     */
    public static CooVector getRow(CooMatrix src, int rowIdx) {
        ValidateParameters.validateArrayIndices(src.numRows, rowIdx);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.rowIndices[i]==rowIdx) {
                entries.add(src.data[i]);
                indices.add(src.colIndices[i]);
            }
        }

        return new CooVector(src.numCols, entries, indices);
    }


    /**
     * Gets a specified row range from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @param start Staring column index of the column to be extracted (inclusive).
     * @param end Ending column index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static CooVector getRow(CooMatrix src, int rowIdx, int start, int end) {
        ValidateParameters.validateArrayIndices(src.numRows, rowIdx);
        ValidateParameters.validateArrayIndices(src.numCols, start, end-1);
        ValidateParameters.ensureLessEq(end-1, start);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.rowIndices[i]==rowIdx && src.colIndices[i] >= start && src.colIndices[i] < end) {
                entries.add(src.data[i]);
                indices.add(src.colIndices[i]);
            }
        }

        return new CooVector(end-start, entries, indices);
    }


    /**
     * Gets a specified column from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @return Returns the specified column from this sparse matrix.
     */
    public static CooVector getCol(CooMatrix src, int colIdx) {
        ValidateParameters.validateArrayIndices(src.numCols, colIdx);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.colIndices[i]==colIdx) {
                entries.add(src.data[i]);
                indices.add(src.rowIndices[i]);
            }
        }

        return new CooVector(src.numRows, entries, indices);
    }


    /**
     * Gets a specified column range from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @param start Staring row index of the column to be extracted (inclusive).
     * @param end Ending row index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static CooVector getCol(CooMatrix src, int colIdx, int start, int end) {
        ValidateParameters.validateArrayIndices(src.numCols, colIdx);
        ValidateParameters.validateArrayIndices(src.numRows, start, end);
        ValidateParameters.ensureLessEq(end, start);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.colIndices[i]==colIdx && src.rowIndices[i] >= start && src.rowIndices[i] < end) {
                entries.add(src.data[i]);
                indices.add(src.rowIndices[i]);
            }
        }

        return new CooVector(end-start, entries, indices);
    }


    /**
     * Gets a specified rectangular slice of a sparse matrix.
     * @param src Sparse matrix to extract slice from.
     * @param rowStart Starting row index of the slice (inclusive).
     * @param rowEnd Ending row index of the slice (exclusive).
     * @param colStart Staring column index of a slice (inclusive).
     * @param colEnd Ending column index of the slice (exclusive).
     * @return The specified slice of the sparse matrix.
     */
    public static CooMatrix getSlice(CooMatrix src, int rowStart, int rowEnd, int colStart, int colEnd) {
        SparseUtils.validateSlice(src.shape, rowStart, rowEnd, colStart, colEnd);

        Shape shape = new Shape(rowEnd-rowStart, colEnd-colStart);
        List<Double> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        // Search for item with indices matching upper left index of slice.
        int start = SparseElementSearch.matrixBinarySearch(src.rowIndices, src.colIndices, rowStart, colStart);

        if(start < 0) {
            // If no item with the specified indices is found, then begin search at the insertion point.
            start = -start - 1;
        }

        for(int i = start; i<src.data.length; i++) {
            if(inSlice(src.rowIndices[i], src.colIndices[i], rowStart, rowEnd, colStart, colEnd)) {
                entries.add(src.data[i]);
                rowIndices.add(src.rowIndices[i]-rowStart);
                colIndices.add(src.colIndices[i]-colStart);
            }
        }

        return new CooMatrix(shape, entries, rowIndices, colIndices);
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
     * @param entries Destination list to add copied values to.
     * @param rowIndices Destination list to add copied row indices to.
     * @param colIndices Destination list to add copied column indices to.
     * @param rowRange List of row indices to NOT copy from.
     * @param colRange List of column indices to NOT copy from.
     */
    private static void copyValuesNotInSlice(CooMatrix src, List<Double> entries, List<Integer> rowIndices,
                                             List<Integer> colIndices, int[] rowRange, int[] colRange) {
        // Copy values not in slice.
        for(int i = 0; i<src.data.length; i++) {

            if( !(ArrayUtils.contains(rowRange, src.rowIndices[i])
                    && ArrayUtils.contains(colRange, src.colIndices[i])) ) {
                // Then the entry is not in the slice so add it.
                entries.add(src.data[i]);
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i]);
            }
        }
    }
}
