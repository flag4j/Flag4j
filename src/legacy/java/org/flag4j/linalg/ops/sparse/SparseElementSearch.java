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

package org.flag4j.linalg.ops.sparse;

import org.flag4j.arrays.IntPair;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This class provides methods for efficiently finding if a sparse vector, matrix, or tensor contains a non-zero
 * item at a specified index.
 */
public final class SparseElementSearch {

    private SparseElementSearch() {
        // Hide default constructor for utility class.
    }


    /**
     * Preforms a binary search along the row and column indices of the non-zero values of a sparse matrix for the location
     * of an entry with the specified target indices.
     *
     * @param rowIndices Row indices of the matrix to search within.
     * @param colIndices Column indices of the matrix to search within.
     * @param rowKey Target row index.
     * @param colKey Target col index.
     * @return The location of the non-zero-element (within the non-zero values array of {@code src}) with the specified
     *         row and column indices. If this value does not exist, then <code>(-(<i>insertion point</i>) - 1)</code>
     *         will be returned. The <i>insertion point</i> is defined as the point at which the
     *         value, with the row and column key, would be inserted into the array: the index of the first
     *         element greater than the key, or {@code src.items.length} if all
     *         elements in the array are less than the specified key.  Note
     *         that this guarantees that the return value will be &gt;= 0 if
     *         and only if the key is found.
     */
    public static int matrixBinarySearch(int[] rowIndices, int[] colIndices, int rowKey, int colKey) {
        if(rowKey < 0 || colKey < 0)
            throw new IllegalArgumentException("rowKey and colKey must be non-negative but got " + rowKey + " and " + colKey + ".");

        IntPair rowStartEnd = matrixFindRowStartEnd(rowIndices, rowKey);
        int rowStart = rowStartEnd.first();
        int rowEnd = rowStartEnd.second();

        if(rowStart < 0) return rowStart;

        // Perform binary search on the column indices within the found range.
        int colIdx = Arrays.binarySearch(colIndices, rowStart, rowEnd, colKey);

        return colIdx;
    }


    /**
     * Finds the indices of the first and last non-zero-element in the specified row of a sparse matrix. If there is no non-zero
     * element in the sparse matrix at the specified row, negative values will be returned.
     * @param rowIndices Row indices of the matrix to search within.
     * @param rowKey Index of the row to search for within the row indices of the {@code src} matrix.
     * @return If it exists, the first and last index of the non-zero-element in the sparse matrix which has the specified
     * {@code rowKey} as its row index.
     */
    public static IntPair matrixFindRowStartEnd(int[] rowIndices, int rowKey) {
        int rowIdx = Arrays.binarySearch(rowIndices, rowKey);
        if(rowIdx < 0) return new IntPair(rowIdx, rowIdx); // Row not found.

        // Find the first entry with the specified row key.
        int lowerBound = rowIdx;
        while(lowerBound > 0 && rowIndices[lowerBound - 1] == rowKey)
            lowerBound--;

        // Find the last entry with the specified row key.
        int upperBound = rowIdx + 1;
        int length = rowIndices.length - 1;

        while(upperBound < length && rowIndices[upperBound + 1] == rowKey)
            upperBound++;

        return new IntPair(lowerBound, upperBound);
    }


    /**
     * Performs a binary search of the indices of a sparse COO tensor to find a target index.
     * @param indices The non-zero indices of the COO tensor. Assumed to be rectangular array.
     * @param target The target index to find in {@code indices}. Must satisfy {@code target.length == indices[0].length} if {@code
     * indices.length > 0}.
     * @return index of the search key, if it is contained in the array
     *         within the specified range;
     *         otherwise, <code>(-(<i>insertion point</i>) - 1)</code>.  The
     *         <i>insertion point</i> is defined as the point at which the
     *         key would be inserted into the array: the index of the first
     *         element in the range greater than the key,
     *         or {@code toIndex} if all
     *         elements in the range are less than the specified key.  Note
     *         that this guarantees that the return value will be &gt;= 0 if
     *         and only if the key is found.
     * @throws IllegalArgumentException If {@code target.length != indices[0].length}.
     */
    public static int binarySearchCoo(int[][] indices, int[] target) {
        if (indices.length == 0) return -1; // Quick return for zero length array.

        ValidateParameters.ensureArrayLengthsEq(indices[0].length, target.length);
        int left = 0;
        int right = indices.length - 1;

        while (left <= right) {
            int mid = (left + right) >>> 1;
            int cmp = Arrays.compare(indices[mid], target); // Perform lexicographical comparison.

            if (cmp == 0)
                return mid; // Found the target indices.
            else if (cmp < 0)
                left = mid + 1;
            else
                right = mid - 1;
        }

        // Target indices not found; return the insertion point instead.
        return - (left + 1);
    }
}
