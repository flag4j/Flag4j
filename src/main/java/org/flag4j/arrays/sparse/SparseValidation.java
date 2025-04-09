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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.Shape;

import java.util.Arrays;

/**
 * Utility class for validating parameters of sparse vectors, matrices, and tensors.
 */
public final class SparseValidation {


    /**
     * Validates constructor parameters for sparse COO vectors.
     * @param size Full size of the COO vector.
     * @param nnz Number of non-zero entries in the COO vector.
     * @param indices Non-zero indices of the COO vector.
     * @throws IllegalArgumentException If the parameters do not specify a valid COO vector.
     */
    public static void validateCoo(int size, int nnz, int[] indices) {
        if(nnz > size) {
            throw new IllegalArgumentException("The number of non-zero entries cannot be greater than the size of the array but got" +
                    " nnz=" + nnz + " and size=" + Arrays.toString(indices));
        }

        if(nnz != indices.length) {
            throw new IllegalArgumentException("The number of non-zero entries must match the number of indices but got nnz="
                    + nnz + " and indices.length=" + indices.length);
        }

        for(int i : indices) {
            if(i < 0 || i >= size)
                throw new IllegalArgumentException("Invalid index (" + i + ") encountered for COO vector of size " + size);
        }
    }


    /**
     * Validates constructor parameters for sparse COO matrices.
     * @param shape Shape of the matrix.
     * @param nnz Number of non-zero values entries in the COO matrix.
     * @param rowIndices The non-zero row indices in the COO matrix.
     * @param colIndices The non-zero column indices in the COO matrix.
     * @throws IllegalArgumentException If the parameters do not specify a valid COO Matrix.
     */
    public static void validateCoo(Shape shape, int nnz, int[] rowIndices, int[] colIndices) {
        if(shape.getRank() != 2)
            throw new IllegalArgumentException("Matrix shape must have rank 2 but got rank " + shape);

        if (shape.isIntSized()) {
            int totalEntries = shape.totalEntriesIntValueExact();

            if(nnz > totalEntries) {
                throw new IllegalArgumentException("The number of non-zero entries cannot be greater than the total number of entries in" +
                        " the matrix but got" + " nnz=" + nnz + " and totalEntries=" + totalEntries);
            }
        }

        int numRows = shape.get(0);
        int numCols = shape.get(1);

        if(nnz != rowIndices.length || nnz != colIndices.length) {
            throw new IllegalArgumentException("The number of non-zero entries must match the number of row and" +
                    " column indices but got nnz="
                    + nnz + ", rowIndices.length=" + rowIndices.length + ", and colIndices.length=" + colIndices.length);
        }

        for(int i=0; i<nnz; i++) {
            if(i < 0 || rowIndices[i] >= numRows || colIndices[i] >= numCols) {
                throw new IllegalArgumentException(String.format("Invalid index [%d, %d] encountered for COO matrix of shape %s",
                        rowIndices[i], colIndices[i], shape));
            }
        }
    }


    /**
     * Validates constructor parameters for sparse COO tensors.
     * @param shape Shape of the tensor.
     * @param nnz Number of non-zero values entries in the COO tensor.
     * @param indices The non-zero indices of the COO tensor.
     */
    public static void validateCoo(Shape shape, int nnz, int[][] indices) {
        int rank = shape.getRank();

        if(shape.isIntSized()) {
            int totalEntries = shape.totalEntriesIntValueExact();
            if(nnz > totalEntries) {
                throw new IllegalArgumentException("The number of non-zero entries cannot be greater than the total number of entries in" +
                        " the tensor but got" + " nnz=" + nnz + " and totalEntries=" + totalEntries);
            }
        }

        if(nnz != indices.length) {
            throw new IllegalArgumentException("The number of non-zero entries must match the number of indices " +
                    "but got nnz=" + nnz + " and indices.length=" + indices.length);
        }

        for(int i=0; i<nnz; i++) {
            int[] idx = indices[i];

            if(idx.length != rank) {
                throw new IllegalArgumentException(String.format(
                        "The dimension of each index must match the rank but got rank=%d and indices[%d].length=%d",
                        rank, i, idx.length));
            }

            for(int j=0; j<rank; j++) {
                int idxDim = idx[j];
                if(idxDim < 0 || idxDim >= shape.get(j) ) {
                    throw new IllegalArgumentException("Invalid nD index " + Arrays.toString(idx)
                            + " encountered for COO tensor of " + "shape " + shape);
                }
            }
        }
    }


    /**
     * Validates constructor parameters for sparse CSR matrices.
     * @param shape Shape of the matrix.
     * @param nnz Number of non-zero values entries in the CSR matrix.
     * @param rowPointers The non-zero row pointers in the CSR matrix.
     * @param colIndices The non-zero column indices in the CSR matrix.
     * @throws IllegalArgumentException If the parameters do not specify a valid CSR Matrix.
     */
    public static void validateCsr(Shape shape, int nnz, int[] rowPointers, int[] colIndices) {
        if(shape.getRank() != 2)
            throw new IllegalArgumentException("Matrix shape must have rank 2 but got rank " + shape);

        if (shape.isIntSized()) {
            int totalEntries = shape.totalEntriesIntValueExact();

            if(nnz > totalEntries) {
                throw new IllegalArgumentException("The number of non-zero entries cannot be greater than the total number " +
                        "of entries in the matrix but got" + " nnz=" + nnz + " and totalEntries=" + totalEntries);
            }
        }

        int numRows = shape.get(0);
        int numCols = shape.get(1);

        if(rowPointers.length != numRows + 1) {
            throw new IllegalArgumentException("Expecting rowPointers to have length (numRows + 1)=" + (numRows+1) +
                    " but got rowPointers.length=" + rowPointers.length);
        }

        if(nnz != colIndices.length) {
            throw new IllegalArgumentException("The number of non-zero entries must match the number of " +
                    " column indices but got nnz=" + nnz + " and colIndices.length=" + colIndices.length);
        }

        int seenItems = 0;

        for(int i=0; i<numRows; i++) {
            int start = rowPointers[i];
            int end = rowPointers[i+1];

            if(start < 0) {
                throw new IllegalArgumentException("All rowPointers must be positive but got " + start + " at index " + i + ".");
            }

            if(start > end) {
                throw new IllegalArgumentException(String.format("rowPointers must be monotonically increasing but got " +
                        "values %d and %d at indices %d and %d.", start, end, i, i+1));
            }

            seenItems += (end - start);

            for(int j=start; j<end; j++) {
                if(colIndices[j] < 0 || colIndices[j] >= numCols) {
                    throw new IllegalArgumentException(String.format("Invalid column index [%d] encountered for " +
                                    "CSR matrix of shape %s", colIndices[i], shape));
                }
            }
        }

        if(seenItems != nnz) {
            throw new IllegalArgumentException(
                    String.format("rowIndices specifies %d elements but nnz=%d. These values must be equal.", seenItems, nnz));
        }
    }
}
