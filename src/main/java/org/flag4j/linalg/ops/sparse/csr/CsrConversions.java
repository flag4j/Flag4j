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

package org.flag4j.linalg.ops.sparse.csr;

import org.flag4j.arrays.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * Utility class for converting CSR matrices to flat matrices, dense matrices, etc.
 */
public final class CsrConversions {

    private CsrConversions() {
        // Hide default constructor for utility class.
    }

    /**
     * Computes the new row pointers and column indices for a sparse CSR matrix flattened along some {@code axis}.
     * @param shape Shape of the CSR matrix to be flattened.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param axis Axis along which to flatten matrix. Must be 0 or 1.
     * <ul>
     *     <li>If {@code axis==0} then the matrix will be flattened to a single row.</li>
     *     <li>If {@code axis==1} then the matrix will be flattened to a single column.</li>
     * </ul>
     * @param destRowPointers Array to store the non-zero row pointers resulting from flattening the CSR matrix along the specified
     * axis.
     * @param destColIndices Array to store the non-zero column indices resulting from flattening the CSR matrix along the specified
     * axis.
     * @return The shape of the flattened matrix.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code axis != 0 || axis != 1}.
     */
    public static <T> Shape flatten(Shape shape, T[] entries, int[] rowPointers, int[] colIndices,
                                   int axis, int[] destRowPointers, int[] destColIndices) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int numRows = shape.getSize(0);
        int numCols = shape.getSize(1);
        int nnz = entries.length;

        Shape destShape;

        if (axis == 0) {
            // Flatten to a single row.
            destShape = new Shape(1, shape.numelIntValueExact());

            // Compute flattened column indices.
            for(int i=0; i<numRows; i++) {
                int rowOffset = i*numCols;

                for(int j=rowPointers[i], rowLength = rowPointers[i+1]; j<rowLength; j++)
                    destColIndices[j] = rowOffset + colIndices[j];
            }

        } else {
            // Flatten to a single column.
            int flatSize = shape.numelIntValueExact();
            destShape = new Shape(flatSize, 1);

            // Identify rows with non-zero value.
            for(int i=0; i<numRows; i++) {
                int rowOffset = i*numCols;

                for(int j=rowPointers[i], rowLength = rowPointers[i+1]; j<rowLength; j++)
                    destRowPointers[rowOffset + colIndices[j] + 1] = 1;
            }

            // Accumulate row pointers.
            ArrayUtils.cumSum(destRowPointers, destRowPointers);
        }

        return destShape;
    }


    /**
     * Converts a sparse CSR matrix to a dense matrix.
     * @param shape Shape of the CSR matrix. Must be rank 2.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param dest Array to store the dense result in.
     * @param zero Zero element to fill zero value of the sparse matrix with.
     * @throws IllegalArgumentException If {@code dest.length < shape.totalEntriesIntValueExact()}
     */
    public static <T> void toDense(Shape shape, T[] entries, int[] rowPointers, int[] colIndices,
                                   T[] dest, T zero) {
        if(dest.length < shape.numelIntValueExact()) {
            throw new IllegalArgumentException("Dense destination array of length " + dest.length + " is too small to store values " +
                    "for shape " + shape);
        }
        ValidateParameters.ensureRank(shape, 2);
        Arrays.fill(dest, zero);

        int numCols = shape.getSize(1);

        for(int i = 0, numRows = shape.getSize(0); i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=rowPointers[i], rowEnd=rowPointers[i+1]; j<rowEnd; j++)
                dest[rowOffset + colIndices[j]] = entries[j];
        }
    }


    /**
     * Converts a sparse CSR matrix to an equivalent sparse COO matrix.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param destEntries Array to store the non-zero data of the COO matrix.
     * @param rowIndices Array to store the non-zero row indices of the COO matrix.
     * @param destColIndices Array to store the non-zero column indices of the COO matrix.
     */
    public static <T> void toCoo(Shape shape, T[] entries, int[] rowPointers, int[] colIndices,
                                 T[] destEntries, int[] destRowIndices, int[] destColIndices) {
        final int numRows = shape.getSize(0);

        // Find and copy row indices of non-zero data in the CSR matrix.
        for(int i=0; i<numRows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++)
                destRowIndices[j] = i;
        }

        // Copy non-zero data and column indices.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
    }
}
