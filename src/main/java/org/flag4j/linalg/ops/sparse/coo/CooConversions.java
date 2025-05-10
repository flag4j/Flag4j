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

import org.flag4j.arrays.Shape;

/**
 * Utility class for converting COO tensor to another type of tensor.
 */
public final class CooConversions {

    private CooConversions() {
        // Hide default constructor for utility class.
    }


    /**
     * Converts a sparse COO tensor to an equivalent dense tensor.
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices Non-zero indices of the COO tensor.
     * @param dest Array to store the dense result in.
     * @throws IllegalArgumentException If {@code dest.length != shape.totalEntriesIntValueExact()}.
     */
    public static <T> void toDense(Shape shape, T[] entries, int[][] indices, T[] dest) {
        if(dest.length != shape.totalEntriesIntValueExact()) {
            throw new IllegalArgumentException("Cannot store data from tensor with shape "
                    + shape + " in an array of length " + dest.length + ".");
        }

        for(int i=0, nnz=entries.length; i<nnz; i++)
            dest[shape.unsafeGet1DIndex(indices[i])] = entries[i];
    }


    /**
     * Converts a COO matrix to an equivalent CSR matrix.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param destEntries Array to store non-zero data of the CSR matrix.
     * @param destRowPointers Array to store non-zero row pointers of the CSR matrix.
     * @param destColIndices Array to store non-zero column indices of the CSR matrix.
     */
    public static <T> void toCsr(Shape shape, T[] entries, int[] rowIndices, int[] colIndices,
                                 T[] destEntries, int[] destRowPointers, int[] destColIndices) {
        final int numRows = shape.getSize(0);

        // Copy the non-zero data and column indices. Count number of data per row.
        for(int i=0, size=entries.length; i<size; i++)
            destRowPointers[rowIndices[i] + 1]++;

        //Accumulate row pointers.
        for(int i=1, size=destRowPointers.length; i<size; i++)
            destRowPointers[i] += destRowPointers[i-1];

        // Copy non-zero data and column indices.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
    }
}
