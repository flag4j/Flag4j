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
import org.flag4j.util.ValidateParameters;

/**
 * Utility class containing methods for manipulating a sparse COO tensor, matrix, or vector.
 */
public final class CooManipulations {

    private CooManipulations() {
        // Hide default constructor for utility class.
    }


    /**
     * Swaps two rows, in place, in a sparse COO matrix.
     * @param shape Shape of the matrix to make swap in.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the COO matrix.
     * @param colIndices Column indices of the COO matrix.
     * @param rowIdx1 Index of the first row in the swap.
     * @param rowIdx2 Index of the second row in the swap.
     */
    public static void swapRows(Shape shape, Object[] entries, int[] rowIndices, int[] colIndices, int rowIdx1, int rowIdx2) {
        ValidateParameters.validateArrayIndices(shape.getSize(0), rowIdx1, rowIdx2);

        for(int i=0, size=entries.length; i<size; i++) {
            // Swap row indices.
            if(rowIndices[i]==rowIdx1) rowIndices[i] = rowIdx2;
            else if(rowIndices[i]==rowIdx2) rowIndices[i] = rowIdx1;
        }

        // Ensure the values are properly sorted.
        CooDataSorter.wrap(entries, rowIndices, colIndices).
                sparseSort().
                unwrap(entries, rowIndices, colIndices);
    }


    /**
     * Swaps two columns, in place, in a sparse COO matrix.
     * @param shape Shape of the matrix to make swap in.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the COO matrix.
     * @param colIndices Column indices of the COO matrix.
     * @param colIdx1 Index of the first column in the swap.
     * @param colIdx2 Index of the second column in the swap.
     */
    public static void swapCols(Shape shape, Object[] entries, int[] rowIndices, int[] colIndices, int colIdx1, int colIdx2) {
        ValidateParameters.validateArrayIndices(shape.getSize(1), colIdx1, colIdx2);

        for(int i=0, size=colIndices.length; i<size; i++) {
            // Swap column indices.
            if(colIndices[i]==colIdx1) colIndices[i] = colIdx2;
            else if(colIndices[i]==colIdx2) colIndices[i] = colIdx1;
        }

        // Ensure the values are properly sorted.
        CooDataSorter.wrap(entries, rowIndices, colIndices).
                sparseSort().
                unwrap(entries, rowIndices, colIndices);
    }
}
