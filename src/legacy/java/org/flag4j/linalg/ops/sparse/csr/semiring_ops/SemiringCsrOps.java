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

package org.flag4j.linalg.ops.sparse.csr.semiring_ops;

import org.flag4j.numbers.SemiringElement;

/**
 * Utility class for computing ops on sparse CSR {@link SemiringElement} tensors.
 */
public final class SemiringCsrOps {

    private SemiringCsrOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the trace of a sparse CSR matrix. That is, the sum of values along the principle diagonal of the matrix.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return The trace of the CSR matrix. If there are no non-zero items in this matrix along the principle diagonal, this method
     * will attempt to return the zero-element of the semiring. However, if {@code items.length == 0} then this cannot be
     * determined and {@code null} will be returned instead.
     */
    public static <T extends SemiringElement<T>> T trace(T[] entries, int[] rowPointers, int[] colIndices) {
        T tr = (entries.length > 0) ? entries[0].getZero() : null;

        for(int i=0, numRows=rowPointers.length-1; i<numRows; i++) {
            for(int j=rowPointers[i], rowEnd=rowPointers[i+1]; j<rowEnd; j++)
                if(colIndices[j] == i) tr = tr.add(entries[j]);
        }

        return tr;
    }
}
