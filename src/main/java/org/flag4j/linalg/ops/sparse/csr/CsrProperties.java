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

import java.util.Arrays;
import java.util.Objects;

/**
 * Utility class containing methods useful for determining certain properties of general CSR matrices.
 */
public final class CsrProperties {

    private CsrProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a sparse CSR matrix is symmetric.
     * @param shape Shape of the CSR matrix.
     * @param values Non-zero values of a CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param zeroValue Any value in {@code values} equal to {@code zeroValue}
     * will be considered zero and will not be considered when determining the symmetry. Equality is determined according to
     * {@link Objects#equals(Object, Object)} where if one of the parameters is {@code null} then the result will always be {@code
     * false}. This means passing {@code zeroValue = null} will result in all items in {@code values} being considered. This is
     * useful if there is no definable zero value for the values of the CSR matrix.
     * @return {@code true} if the CSR matrix is symmetric; {@code false} otherwise.
     */
    public static <T> boolean isSymmetric(Shape shape, T[] values, int[] rowPointers, int[] colIndices, T zeroValue) {
        int numRows = shape.getSize(0);
        int numCols = shape.getSize(1);

        if(numRows != numCols) return false; // Early return for non-square matrix.

        for (int i = 0; i < numRows; i++) {
            int rowStart = rowPointers[i];
            int rowEnd = rowPointers[i + 1];

            for (int idx = rowStart; idx < rowEnd; idx++) {
                int j = colIndices[idx];

                if (j >= i && !Objects.equals(values[idx], zeroValue)) {
                    T val1 = values[idx];

                    // Search for the value with swapped row and column indices.
                    int pos = Arrays.binarySearch(colIndices,  rowPointers[j], rowPointers[j + 1], i);

                    if (pos >= 0) {
                        T val2 = values[pos];

                        // Ensure values  are Equal
                        if (!Objects.equals(val1, val2)) return false;

                    } else {
                        // Corresponding value not found.
                        return false;
                    }
                }
            }
        }

        return true;
    }
}
