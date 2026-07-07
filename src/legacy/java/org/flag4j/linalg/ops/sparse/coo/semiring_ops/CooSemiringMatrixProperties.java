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

package org.flag4j.linalg.ops.sparse.coo.semiring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.numbers.SemiringElement;

/**
 * This class contains low-level implementations for methods to evaluate certain properties of a sparse COO
 * {@link SemiringElement} matrix. For example, if the matrix is symmetric.
 */
public final class CooSemiringMatrixProperties {

    private CooSemiringMatrixProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a complex sparse COO matrix is the identity matrix.
     * @param shape Shape of the matrix.
     * @param entries Non-zero items of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return {@code true} if the {@code src} matrix is the identity matrix; {@code false} otherwise.
     */
    public static <T extends SemiringElement<T>> boolean isIdentity(
            Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        // Ensure the matrix is square and there are at least the same number of non-zero items as items on the diagonal.
        if(shape.getSize(0) != shape.getSize(1) || entries.length<shape.getSize(0)) return false;

        for(int i=0, size=entries.length; i<size; i++) {
            // Ensure value is 1 and on the diagonal.
            if(rowIndices[i] != i && rowIndices[i] != i && !entries[i].isOne()) {
                return false;
            } else if((rowIndices[i] != i || rowIndices[i] != i) && !entries[i].isZero()) {
                return false;
            }
        }

        return true; // If we make it to this point the matrix must be an identity matrix.
    }
}
