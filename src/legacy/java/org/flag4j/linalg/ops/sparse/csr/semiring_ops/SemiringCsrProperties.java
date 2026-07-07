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

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCsrSemiringMatrix;
import org.flag4j.numbers.SemiringElement;

/**
 * Utility class containing methods useful for determining certain properties of a
 * sparse CSR {@link SemiringElement} matrix.
 */
public final class SemiringCsrProperties {

    private SemiringCsrProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a sparse CSR matrix is upper-triangular.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR Matrix.
     * @return {@code true} if the CSR matrix is upper-triangular; {@code false} otherwise.
     */
    public static <T extends SemiringElement<T>> boolean isTriU(Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        final int numRows = shape.getSize(0);
        final int numCols = shape.getSize(1);

        if(numRows != numCols) return false; // Early return for non-square matrix.

        for(int i=1; i<numRows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++) {
                if(colIndices[j] >= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(!entries[j].isZero()) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point, then the matrix must be upper triangular.
    }


    /**
     * Checks if a sparse CSR matrix is lower-triangular.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero items of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR Matrix.
     * @return {@code true} if the CSR matrix is lower-triangular; {@code false} otherwise.
     */
    public static <T extends SemiringElement<T>> boolean isTriL(Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        final int numRows = shape.getSize(0);
        final int numCols = shape.getSize(1);

        if(numRows != numCols) return false; // Early return for non-square matrix.

        for(int i=0; i<numRows; i++) {
            for(int j=rowPointers[i+1]-1, rowStart=rowPointers[i]; j>=rowStart; j--) {
                if(colIndices[j] <= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(!entries[j].isZero()) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point, then the matrix must be lower-triangular.
    }


    /**
     * Checks if the {@code src} matrix is the identity matrix.
     * @param src The matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. False otherwise.
     */
    public static <T extends SemiringElement<T>> boolean isIdentity(Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        final int numRows = shape.getSize(0);
        final int numCols = shape.getSize(1);

        // Check for early return for non-square matrix or if there are not enough non-zeros to cover principle diagonal.
        if(numRows != numCols || colIndices.length < numCols)
            return false;

        int diagCount = 0;

        for(int i=0; i<rowPointers.length-1; i++) {
            for(int j=rowPointers[i], rowStop=rowPointers[i+1]; j<rowStop; j++) {
                if(entries[j].isOne()) {
                    if(colIndices[j] != i) return false;
                    diagCount++;
                } else if(!entries[j].isZero()) {
                    return false;
                }
            }
        }

        return diagCount == numCols;
    }


    /**
     * Checks if the {@code src} matrix is the identity matrix.
     * @param src The matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. False otherwise.
     */
    public static <T extends SemiringElement<T>> boolean isIdentity(AbstractCsrSemiringMatrix<?, ?, ?, T> src) {
        if(src.isSquare() && src.colIndices.length >= src.numCols) {
            int diagCount = 0;

            for(int i=0; i<src.rowPointers.length-1; i++) {
                for(int j=src.rowPointers[i]; j<src.rowPointers[i+1]; j++) {
                    if(src.data[j].isOne()) {
                        if(src.colIndices[j] != i) return false;
                        diagCount++;
                    } else if(!src.data[j].isZero()) {
                        return false;
                    }
                }
            }

            return diagCount == src.numCols;
        } else {
            return false;
        }
    }
}
