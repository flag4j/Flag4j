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

package org.flag4j.linalg.ops.sparse.csr.ring_ops;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.ring_arrays.AbstractCsrRingMatrix;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.ArrayConversions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * This class contains methods to check properties of sparse CSR
 * {@link RingElement} matrices.
 */
public final class CsrRingProperties {

    private CsrRingProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if all items of this tensor are close to the items of the argument {@code tensor}.
     * @param src1 First matrix in the comparison.
     * @param src2 Second matrix in the comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if the {@code src1} matrix is the same shape as the {@code src2} matrix and all items
     * are "close", i.e., elements {@code a} and {@code b} at the same positions in the two matrices respectively
     * satisfy {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     */
    public static <T extends RingElement<T>> boolean allClose(
            AbstractCsrRingMatrix<?, ?, ?, T> src1,
            AbstractCsrRingMatrix<?, ?, ?, T> src2,
            double relTol, double absTol) {
        boolean close = src1.shape.equals(src2.shape);

        if(close) {
            // Remove values which are "close" to zero.
            List<T> src1Entries = new ArrayList<>(src1.data.length);
            List<Integer> src1ColIndices = new ArrayList<>(src1Entries.size());
            int[] src1RowPointers = new int[src1.rowPointers.length];
            removeCloseToZero(src1, src1Entries, src1RowPointers, src1ColIndices, absTol);

            List<T> src2Entries = new ArrayList<>(src2.data.length);
            List<Integer> src2ColIndices = new ArrayList<>(src2Entries.size());
            int[] src2RowPointers = new int[src2.rowPointers.length];
            removeCloseToZero(src2, src2Entries, src2RowPointers, src2ColIndices, absTol);

            close = Arrays.equals(src1RowPointers, src2RowPointers)

                    && Arrays.equals(ArrayConversions.fromIntegerList(src1ColIndices),
                    ArrayConversions.fromIntegerList(src2ColIndices))

                    && RingProperties.allClose(src1Entries.toArray(new RingElement[0]),
                    src2Entries.toArray(new RingElement[0]), relTol, absTol);
        }

        return close;
    }


    /**
     * Removes items in {@code src} which are within {@code atol} in absolute value from zero.
     * @param src Source CSR matrix.
     * @param entries List to store value in {@code src} which are not within {@code atol} in absolute value from zero.
     * @param colIndices Column indices of items.
     * @param rowPointers Row pointers for items.
     * @param aTol Absolute tolerance for value to be considered close to zero.
     */
    private static <T extends RingElement<T>> void removeCloseToZero(
            AbstractCsrRingMatrix<?, ?, ?, T> src,
            List<T> entries, int[] rowPointers,
            List<Integer> colIndices, double aTol) {
        for(int i=0, size=src.numRows; i<size; i++) {
            int start = src.rowPointers[i];
            int stop = src.rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                T value = src.data[j];

                if(value.abs() > aTol) {
                    // Then keep value.
                    entries.add(value);
                    colIndices.add(src.colIndices[j]);
                    rowPointers[i]++;
                }
            }
        }

        // Accumulate row pointers.
        int size = rowPointers.length-1;

        for(int i=0; i<size; i++)
            rowPointers[i+1] += rowPointers[i];
    }


    /**
     * Checks if the {@code src} matrix is close to the identity matrix.
     * @param src The matrix to check.
     * @return True if the {@code src} matrix is close to identity matrix. False otherwise.
     */
    public static <T extends RingElement<T>> boolean isCloseToIdentity(AbstractCsrRingMatrix<?, ?, ?, T> src) {
        if(src.isSquare() && src.colIndices.length >= src.numCols) {
            // Tolerances corresponds to the allClose(...) methods.
            double diagTol = 1.001E-5;
            double nonDiagTol = 1e-08;
            int diagCount = 0;

            final T ONE = src.nnz > 0 ? src.data[0].getOne() : null;

            for(int i=0; i<src.rowPointers.length-1; i++) {
                for(int j=src.rowPointers[i]; j<src.rowPointers[i+1]; j++) {
                    if(src.data[j].sub(ONE).abs() > diagTol) {
                        if(src.colIndices[j] != i) return false; // Diagonal value not close to one.
                        diagCount++;
                    } else if(src.data[i].abs() > nonDiagTol) {
                        return false; // Non-diagonal value is not close to one.
                    }
                }
            }

            return diagCount == src.numCols;
        } else {
            return false;
        }
    }


    /**
     * Checks if a sparse CSR matrix is Hermitian.
     * @param shape Shape of the CSR matrix.
     * @param values Non-zero values of a CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return {@code true} if the CSR matrix is Hermitian (i.e., equal to its conjugate transpose); {@code false} otherwise.
     */
    public static <T extends RingElement<T>> boolean isHermitian(Shape shape, T[] values, int[] rowPointers, int[] colIndices) {
        int numRows = shape.getSize(0);
        int numCols = shape.getSize(1);

        if(numRows != numCols) return false; // Early return for non-square matrix.

        for (int i = 0; i < numRows; i++) {
            int rowStart = rowPointers[i];
            int rowEnd = rowPointers[i + 1];

            for (int idx = rowStart; idx < rowEnd; idx++) {
                int j = colIndices[idx];

                if (j >= i && !values[idx].isZero()) {
                    T val1 = values[idx];

                    // Search for the value with swapped row and column indices.
                    int pos = Arrays.binarySearch(colIndices,  rowPointers[j], rowPointers[j + 1], i);

                    if (pos >= 0) {
                        T val2 = values[pos];

                        // Ensure values  are Equal
                        if (!Objects.equals(val1, val2.conj())) return false;

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
