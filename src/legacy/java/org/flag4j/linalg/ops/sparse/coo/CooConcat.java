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
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * <p>This utility class contains methods for combining or joining sparse COO tensors, matrices, and vectors.
 * <p>All methods in this class will result in non-zero items and indices being lexicographically sorted.
 */
public final class CooConcat {

    private CooConcat() {
        // Hide default constructor for utility class.
    }


    /**
     * <p>Stacks two sparse COO matrices along columns.
     * <p>WARNING: This method does <em>not</em> perform any bounds checks. If the destination ndarrays are not large enough, an
     * {@link IndexOutOfBoundsException} will be thrown.
     *
     * @param src1Entries Non-zero items of the first COO matrix to stack.
     * @param src1RowIndices Row indices of the first COO matrix to stack.
     * @param src1ColIndices Column indices of the first COO matrix to stack.
     * @param src1NumRows The number of rows in the first COO matrix. This is required to compute the shifted indices of the second
     * matrix.
     * @param src2Entries Non-zero items of the second COO matrix to stack.
     * @param src2RowIndices Row indices of the second COO matrix to stack.
     * @param src2ColIndices Column indices of the second COO matrix to stack.
     * @param destEntries Array to store the non-zero items resulting from the stacking the two COO matrices.
     * @param destRowIndices Array to store the row indices resulting from the stacking the two COO matrices.
     * @param destColIndices Array to store the column indices resulting from the stacking the two COO matrices.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     * @see #augment(Object[], int[], int[], int, Object[], int[], int[], Object[], int[], int[])
     * @see #augmentVector(Object[], int[], int[], int, Object[], int[], Object[], int[], int[])
     * @throws IndexOutOfBoundsException If copying ndarrays to the destination ndarrays would cause access of items outside the array.
     */
    public static <T> void stack(T[] src1Entries, int[] src1RowIndices, int[] src1ColIndices, int src1NumRows,
                                 T[] src2Entries, int[] src2RowIndices, int[] src2ColIndices,
                                 T[] destEntries, int[] destRowIndices, int[] destColIndices) {
        // Copy non-zero values.
        System.arraycopy(src1Entries, 0, destEntries, 0, src1Entries.length);
        System.arraycopy(src2Entries, 0, destEntries, src1Entries.length, src2Entries.length);

        // Copy row indices.
        int[] shiftedRowIndices = ArrayUtils.shift(src1NumRows, src2RowIndices.clone());
        System.arraycopy(src1RowIndices, 0, destRowIndices, 0, src1RowIndices.length);
        System.arraycopy(shiftedRowIndices, 0, destRowIndices, src1RowIndices.length, src2RowIndices.length);

        // Copy column indices.
        System.arraycopy(src1ColIndices, 0, destColIndices, 0, src1ColIndices.length);
        System.arraycopy(src2ColIndices, 0, destColIndices, src1ColIndices.length, src2ColIndices.length);
    }


    /**
     * Augments two matrices. This is equivalent to joining the matrices stacks matrices along rows.
     *
     * @param src1Entries Non-zero items of the first COO matrix to augment.
     * @param src1RowIndices Row indices of the first COO matrix to augment.
     * @param src1ColIndices Column indices of the first COO matrix to augment.
     * @param src1NumCols The number of columns in the first COO matrix. This is required to compute the shifted indices of the second
     * matrix.
     * @param src2Entries Non-zero items of the second COO matrix to augment.
     * @param src2RowIndices Row indices of the second COO matrix to augment.
     * @param src2ColIndices Column indices of the second COO matrix to augment.
     * @param destEntries Array to store the non-zero items resulting from the augmenting the two COO matrices.
     * @param destRowIndices Array to store the row indices resulting from the augmenting the two COO matrices.
     * @param destColIndices Array to store the column indices resulting from the augmenting the two COO matrices.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(Object[], int[], int[], int, Object[], int[], int[], Object[], int[], int[])
     * @see #augmentVector(Object[], int[], int[], int, Object[], int[], Object[], int[], int[])
     */
    public static <T> void augment(T[] src1Entries, int[] src1RowIndices, int[] src1ColIndices, int src1NumCols,
                                   T[] src2Entries, int[] src2RowIndices, int[] src2ColIndices,
                                   T[] destEntries, int[] destRowIndices, int[] destColIndices) {
        // Copy non-zero values.
        System.arraycopy(src1Entries, 0, destEntries, 0, src1Entries.length);
        System.arraycopy(src2Entries, 0, destEntries, src1Entries.length, src2Entries.length);

        // Copy row indices.
        System.arraycopy(src1RowIndices, 0, destRowIndices, 0, src1RowIndices.length);
        System.arraycopy(src2RowIndices, 0, destRowIndices, src1RowIndices.length, src2RowIndices.length);

        // Copy column indices (with shifts if appropriate).
        int[] shifted = src2ColIndices.clone();
        System.arraycopy(src1ColIndices, 0, destColIndices, 0, src1ColIndices.length);
        System.arraycopy(ArrayUtils.shift(src1NumCols, shifted), 0,
                destColIndices, src1ColIndices.length, src2ColIndices.length);

        CooDataSorter.wrap(destEntries, destRowIndices, destColIndices)
                .sparseSort()
                .unwrap(destEntries, destRowIndices, destColIndices);
    }


    /**
     * Augments two matrices. This is equivalent to joining the matrices stacks matrices along rows.
     *
     * @param src1Entries Non-zero items of the first COO matrix to augment.
     * @param src1RowIndices Row indices of the first COO matrix to augment.
     * @param src1ColIndices Column indices of the first COO matrix to augment.
     * @param src1NumCols The number of columns in the first COO matrix. This is required to compute the shifted indices of vector.
     * @param src2Entries Non-zero items of the COO vector to augment.
     * @param src2RowIndices Indices of the COO vector to augment.
     * @param destEntries Array to store the non-zero items resulting from the augmenting the two COO matrices.
     * @param destRowIndices Array to store the row indices resulting from the augmenting the two COO matrices.
     * @param destColIndices Array to store the column indices resulting from the augmenting the two COO matrices.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(Object[], int[], int[], int, Object[], int[], int[], Object[], int[], int[])
     * @see #augment(Object[], int[], int[], int, Object[], int[], int[], Object[], int[], int[])
     */
    public static <T> void augmentVector(
            T[] src1Entries, int[] src1RowIndices, int[] src1ColIndices, int src1NumCols,
            T[] src2Entries, int[] src2Indices,
            T[] destEntries, int[] destRowIndices, int[] destColIndices) {
        // Copy items and indices from this matrix.
        System.arraycopy(src1Entries, 0, destEntries, 0, src1Entries.length);
        System.arraycopy(src1RowIndices, 0, destRowIndices, 0, src1Entries.length);
        System.arraycopy(src1ColIndices, 0, destColIndices, 0, src1Entries.length);

        // Copy items and indices from vector.
        System.arraycopy(src2Entries, 0, destEntries, src1Entries.length, src2Entries.length);
        System.arraycopy(src2Indices, 0, destRowIndices, src1Entries.length, src2Entries.length);
        Arrays.fill(destColIndices, src1Entries.length, destColIndices.length, src1NumCols);

        // Ensure the values are properly sorted.
        CooDataSorter.wrap(destEntries, destRowIndices, destColIndices)
                .sparseSort()
                .unwrap(destEntries, destRowIndices, destColIndices);
    }


    /**
     * Joins two sparse COO vectors into one vector.
     * @param src1Entries Non-zero items of the first vector to join.
     * @param src1Indices Non-zero indices of the first vector to join.
     * @param src1Size Full size of the first vector.
     * @param src2Entries Non-zero items of the second vector to join.
     * @param src2Indices Non-zero indices of the second vector to join.
     * @param destEntries Array to store the resulting non-zero items of the vector join.
     * @param destIndices Array to store the resulting non-zero indices of the vector join.
     * @throws IndexOutOfBoundsException If {@code destEntries.length < src1Entries.length + src2Entries.length} or
     * {@code destIndices.length < src1Indices.length + src2Indices.length}
     */
    public static <T> void join(T[] src1Entries, int[] src1Indices, int src1Size,
                                T[] src2Entries, int[] src2Indices,
                                T[] destEntries, int[] destIndices) {
        // Copy values from this vector.
        System.arraycopy(src1Entries, 0, destEntries, 0, src1Entries.length);
        // Copy values from vector b.
        System.arraycopy(src2Entries, 0, destEntries, src1Entries.length, src2Entries.length);

        // Copy indices from this vector.
        System.arraycopy(src1Indices, 0, destIndices, 0, src1Indices.length);

        // Copy the indices from vector b with a shift.
        for(int i=0, size=src2Indices.length; i<size; i++) {
            destIndices[src1Indices.length+i] = src2Indices[i] + src1Size;
        }
    }


    /**
     * Repeats a sparse COO vector {@code n} times along a certain axis to create a matrix.
     * @param src Non-zero items of the vector.
     * @param srcIndices Non-zero indices of the vector.
     * @param size The full size of the vector.
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector. If {@code axis=0} then each row of the resulting matrix will be equivalent to
     * this vector. If {@code axis=1} then each column of the resulting matrix will be equivalent to this vector.
     * @param destEntries Array to store the non-zero items of the resulting matrix.
     * @param destRows Array to store the non-zero row indices of the resulting matrix.
     * @param destCols Array to store the non-zero column indices of the resulting matrix.
     * @return The shape of the resulting matrix.
     */
    public static <T> Shape repeat(T[] src, int[] srcIndices, int size,
                                   int n, int axis,
                                   T[] destEntries, int[] destRows, int[] destCols) {
        ValidateParameters.ensureInRange(axis, 0, 1, "axis");
        ValidateParameters.ensureGreaterEq(0, n, "n");

        Shape tiledShape;
        int nnz = src.length;

        if(axis==0) {
            tiledShape = new Shape(n, size);

            for(int i=0; i<n; i++) { // Copy values into row and set col indices as vector indices.
                System.arraycopy(src, 0, destEntries, i*nnz, nnz);
                System.arraycopy(srcIndices, 0, destCols, i*nnz, srcIndices.length);
                Arrays.fill(destRows, i*nnz, (i+1)*nnz, i);
            }
        } else {
            int[] colIndices = ArrayBuilder.intRange(0, n);
            tiledShape = new Shape(size, n);

            for(int i=0; i<nnz; i++) {
                Arrays.fill(destEntries, i*n, (i+1)*n, src[i]);
                Arrays.fill(destRows, i*n, (i+1)*n, srcIndices[i]);
                System.arraycopy(colIndices, 0, destCols, i*n, n);
            }
        }

        return tiledShape;
    }


    /**
     * Stacks two sparse COO vectors joining along columns as if they were row vectors. That is, constructs a sparse COO matrix
     * from the two vectors where the first row of the matrix is given by the fist vector in the stack operation and the second
     * row is given by the second vector.
     *
     * @param src1 Non-zero items of the fist COO vector to stack.
     * @param src1Indices Non-zero indices of the first COO vector to stack.
     * @param src2 Non-zero items of the second COO vector to stack.
     * @param src2Indices Non-zero indices of the second COO vector to stack.
     * @param destEntries Array to store the non
     */
    public static <T> void stack(
            T[] src1, int[] src1Indices,
            T[] src2, int[] src2Indices,
            T[] destEntries, int[] rowIndices, int[] colIndices) {
        // Copy values from vector src1.
        System.arraycopy(src1, 0, destEntries, 0, src1.length);
        // Copy values from vector src2.
        System.arraycopy(src2, 0, destEntries, src1.length, src2.length);

        // Set row indices to 1 for src2 values (this vectors row indices are 0 which was implicitly set already).
        Arrays.fill(rowIndices, src1Indices.length, destEntries.length, 1);

        // Copy indices from src1 vector to the column indices.
        System.arraycopy(src1Indices, 0, colIndices, 0, src1.length);
        // Copy indices from src2 vector to the column indices.
        System.arraycopy(src2Indices, 0, colIndices, src1.length, src2.length);
    }
}
