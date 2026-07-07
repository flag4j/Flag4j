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
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringMatrix;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for computing various ops on and between sparse COO
 * {@link SemiringElement} matrices.
 */
public final class CooSemiringMatrixOps {

    private CooSemiringMatrixOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Adds two sparse matrices. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param shape1 Shape of the first matrix.
     * @param src1Entries Non-zero items of the first matrix.
     * @param src1RowIndices Non-zero row indices of the first matrix.
     * @param src1ColIndices Non-zero column indices of the first matrix.
     * @param shape2 Shape of the second matrix.
     * @param src2Entries Non-zero items of the second matrix.
     * @param src2RowIndices Non-zero row indices of the second matrix.
     * @param src2ColIndices Non-zero column indices of the second matrix.
     * @return The sum of the two matrices.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends SemiringElement<V>> SparseMatrixData<V> add(
            Shape shape1, V[] src1Entries, int[] src1RowIndices, int[] src1ColIndices,
            Shape shape2, V[] src2Entries, int[] src2RowIndices, int[] src2ColIndices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        int initCapacity = Math.max(src1Entries.length, src2Entries.length);

        List<V> sum = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1Entries.length || src2Counter < src2Entries.length) {

            if(src1Counter >= src1Entries.length || src2Counter >= src2Entries.length) {
                add1 = src2Counter >= src2Entries.length;
                add2 = !add1;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]
                    && src1ColIndices[src1Counter] == src2ColIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1ColIndices[src1Counter] < src2ColIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1RowIndices[src1Counter] < src2RowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                sum.add(src1Entries[src1Counter].add(src2Entries[src2Counter]));
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                sum.add(src1Entries[src1Counter]);
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
            } else {
                sum.add(src2Entries[src2Counter]);
                rowIndices.add(src2RowIndices[src2Counter]);
                colIndices.add(src2ColIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new SparseMatrixData<V>(shape1, sum, rowIndices, colIndices);
    }


    /**
     * Multiplies two sparse matrices element-wise. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param shape1 Shape of the first matrix.
     * @param src1Entries Non-zero items of the first matrix.
     * @param src1RowIndices Non-zero row indices of the first matrix.
     * @param src1ColIndices Non-zero column indices of the first matrix.
     * @param shape2 Shape of the second matrix.
     * @param src2Entries Non-zero items of the second matrix.
     * @param src2RowIndices Non-zero row indices of the second matrix.
     * @param src2ColIndices Non-zero column indices of the second matrix.
     * @return The element-wise product of the two matrices.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends SemiringElement<V>> SparseMatrixData<V> elemMult(
            Shape shape1, V[] src1Entries, int[] src1RowIndices, int[] src1ColIndices,
            Shape shape2, V[] src2Entries, int[] src2RowIndices, int[] src2ColIndices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        int initCapacity = Math.max(src1Entries.length, src2Entries.length);

        List<V> product = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1Entries.length && src2Counter < src2Entries.length) {
            if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]
                    && src1ColIndices[src1Counter] == src2ColIndices[src2Counter]) {
                product.add(src1Entries[src1Counter].mult(src2Entries[src2Counter]));
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]) {
                // Matching row indices.
                if(src1ColIndices[src1Counter] < src2ColIndices[src2Counter])
                    src1Counter++;
                else
                    src2Counter++;

            } else {
                if(src1RowIndices[src1Counter] < src2RowIndices[src2Counter])
                    src1Counter++;
                else
                    src2Counter++;
            }
        }

        return new SparseMatrixData<V>(shape1, product, rowIndices, colIndices);
    }


    /**
     * Checks if a complex sparse matrix is the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return {@code true} if the {@code src} matrix is the identity matrix; {@code false} otherwise.
     */
    public static <T extends SemiringElement<T>> boolean isIdentity(AbstractCooSemiringMatrix<?, ?, ?, T> src) {
        // Ensure the matrix is square and there are at least the same number of non-zero items as items on the diagonal.
        if(!src.isSquare() || src.data.length<src.numRows) return false;

        for(int i = 0, size = src.data.length; i<size; i++) {
            // Ensure value is 1 and on the diagonal.
            if(src.rowIndices[i] != i && src.colIndices[i] != i && !src.data[i].isOne()) {
                return false;
            } else if((src.rowIndices[i] != i || src.colIndices[i] != i) && !src.data[i].isZero()) {
                return false;
            }
        }

        return true; // If we make it to this point the matrix must be an identity matrix.
    }
}
