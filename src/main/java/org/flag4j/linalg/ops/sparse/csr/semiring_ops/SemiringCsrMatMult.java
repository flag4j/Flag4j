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

package org.flag4j.linalg.ops.sparse.csr.semiring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ValidateParameters;

import java.util.*;

/**
 * Utility class for computing matrix multiplication of a sparse CSR {@link Semiring} matrix.
 */
public final class SemiringCsrMatMult {

    private SemiringCsrMatMult() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param shape1 Shape of the first matrix in the matrix multiplication.
     * @param src1Entries Non-zero data of the first CSR matrix in the matrix multiplication.
     * @param src1RowPointers Non-zero row pointers of the first CSR matrix in the matrix multiplication.
     * @param src1ColIndices Non-zero column indices of the first CSR matrix in the matrix multiplication.
     * @param shape2 Shape of the second matrix in the matrix multiplication.
     * @param src2Entries Non-zero data of the second CSR matrix in the matrix multiplication.
     * @param src2RowPointers Non-zero row pointers of the second CSR matrix in the matrix multiplication.
     * @param src2ColIndices Non-zero column indices of the second CSR matrix in the matrix multiplication.
     * @param destEntries Array to store the dense result of the matrix multiplication in (modified).
     * @param zero The zero value of the semiring.
     */
    public static <T extends Semiring<T>> void standard(
            Shape shape1, T[] src1Entries, int[] src1RowPointers, int[] src1ColIndices,
            Shape shape2, T[] src2Entries, int[] src2RowPointers, int[] src2ColIndices,
            T[] destEntries, Semiring<T> zero) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        zero = (zero == null && src1Entries.length > 0) ?  src1Entries[0].getZero() : zero;
        int src1Rows = shape1.get(0);
        int src2Cols = shape2.get(1);
        Arrays.fill(destEntries, zero);

        for(int i=0; i<src1Rows; i++) {
            int rowOffset = i*src2Cols;
            int stop = src1RowPointers[i+1];

            for(int aIndex=src1RowPointers[i]; aIndex<stop; aIndex++) {
                int aCol = src1ColIndices[aIndex];
                T aVal = src1Entries[aIndex];
                int innerStop = src2RowPointers[aCol+1];

                for(int bIndex=src2RowPointers[aCol]; bIndex<innerStop; bIndex++) {
                    int bCol = src2ColIndices[bIndex];
                    T bVal = src2Entries[bIndex];

                    destEntries[rowOffset + bCol] = destEntries[rowOffset + bCol].add(bVal.mult(aVal));
                }
            }
        }
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix.
     *
     * <p>Warning: This method may be slower than
     * {@link #standard(Shape, Semiring[], int[], int[], Shape, Semiring[], int[], int[], Semiring[], Semiring)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     *
     * @param shape1 Shape of the first matrix in the matrix multiplication.
     * @param src1Entries Non-zero data of the first CSR matrix in the matrix multiplication.
     * @param src1RowPointers Non-zero row pointers of the first CSR matrix in the matrix multiplication.
     * @param src1ColIndices Non-zero column indices of the first CSR matrix in the matrix multiplication.
     * @param shape2 Shape of the second matrix in the matrix multiplication.
     * @param src2Entries Non-zero data of the second CSR matrix in the matrix multiplication.
     * @param src2RowPointers Non-zero row pointers of the second CSR matrix in the matrix multiplication.
     * @param src2ColIndices Non-zero column indices of the second CSR matrix in the matrix multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static <T extends Semiring<T>> SparseMatrixData<T> standardToSparse(
            Shape shape1, T[] src1Entries, int[] src1RowPointers, int[] src1ColIndices,
            Shape shape2, T[] src2Entries, int[] src2RowPointers, int[] src2ColIndices) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        Shape destShape = new Shape(rows1, cols2);

        int[] resultRowPtr = new int[rows1 + 1];
        List<T> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<rows1; i++) {
            Map<Integer, T> tempMap = new HashMap<>();
            int start = src1RowPointers[i];
            int stop = src1RowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1ColIndices[aIndex];
                T aVal = (T) src1Entries[aIndex];
                int innerStart = src2RowPointers[aCol];
                int innerStop = src2RowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2ColIndices[bIndex];
                    T bVal = src2Entries[bIndex];

                    tempMap.merge(bCol, bVal.mult(aVal), T::add);
                }
            }

            // Ensure data within each row are sorted by the column indices.
            List<Integer> tempColIndices = new ArrayList<>(tempMap.keySet());
            Collections.sort(tempColIndices);

            for (int colIndex : tempColIndices) {
                resultColIndexList.add(colIndex);
                resultList.add(tempMap.get(colIndex));
            }

            resultRowPtr[i + 1] = resultList.size();
        }

        return new SparseMatrixData<T>(destShape, resultList,
                ArrayConversions.toArrayList(resultRowPtr), resultColIndexList);
    }


    /**
     * Computes the matrix-vector multiplication between a sparse CSR matrix and a sparse COO vector.
     * @param shape Shape of the CSR matrix.
     * @param src1 Non-zero data of the CSR matrix
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param size Full size of the COO vector.
     * @param src2 Non-zero data of the COO vector.
     * @param indices Non-zero indices of the COO Vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication in.
     * @param zero Zero element of the semiring. If {@code null} then the zero value will attempt to be discerned from {@code src1}.
     * However, if {@code zero == null && src1.length == 0}, then zeros in the resulting dense vector will instead be {@code null}.
     *
     * @throws IllegalArgumentException If {@code shape.get(1) != size} or {@code dest.length != size}.
     */
    public static <T extends Semiring<T>> void standardVector(
            Shape shape, T[] src1, int[] rowPointers, int[] colIndices,
            int size, T[] src2, int[] indices,
            T[] dest, T zero) {
        // Ensure the matrix and vector have shapes conducive to matrix-vector multiplication.
        int rows1 = shape.get(0);
        ValidateParameters.ensureAllEqual(shape.get(1), size);
        ValidateParameters.ensureAllEqual(dest.length, size);
        zero = (zero == null && src1.length > 0) ? src1[0].getZero() : zero;
        Arrays.fill(dest, zero);

        // Iterate over the non-zero-elements of the sparse vector.
        for (int k=0, src2Nnz = src2.length; k < src2Nnz; k++) {
            int col = indices[k];
            T val = src2[k];

            // Perform multiplication only for the non-zero-elements.
            for (int i=0; i<rows1; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i + 1];
                T sum = dest[i];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = colIndices[aIndex];

                    if (aCol == col)
                        sum = sum.add(val.mult(src1[aIndex]));
                }

                dest[i] = sum;
            }
        }
    }
}
