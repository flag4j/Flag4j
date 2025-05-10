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
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

/**
 * Utility class for computing the transpose of a sparse COO tensor or matrix.
 */
public final class CooTranspose {

    private CooTranspose() {
        // Hide default constructor for utility class.
    }


    /**
     * <p>Computes the transpose of a sparse COO tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param shape Shape of the tensor to transpose.
     * @param srcEntries The non-zero data of the tensor to transpose.
     * @param srcIndices The non-zero indices of the tensor to transpose. Must have shape
     * {@code [srcEntries.length][shape.getRank()]}.
     * @param srcIndices
     * @param axis1 First axis to exchange. Must be in the range [0, shape.getRank()).
     * @param axis2 Second axis to exchange. Must be in the range [0, shape.getRank()).
     * @param destEntries Array to store the non-zero data of the transpose. Must have the same length as {@code srcEntries}.
     * @param destIndices Array to store the non-zero indices of the transpose. Must have shape
     * {@code [srcEntries.length][shape.getRank()]}
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #tensorTranspose(Shape, Object[], int[][], int[], Object[], int[][])
     */
    public static void tensorTranspose(Shape shape, Object[] srcEntries, int[][] srcIndices,
                                int axis1, int axis2,
                                Object[] destEntries, int[][] destIndices) {
        int rank = shape.getRank();
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        ValidateParameters.ensureArrayLengthsEq(srcEntries.length, destEntries.length);

        if(axis1 == axis2)
            System.arraycopy(srcEntries, 0, destEntries, 0, srcEntries.length);

        for(int i=0, nnz=srcEntries.length; i<nnz; i++) {
            destEntries[i] = srcEntries[i];
            destIndices[i] = srcIndices[i].clone();
            ArrayUtils.swap(destIndices[i], axis1, axis2);
        }

        // Ensure the values are sorted lexicographically.
        CooDataSorter.wrap(destEntries, destIndices).sparseSort().unwrap(destEntries, destIndices);
    }


    /**
     * Computes the transpose of a sparse COO tensor. That is, permutes the axes of the tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param shape Shape of the tensor to transpose.
     * @param srcEntries The non-zero data of the tensor to transpose.
     * @param srcIndices The non-zero indices of the tensor to transpose. Must have shape
     * {@code [srcEntries.length][shape.getRank()]}.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @param destEntries Array to store the non-zero data of the transpose. Must have the same length as {@code srcEntries}.
     * @param destIndices Array to store the non-zero indices of the transpose. Must have shape
     * {@code [srcEntries.length][shape.getRank()]}
     *
     * @throws IllegalArgumentException If {@code srcEntries}, {@code srcIndices}, {@code destEntries}, and {@code destIndices} do not
     * all have the same length.
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     */
    public static void tensorTranspose(Shape shape, Object[] srcEntries, int[][] srcIndices, int[] axes,
                                       Object[] destEntries, int[][] destIndices) {
        int rank = shape.getRank();
        ValidateParameters.ensurePermutation(axes, rank);

        // Permute the indices according to the permutation array.
        for(int i=0, nnz=srcEntries.length; i < nnz; i++) {
            destEntries[i] = srcEntries[i];
            destIndices[i] = srcIndices[i].clone();

            for(int j = 0; j < rank; j++)
                destIndices[i][j] = srcIndices[i][axes[j]];
        }

        // Ensure the values are sorted lexicographically.
        CooDataSorter.wrap(destEntries, destIndices).sparseSort().unwrap(destEntries, destIndices);
    }
}
