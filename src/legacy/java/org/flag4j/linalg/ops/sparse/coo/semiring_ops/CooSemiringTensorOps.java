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
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.*;

/**
 * This utility class contains methods for computing ops on two {@link SemiringElement}
 * tensors.
 */
public final class CooSemiringTensorOps {

    private CooSemiringTensorOps() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Sums two complex sparse COO tensors and stores result in a new COO tensor.
     * @param shape1 Shape of the first tensor.
     * @param src1Entries Non-zero items of the first tensor.
     * @param src1Indices Non-zero indices of the first tensor.
     * @param shape2 Shape of the second tensor.
     * @param src2Entries Non-zero items of the second tensor.
     * @param src2Indices Non-zero indices of the second tensor.
     * @return The element-wise tensor sum of the two tensors.
     * @throws LinearAlgebraException If the tensors {@code src1} and {@code src2} do not have the same shape.
     */
    public static <V extends SemiringElement<V>> SparseTensorData<V> add(
            Shape shape1, V[] src1Entries, int[][] src1Indices,
            Shape shape2, V[] src2Entries, int[][] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        final int src1Nnz = src1Entries.length;
        final int src2Nnz = src2Entries.length;

        // Roughly estimate the number of non-zero items in sum.
        int estimatedEntries = src1Nnz + src2Nnz;
        List<V> sumEntries = new ArrayList<>(estimatedEntries);
        List<int[]> sumIndices = new ArrayList<>(estimatedEntries);

        int src2Pos = 0;
        for(int i = 0; i < src1Nnz; i++) {
            V val1 = src1Entries[i];
            int[] src1Idx = src1Indices[i].clone();

            // Insert elements from src2 whose index is less than the current element from src1
            while(src2Pos < src2Nnz && Arrays.compare(src2Indices[src2Pos], src1Idx) < 0) {
                sumEntries.add(src2Entries[src2Pos]);
                sumIndices.add(src2Indices[src2Pos++].clone());
            }

            // Add the current element from src1 and handle matching indices from src2
            if(src2Pos < src2Nnz && Arrays.equals(src1Idx, src2Indices[src2Pos])) {
                sumEntries.add(val1.add(src2Entries[src2Pos++]));
            } else {
                sumEntries.add(val1);
            }
            sumIndices.add(src1Idx);
        }

        // Insert any remaining elements from src2
        while(src2Pos < src2Nnz) {
            sumEntries.add(src2Entries[src2Pos]);
            sumIndices.add(src2Indices[src2Pos++]);
        }

        return new SparseTensorData<V>(shape1, sumEntries, sumIndices);
    }


    /**
     * <p>Computes the element-wise multiplication between two complex sparse COO tensors.
     *
     * <p>Assumes indices of both tensors are sorted lexicographically by their indices.
     * @param shape1 Shape of the first tensor.
     * @param src1Entries Non-zero items of the first tensor.
     * @param src1Indices Non-zero indices of the first tensor.
     * @param shape2 Shape of the second tensor.
     * @param src2Entries Non-zero items of the second tensor.
     * @param src2Indices Non-zero indices of the second tensor.
     * @return The element-wise product of the two specified tensors.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     */
    public static <V extends SemiringElement<V>> SparseTensorData<V> elemMult(
            Shape shape1, V[] src1Entries, int[][] src1Indices,
            Shape shape2, V[] src2Entries, int[][] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        int src1Nnz = src1Entries.length;
        int src2Nnz = src2Entries.length;

        // Swap src1 and src2 if src2 has fewer non-zero items for possibly better performance.
        if (src2Nnz < src1Nnz) {
            Shape tempShape = shape1;
            shape1 = shape2;
            shape2 = tempShape;

            V[] tempEntries = src1Entries;
            src1Entries = src2Entries;
            src2Entries = tempEntries;

            int[][] tempIndices = src1Indices;
            src1Indices = src2Indices;
            src2Indices = tempIndices;
        }

        List<V> productList = new ArrayList<>(Math.min(src1Nnz, src2Nnz));
        List<int[]> productIndicesList = new ArrayList<>(Math.min(src1Nnz, src2Nnz));

        int src2Idx = 0;
        for(int i = 0; i < src1Nnz && src2Idx < src2Nnz; i++) {
            int cmp = -1;

            while(src2Idx < src2Nnz && (cmp = Arrays.compare(src2Indices[src2Idx], src1Indices[i])) < 0)
                src2Idx++;

            if(src2Idx < src2Nnz && cmp == 0) {
                productList.add(src1Entries[i].mult(src2Entries[src2Idx]));
                productIndicesList.add(src1Indices[i]);
            }
        }

        return new SparseTensorData<V>(
                shape1, productList, productIndicesList);
    }


    /**
     * <p>Computes the generalized trace of a tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D subarrays of the {@code src} tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to the {@code src} tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param shape1 Shape of the tensor.
     * @param src1Entries Non-zero items of the tensor.
     * @param src1Indices Non-zero indices of the tensor.
     * @param axis1 First axis for 2D subarray.
     * @param axis2 Second axis for 2D subarray.
     *
     * @return The generalized trace of the {@code src} tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than the {@code src} tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code src.shape.get(axis1) != src.shape.get(axis1)}
     *                                   (i.e., the axes are equal or the tensor does not have the same length along the two axes.)
     */
    public static <T extends SemiringElement<T>> SparseTensorData<T> tensorTr(
            Shape shape, T[] entries, int[][] indices,
            int axis1, int axis2) {
        // Validate parameters.
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.validateArrayIndices(shape.getRank(), axis1, axis2);
        ValidateParameters.ensureAllEqual(shape.getSize(axis1), shape.getSize(axis2));

        int rank = shape.getRank();
        final int nnz = entries.length;
        int[] dims = shape.getDims();

        // Determine the shape of the resulting tensor.
        int[] traceShape = new int[rank - 2];
        int newShapeIndex = 0;
        for (int i = 0; i < rank; i++) {
            if (i != axis1 && i != axis2)
                traceShape[newShapeIndex++] = dims[i];
        }

        // Lists to accumulate the non-zero items and their indices.
        List<int[]> resultIndicesList = new ArrayList<>();
        List<T> resultEntriesList = new ArrayList<>();

        // Map to keep track of linear indices and their positions in the lists.
        Map<Integer, Integer> indexMap = new HashMap<>();

        // Iterate through the non-zero items and accumulate trace for those on the diagonal.
        for (int i = 0; i < nnz; i++) {
            int[] idxs = indices[i];
            T value = entries[i];

            // Check if the current entry is on the diagonal.
            if (idxs[axis1] == idxs[axis2]) {
                // Compute a linear index for the resulting tensor by ignoring axis1 and axis2.
                int linearIndex = 0;
                int stride = 1;

                for (int j = rank - 1; j >= 0; j--) {
                    if (j != axis1 && j != axis2) {
                        linearIndex += idxs[j] * stride;
                        stride *= dims[j];
                    }
                }

                // Check if we've already seen this linearIndex.
                if (indexMap.containsKey(linearIndex)) {
                    int position = indexMap.get(linearIndex);
                    // Accumulate the value
                    resultEntriesList.set(position, resultEntriesList.get(position).add((T) value));
                } else {
                    // Build the indices for the resulting tensor by excluding axis1 and axis2.
                    int[] resultIndicesEntry = new int[rank - 2];
                    int idx = 0;
                    for (int j = 0; j < rank; j++) {
                        if (j != axis1 && j != axis2) {
                            resultIndicesEntry[idx++] = idxs[j];
                        }
                    }

                    // Add new index and value to the lists
                    resultIndicesList.add(resultIndicesEntry);
                    resultEntriesList.add(value);
                    indexMap.put(linearIndex, resultEntriesList.size() - 1);
                }
            }
        }

        return new SparseTensorData<T>(new Shape(traceShape), resultEntriesList, resultIndicesList);
    }
}
