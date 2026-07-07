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

package org.flag4j.linalg.ops.sparse.coo.ring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class for computing ops on sparse COO {@link RingElement} tensors.
 */
public final class CooRingTensorOps {

    private CooRingTensorOps() {
        // Hide default constructor for utility class.
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
    public static <V extends RingElement<V>> SparseTensorData<V> sub(
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
                sumEntries.add(src2Entries[src2Pos].negate());
                sumIndices.add(src2Indices[src2Pos++].clone());
            }

            // Add the current element from src1 and handle matching indices from src2
            if(src2Pos < src2Nnz && Arrays.equals(src1Idx, src2Indices[src2Pos]))
                sumEntries.add(val1.sub(src2Entries[src2Pos++]));
            else
                sumEntries.add(val1);

            sumIndices.add(src1Idx);
        }

        // Insert any remaining elements from src2.
        while(src2Pos < src2Nnz) {
            sumEntries.add(src2Entries[src2Pos].negate());
            sumIndices.add(src2Indices[src2Pos++]);
        }

        return new SparseTensorData<V>(shape1, sumEntries, sumIndices);
    }
}
