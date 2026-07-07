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

package org.flag4j.linalg.ops.sparse.coo.real;


import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class for computing ops between two real sparse COO tensors backed by a primitive double array.
 */
public final class RealCooTensorOps {

    private RealCooTensorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Sums two sparse COO tensors and stores result in a new COO tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The element-wise tensor sum of {@code src1} and {@code src2}.
     * @throws LinearAlgebraException If the tensors {@code src1} and {@code src2} do not have the same shape.
     */
    public static CooTensor add(CooTensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        // Create deep copies of indices.
        int[][] src1Indices = ArrayUtils.deepCopy2D(src1.indices, null);
        int[][] src2Indices = ArrayUtils.deepCopy2D(src2.indices, null);

        // Roughly estimate the number of non-zero items in sum.
        int estimatedEntries = src1.nnz + src2.nnz;
        List<Double> sumEntries = new ArrayList<>(estimatedEntries);
        List<int[]> sumIndices = new ArrayList<>(estimatedEntries);

        int src2Pos = 0;
        for(int i = 0; i < src1.nnz; i++) {
            double val1 = src1.data[i];
            int[] src1Idx = src1Indices[i];

            // Insert elements from src2 whose index is less than the current element from src1.
            while(src2Pos < src2.nnz && Arrays.compare(src2Indices[src2Pos], src1Idx) < 0) {
                sumEntries.add(src2.data[src2Pos]);
                sumIndices.add(src2Indices[src2Pos++]);
            }

            // Add the current element from src1 and handle matching indices from src2.
            if(src2Pos < src2.nnz && Arrays.equals(src1Idx, src2Indices[src2Pos])) {
                sumEntries.add(val1 + src2.data[src2Pos++]);
            } else {
                sumEntries.add(val1);
            }

            sumIndices.add(src1Idx);
        }

        // Insert any remaining elements from src2.
        while(src2Pos < src2.nnz) {
            sumEntries.add(src2.data[src2Pos]);
            sumIndices.add(src2Indices[src2Pos++]);
        }

        return src1.makeLikeTensor(src1.shape, sumEntries, sumIndices);
    }


    /**
     * Computes difference between two sparse COO tensors and stores result in a new COO tensor.
     * @param src1 First tensor in the difference.
     * @param src2 Second tensor in the difference.
     * @return The element-wise tensor difference of {@code src1} and {@code src2}.
     * @throws LinearAlgebraException If the tensors {@code src1} and {@code src2} do not have the same shape.
     */
    public static CooTensor sub(CooTensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        // Create deep copies of indices.
        int[][] src1Indices = ArrayUtils.deepCopy2D(src1.indices, null);
        int[][] src2Indices = ArrayUtils.deepCopy2D(src2.indices, null);

        // Roughly estimate the number of non-zero items in sum.
        int estimatedEntries = src1.nnz + src2.nnz;
        List<Double> sumEntries = new ArrayList<>(estimatedEntries);
        List<int[]> sumIndices = new ArrayList<>(estimatedEntries);

        int src2Pos = 0;
        for(int i = 0; i < src1.nnz; i++) {
            double val1 = src1.data[i];
            int[] src1Idx = src1Indices[i];

            // Insert elements from src2 whose index is less than the current element from src1.
            while(src2Pos < src2.nnz && Arrays.compare(src2Indices[src2Pos], src1Idx) < 0) {
                sumEntries.add(-src2.data[src2Pos]);
                sumIndices.add(src2Indices[src2Pos++]);
            }

            // Add the current element from src1 and handle matching indices from src2.
            if(src2Pos < src2.nnz && Arrays.equals(src1Idx, src2Indices[src2Pos])) {
                sumEntries.add(val1 - src2.data[src2Pos++]);
            } else {
                sumEntries.add(val1);
            }
            sumIndices.add(src1Idx);
        }

        // Insert any remaining elements from src2.
        while(src2Pos < src2.nnz) {
            sumEntries.add(-src2.data[src2Pos]);
            sumIndices.add(src2Indices[src2Pos++]);
        }

        return src1.makeLikeTensor(src1.shape, sumEntries, sumIndices);
    }


    /**
     * <p>Computes the element-wise multiplication between two sparse COO tensors.
     *
     * <p>Assumes indices of both tensors are sorted lexicographically.
     *
     * @param src1 First tensor in the element-wise multiplication.
     * @param src2 Second tensor in the element-wise multiplication.
     * @return The element-wise product of {@code src1} and {@code src2}.
     */
    public static CooTensor elemMult(CooTensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        // Swap src1 and src2 if src2 has fewer non-zero items for possibly better performance.
        if (src2.nnz < src1.nnz) {
            CooTensor temp = src1;
            src1 = src2;
            src2 = temp;
        }

        double[] productEntries = new double[Math.min(src1.nnz, src2.nnz)];
        int[][] productIndices = new int[Math.min(src1.nnz, src2.nnz)][src1.indices[0].length];
        int count = 0;

        int src2Idx = 0;
        for(int i = 0; i < src1.nnz && src2Idx < src2.nnz; i++) {
            int cmp = -1;

            while(src2Idx < src2.nnz && (cmp = Arrays.compare(src2.indices[src2Idx], src1.indices[i])) < 0) {
                src2Idx++;
            }

            if(src2Idx < src2.nnz && cmp == 0) {
                productEntries[count] = src1.data[i] * src2.data[src2Idx];
                productIndices[count] = src1.indices[i];
                count++;
            }
        }

        // Truncate ndarrays if necessary.
        return src1.makeLikeTensor(src1.shape, Arrays.copyOf(productEntries, count), Arrays.copyOf(productIndices, count));
    }
}
