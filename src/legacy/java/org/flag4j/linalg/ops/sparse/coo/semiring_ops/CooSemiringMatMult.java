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
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.numbers.SemiringElement;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * This utility class contains methods useful for computing the matrix multiplication between two sparse COO
 * {@link SemiringElement} tensors.
 */
public final class CooSemiringMatMult {

    private CooSemiringMatMult() {
        
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero items for the second sparse matrix.
     * @param colIndices2 column indices of non-zero items for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication in.
     * Must have length {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends SemiringElement<T>> void standard(
            T[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            T[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, src1[0].getZero());

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        for(int i=0; i<src1.length; i++) {
            int c1 = colIndices1[i]; // = k
            T value = src1[i];

            // Check if any values in src2 have the same row index as the column index of the value in src1.
            if(map.containsKey(c1)) {
                int r1 = rowIndices1[i]; // = i
                int rowIdx = r1*cols2;

                for(int j : map.get(c1)) { // Iterate over all items in src2 where rowData[j] == colData[j]
                    int c2 = colIndices2[j]; // = j
                    dest[rowIdx + c2] = dest[rowIdx + c2].add(value.mult(src2[j]));
                }
            }
        }
    }


    /**
     * <p>Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.
     *
     * <p>NOTE: Caution should be exercised when using this method.
     * It is rarely faster than {@link #standard(SemiringElement[], int[], int[], Shape, SemiringElement[], int[], int[], Shape, SemiringElement[])}
     *
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero items for the second sparse matrix.
     * @param colIndices2 column indices of non-zero items for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication in.
     * Must have length {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends SemiringElement<T>> void concurrentStandard(
            T[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            T[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        final T ZERO = src1[0].getZero();
        Arrays.fill(dest, ZERO);
        ConcurrentMap<Integer, T> destMap = new ConcurrentHashMap<>();

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int c1 = colIndices1[i]; // = k

                // Check if any values in src2 have the same row index as the column index of the value in src1.
                if(map.containsKey(c1)) {
                    int r1 = rowIndices1[i]; // = i
                    int rowIdx = r1*cols2;
                    T value = src1[i];

                    for(int j : map.get(c1)) { // Iterate over all items in src2 where rowData[j] == colData[j]
                        int idx = rowIdx + colIndices2[j];
                        destMap.put(idx, destMap.getOrDefault(idx, ZERO).add(value.mult(src2[j])));
                    }
                }
            }
        });

        // Copy values from map to destination array.
        for(int idx : destMap.keySet())
            dest[idx] = destMap.get(idx);
    }


    /**
     * <p>Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     *
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param dest Array to store the resulting dense vector in.
     */
    public static <T extends SemiringElement<T>> void standardVector(
            T[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            T[] src2, int[] indices, T[] dest) {

        final T ZERO = src1[0].getZero();
        int rows1 = shape1.getSize(0);
        Arrays.fill(dest, ZERO);

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0, size1=src1.length; i<size1; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k
            T sum = ZERO;

            for(int j=0, size2=src2.length; j<size2; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) // Then multiply and add to sum.
                    sum = sum.add(src1[i].mult(src2[j]));
            }

            dest[r1] = sum;
        }
    }


    /**
     * <p>Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.
     *
     * <p>NOTE: Caution should be exercised when using this method.
     * It is rarely faster than {@link #standardVector(SemiringElement[], int[], int[], Shape, SemiringElement[], int[], SemiringElement[])}
     * 
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param dest Array to store the dense result of the matrix multiplication in.
     * Must have length {@code shape1.get(0)}.
     */
    public static <T extends SemiringElement<T>> void concurrentStandardVector(
            T[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            T[] src2, int[] indices, T[] dest) {
        int rows1 = shape1.getSize(0);
        Arrays.fill(dest, src1[0].getZero());

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices1[i]; // = i
                int c1 = colIndices1[i]; // = k

                for(int j=0; j<src2.length; j++) {
                    int r2 = indices[j]; // = k

                    if(c1==r2) { // Then we multiply and add to sum.
                        T product = src1[i].mult(src2[j]);

                        synchronized (dest) {
                            dest[r1] = dest[r1].add(product);
                        }
                    }
                }
            }
        });
    }
}
