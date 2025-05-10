/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.linalg.ops.sparse.SparseUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;


/**
 * This class contains low-level implementations of matrix multiplication for real sparse matrices.
 * <b>WARNING:</b> This class does not provide sanity checks.
 */
public final class RealSparseMatMult {

    private RealSparseMatMult() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a standard algorithm.
     * @param src1 Non-zero data of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero data for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero data for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero data of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero data for the second sparse matrix.
     * @param colIndices2 column indices of non-zero data for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static double[] standard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                    double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        for(int i=0; i<src1.length; i++) {
            int c1 = colIndices1[i]; // = k
            double src1Val = src1[i];

            // Check if any values in src2 have the same row index as the column index of the value in src1.
            if(map.containsKey(c1)) {
                int r1 = rowIndices1[i]; // = i
                int rowIdx = r1*cols2;

                for(int j : map.get(c1)) { // Iterate over all data in src2 where rowData[j] == colData[j]
                    int c2 = colIndices2[j]; // = j
                    dest[rowIdx + c2] += src1Val*src2[j];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm. <br><br>
     *
     * NOTE: Caution should be exercised when using this method. It is rarely faster than
     * {@link #standard(double[], int[], int[], Shape, double[], int[], int[], Shape)}
     * @param src1 Non-zero data of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero data for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero data for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero data of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero data for the second sparse matrix.
     * @param colIndices2 column indices of non-zero data for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static double[] concurrentStandard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                    double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];
        ConcurrentMap<Integer, Double> destMap = new ConcurrentHashMap<>();

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int c1 = colIndices1[i]; // = k
                double src1Val = src1[i];

                // Check if any values in src2 have the same row index as the column index of the value in src1.
                if(map.containsKey(c1)) {
                    int r1 = rowIndices1[i]; // = i
                    int rowIdx = r1*cols2;

                    for(int j : map.get(c1)) { // Iterate over all data in src2 where rowData[j] == colData[j]
                        int idx = rowIdx + colIndices2[j];
                        destMap.put(idx, destMap.getOrDefault(idx, 0d) + src1Val*src2[j]);
                    }
                }
            }
        });

        // Copy values from map to destination array.
        for(int idx : destMap.keySet()) {
            dest[idx] = destMap.get(idx);
        }

        return dest;
    }

    // ----------------------------------- Matrix-Vector Multiplication -----------------------------------


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     *
     * @param src1        Non-zero data of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero data for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero data for the first sparse matrix.
     * @param shape1      Shape of the first sparse matrix.
     * @param src2        Non-zero data of the second sparse matrix.
     * @param indices     Indices of non-zero data in the sparse vector.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static double[] standardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                    double[] src2, int[] indices) {
        int rows1 = shape1.getSize(0);
        double[] dest = new double[rows1];

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1] += src1[i]*src2[j];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.
     * @param src1 Non-zero data of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero data for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero data for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero data of the second sparse matrix.
     * @param indices Indices of non-zero data in the sparse vector.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static double[] concurrentStandardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                    double[] src2, int[] indices) {
        int rows1 = shape1.getSize(0);
        double[] dest = new double[rows1];

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices1[i]; // = i
                int c1 = colIndices1[i]; // = k
                double src1Val = src1[i];
                double sum = dest[r1];

                for(int j=0; j<src2.length; j++) {
                    int r2 = indices[j]; // = k

                    if(c1==r2) { // Then we multiply and add to sum.
                        double product = src1Val*src2[j];
                        sum += product;
                    }
                }

                dest[r1] = sum;
            }
        });

        return dest;
    }
}
