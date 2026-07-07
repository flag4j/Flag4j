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

package org.flag4j.linalg.ops.sparse.coo.real_complex;

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.numbers.Complex128;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;


/**
 * This class contains low level methods for computing the multiplication between a real/complex matrix and a complex/real
 * matrix/vector. <br>
 * <b>WARNING:</b> The methods in this class do not provide sanity checks.
 */
public final class RealComplexSparseMatrixMultiplication {

    private RealComplexSparseMatrixMultiplication() {
        // Hide default constructor for utility class.
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
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static Complex128[] standard(Complex128[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                        double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Complex128[] dest = new Complex128[rows1*cols2];
        Arrays.fill(dest, Complex128.ZERO);

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        for(int i=0; i<src1.length; i++) {
            int c1 = colIndices1[i]; // = k
            var src1Value = src1[i];

            // Check if any values in src2 have the same row index as the column index of the value in src1.
            if(map.containsKey(c1)) {
                int r1 = rowIndices1[i]; // = i
                int rowIdx = r1*cols2;

                for(int j : map.get(c1)) { // Iterate over all items in src2 where rowData[j] == colData[j]
                    int c2 = colIndices2[j]; // = j
                    dest[rowIdx + c2] = dest[rowIdx + c2].add(src1Value.mult(src2[j]));
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero items for the second sparse matrix.
     * @param colIndices2 column indices of non-zero items for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static Complex128[] concurrentStandard(Complex128[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                  double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Complex128[] dest = new Complex128[rows1*cols2];
        Arrays.fill(dest, Complex128.ZERO);
        ConcurrentMap<Integer, Complex128> destMap = new ConcurrentHashMap<>();

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int c1 = colIndices1[i]; // = k
                var src1Value = src1[i];

                // Check if any values in src2 have the same row index as the column index of the value in src1.
                if(map.containsKey(c1)) {
                    int r1 = rowIndices1[i]; // = i
                    int rowIdx = r1*cols2;

                    for(int j : map.get(c1)) { // Iterate over all items in src2 where rowData[j] == colData[j]
                        int idx = rowIdx + colIndices2[j];
                        destMap.put(idx, destMap.getOrDefault(idx, Complex128.ZERO).add(src1Value.mult(src2[j])));
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


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static Complex128[] standardVector(Complex128[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                              double[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Complex128[] dest = new Complex128[rows1*cols2];
        Arrays.fill(dest, Complex128.ZERO);

        // r1, c1, and r2 store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0, stop=src1.length; i<stop; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k
            var sum = dest[r1*cols2];
            var src1Value = src1[i];

            for(int j=0, stop2=src2.length; j<stop2; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    sum = sum.add(src1Value.mult(src2[j]));
                }
            }

            dest[r1*cols2] = sum;
        }

        return dest;
    }


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static Complex128[] concurrentStandardVector(Complex128[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                        double[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Complex128[] dest = new Complex128[rows1*cols2];
        Arrays.fill(dest, Complex128.ZERO);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices1[i]; // = i
                int c1 = colIndices1[i]; // = k
                Complex128 sum = dest[r1*cols2];
                Complex128 src1Value = src1[i];

                for(int j=0, stop=src2.length; j<stop; j++) {
                    int r2 = indices[j]; // = k

                    if(c1==r2) { // Then we multiply and add to sum.
                        Complex128 product = src1Value.mult(src2[j]);
                        sum = sum.add(product);
                    }
                }

                synchronized (dest) {
                    dest[r1*cols2] = sum;
                }
            }
        });

        return dest;
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
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static Complex128[] standard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                        Complex128[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Complex128[] dest = new Complex128[rows1*cols2];
        Arrays.fill(dest, Complex128.ZERO);

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        for(int i=0; i<src1.length; i++) {
            int c1 = colIndices1[i]; // = k
            var src1Value = src1[i];

            // Check if any values in src2 have the same row index as the column index of the value in src1.
            if(map.containsKey(c1)) {
                int r1 = rowIndices1[i]; // = i
                int rowIdx = r1*cols2;

                for(int j : map.get(c1)) { // Iterate over all items in src2 where rowData[j] == colData[j]
                    int c2 = colIndices2[j]; // = j
                    dest[rowIdx + c2] = dest[rowIdx + c2].add(src2[j].mult(src1Value));
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero items for the second sparse matrix.
     * @param colIndices2 column indices of non-zero items for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static Complex128[] concurrentStandard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                  Complex128[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Complex128[] dest = new Complex128[rows1*cols2];
        Arrays.fill(dest, Complex128.ZERO);
        ConcurrentMap<Integer, Complex128> destMap = new ConcurrentHashMap<>();

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int c1 = colIndices1[i]; // = k
                var src1Value = src1[i];

                // Check if any values in src2 have the same row index as the column index of the value in src1.
                if(map.containsKey(c1)) {
                    int r1 = rowIndices1[i]; // = i
                    int rowIdx = r1*cols2;

                    for(int j : map.get(c1)) { // Iterate over all items in src2 where rowData[j] == colData[j]
                        int idx = rowIdx + colIndices2[j];
                        destMap.put(idx, destMap.getOrDefault(idx, Complex128.ZERO).add(src2[j].mult(src1Value)));
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


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static Complex128[] standardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                              Complex128[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.getSize(0);

        Complex128[] dest = new Complex128[rows1];
        Arrays.fill(dest, Complex128.ZERO);

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k
            var src1Value = src1[i];
            var sum = dest[r1];

            for(int j=0; j<src2.length; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    sum = sum.add(src2[j].mult(src1Value));
                }
            }

            dest[r1] = sum;
        }

        return dest;
    }


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.
     * @param src1 Non-zero items of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero items for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero items for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero items of the second sparse matrix.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static Complex128[] concurrentStandardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                        Complex128[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.getSize(0);

        Complex128[] dest = new Complex128[rows1];
        Arrays.fill(dest, Complex128.ZERO);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices1[i]; // = i
                int c1 = colIndices1[i]; // = k
                var sum = dest[r1];
                var src1Value = src1[i];

                for(int j=0; j<src2.length; j++) {
                    int r2 = indices[j]; // = k

                    if(c1==r2) { // Then we multiply and add to sum.
                        Complex128 product = src2[j].mult(src1Value);
                        sum = sum.add(product);
                    }
                }

                synchronized (dest) {
                    dest[r1] = sum;
                }
            }
        });

        return dest;
    }
}
