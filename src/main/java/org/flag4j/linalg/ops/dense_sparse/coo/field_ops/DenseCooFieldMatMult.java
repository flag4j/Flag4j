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

package org.flag4j.linalg.ops.dense_sparse.coo.field_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.numbers.Field;

import java.util.Arrays;


/**
 * This utility class provides low level methods for computing the matrix multiplication between
 * a sparse/dense matrix and dense/sparse matrix/vector.
 */
public final class DenseCooFieldMatMult {

    private DenseCooFieldMatMult() {
        // Hide default constructor in utility class.
    }

    // TODO: Investigate if blocked algorithms provide any speedup for multiplying a sparse/dense matrix to a dense/sparse matrix.

    /**
     * Computes the matrix multiplication between a dense matrix and a sparse COO matrix using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero data of the sparse matrix.
     * @param rowIndices Row indices for non-zero data of the sparse matrix.
     * @param colIndices Column indices for non-zero data of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends Field<T>> void standard(
            T[] src1, Shape shape1, T[] src2,
            int[] rowIndices, int[] colIndices, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        for(int i=0; i<rows1; i++) {
            int destRowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            // Loop over non-zero data of sparse matrix.
            for(int j=0; j<src2.length; j++) {
                int row = rowIndices[j];
                int col = colIndices[j];

                dest[destRowOffset + col] = dest[destRowOffset + col].add(src1[src1RowOffset + row].mult(src2[j]));
            }
        }
    }


    /**
     * Computes the matrix multiplication between a sparse COO matrix and a dense matrix using a standard algorithm.
     *
     * @param src1 Non-zero data of the sparse matrix.
     * @param rowIndices Row indices for non-zero data of the sparse matrix.
     * @param colIndices Column indices for non-zero data of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends Field<T>> void standard(
            T[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        for(int i=0; i<src1.length; i++) {
            int rowOffset = rowIndices[i]*cols2;
            int colOffset = colIndices[i]*cols2;

            for(int j=0; j<cols2; j++)
                dest[rowOffset + j] = dest[rowOffset + j].add(src1[i].mult(src2[colOffset + j]));
        }
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a real sparse matrix using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero data of the sparse matrix.
     * @param rowIndices Row indices for non-zero data of the sparse matrix.
     * @param colIndices Column indices for non-zero data of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends Field<T>> void concurrentStandard(
            T[] src1, Shape shape1,
            T[] src2, int[] rowIndices, int[] colIndices, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int rowOffset = i*cols2;

                // Loop over non-zero data of sparse matrix.
                for(int j=0; j<src2.length; j++) {
                    int row = rowIndices[j];
                    int col = colIndices[j];
                    T product = src1[i*cols1 + row].mult(src2[j]);

                    synchronized (dest) {
                        dest[rowOffset + col] = dest[rowOffset + col].add(product);
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a real dense matrix
     * using a concurrent standard algorithm.
     *
     * @param src1 Non-zero data of the sparse matrix.
     * @param rowIndices Row indices for non-zero data of the sparse matrix.
     * @param colIndices Column indices for non-zero data of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static <T extends Field<T>> void concurrentStandard(
            T[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                Field<T> v1 = src1[i];
                int row = rowIndices[i];
                int col = colIndices[i];
                int rowOffset = row*cols2;
                int row2Offset = col*cols2;

                for(int j=0; j<cols2; j++) {
                    T product = v1.mult(src2[row2Offset + j]);

                    synchronized (dest) {
                        dest[rowOffset + j] = dest[rowOffset + j].add(product);
                    }
                }
            }
        });
    }


    // -------------------- Below are the matrix-vector multiplication algorithms --------------------

    /**
     * Computes the dense matrix sparse vector multiplication using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends Field<T>> void standardVector(
            T[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int denseRows = shape1.getSize(0);
        int denseCols = shape1.getSize(1);
        int nonZeros = src2.length;

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        for(int i=0; i<denseRows; i++) {
            int rowOffset = i*denseCols;
            T val = dest[i];

            for(int j=0; j<nonZeros; j++)
                val = val.add(src1[rowOffset + indices[j]].mult(src2[j]));

            dest[i] = val;
        }
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero data in sparse matrix.
     * @param colIndices Column indices of non-zero data in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends Field<T>> void standardVector(
            T[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        for(int i=0, size=src1.length; i<size; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            dest[row] = dest[row].add(src1[i].mult(src2[col]));
        }
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends Field<T>> void blockedVector(
            T[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;
        int bsize = Configurations.getBlockSize(); // Get the block size to use.
        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<rows2; jj += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    T val = dest[i];
                    int src1RowOffset = i*cols1;

                    for(int j=jj; j<jj+bsize && j<rows2; j++)
                        val = val.add(src1[src1RowOffset + indices[j]].mult(src2[j]));

                    dest[i] = val; // Update destination value.
                }
            }
        }
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends Field<T>> void concurrentStandardVector(
            T[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                T sum = dest[i];

                for(int j=0; j<rows2; j++) {
                    int k = indices[j];
                    sum = sum.add(src1[i*cols1 + k].mult(src2[j]));
                }

                dest[i] = sum; // Update destination entry.
            }
        });
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero data in sparse matrix.
     * @param colIndices Column indices of non-zero data in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends Field<T>> void concurrentStandardVector(
            T[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros.

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];

                T product = src1[i].mult(src2[col]);

                synchronized (dest) {
                    dest[row] = dest[row].add(product);
                }
            }
        });
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends Field<T>> void concurrentBlockedVector(
            T[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;
        final int bsize = Configurations.getBlockSize(); // Get the block size to use.
        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null); // Initialize to zeros..

        // Blocked matrix-vector multiply.
        ThreadManager.concurrentBlockedOperation(rows1, bsize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii += bsize) {
                for(int jj=0; jj<rows2; jj += bsize) {

                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        T val = dest[i];
                        int src1RowOffset = i*cols1;

                        for(int j=jj; j<jj+bsize && j<rows2; j++)
                            val = val.add(src1[src1RowOffset + indices[j]].mult(src2[j]));

                        dest[i] = val; // Update destination entry.
                    }
                }
            }
        });
    }
}
