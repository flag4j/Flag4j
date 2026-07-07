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

package org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops;


import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.numbers.FieldElement;

import java.util.Arrays;

/**
 * This class contains low level methods for computing the matrix multiplication (and matrix vector multiplication) between
 * a real dense/sparse matrix and a sparse/dense field matrix/vector.
 */
public final class RealFieldDenseCooMatMult {

    private RealFieldDenseCooMatMult() {
        // Hide default constructor.
        
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a sparse field matrix using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void standard(
            double[] src1, Shape shape1, T[] src2,
            int[] rowIndices, int[] colIndices, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        int row;
        int col;

        for(int i=0; i<rows1; i++) {
            // Loop over non-zero items of sparse matrix.
            int destRowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            for(int j=0, len=src2.length; j<len; j++) {
                row = rowIndices[j];
                col = colIndices[j];

                dest[destRowOffset + col] = dest[destRowOffset + col].add(src2[j].mult(src1[src1RowOffset + row]));
            }
        }
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a dense field matrix using a standard algorithm.
     *
     * @param src1 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void standard(
            double[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        int row;
        int col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            int destRowOffset = row*cols2;
            int src2RowOffset = col*cols2;

            for(int j=0; j<cols2; j++)
                dest[destRowOffset + j] = dest[destRowOffset + j].add(src2[src2RowOffset + j].mult(src1[i]));
        }
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a sparse field matrix using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandard(
            double[] src1, Shape shape1, T[] src2,
            int[] rowIndices, int[] colIndices, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int rowOffset = i*cols2;
                int src2RowOffset = i*cols1;

                // Loop over non-zero items of sparse matrix.
                for(int j=0; j<src2.length; j++) {
                    int row = rowIndices[j];
                    int col = colIndices[j];
                    T product = src2[j].mult(src1[src2RowOffset + row]);

                    synchronized (dest) {
                        dest[rowOffset + col] = dest[rowOffset + col].add(product);
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a dense field matrix
     * using a concurrent standard algorithm.
     *
     * @param src1 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandard(
            double[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(var i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                int rowOffset = row*cols2;
                int src2RowOffset = col*cols2;

                for(int j=0; j<cols2; j++) {
                    T product = src2[src2RowOffset + j].mult(src1[i]);

                    synchronized (dest) {
                        dest[rowOffset + j] = dest[rowOffset + j].add(product);
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a sparse field matrix using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication in.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void standard(
            T[] src1, Shape shape1, double[] src2,
            int[] rowIndices, int[] colIndices, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);;

        int row;
        int col;

        for(int i=0; i<rows1; i++) {
            int destRowOffset = i*cols2;

            // Loop over non-zero items of sparse matrix.
            for(int j=0; j<src2.length; j++) {
                row = rowIndices[j];
                col = colIndices[j];

                dest[destRowOffset + col] = dest[destRowOffset + col].add(src1[i*cols1 + row].mult(src2[j]));
            }
        }
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a dense field matrix using a standard algorithm.
     *
     * @param src1 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void standard(
            T[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            double[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        for(int i=0; i<src1.length; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];
            int destRowOffset = row*cols2;
            int src2RowOffset = col*cols2;

            for(int j=0; j<cols2; j++)
                dest[destRowOffset + j] = dest[destRowOffset + j].add(src1[i].mult(src2[src2RowOffset + j]));
        }
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a sparse field matrix using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandard(
            T[] src1, Shape shape1, double[] src2,
            int[] rowIndices, int[] colIndices, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int destRowOffset = i*cols2;
                int productOffset = i*cols1;

                // Loop over non-zero items of sparse matrix.
                for(int j=0; j<src2.length; j++) {
                    int row = rowIndices[j];
                    int col = colIndices[j];
                    T product = src1[productOffset + row].mult(src2[j]);

                    synchronized (dest) {
                        dest[destRowOffset + col] = dest[destRowOffset + col].add(product);
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a dense field matrix
     * using a concurrent standard algorithm.
     *
     * @param src1 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @param dest Array to store the dense result of the matrix multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandard(
            T[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
            double[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                int rowOffset = row*cols2;

                for(int j=0; j<cols2; j++) {
                    T product = src1[i].mult(src2[col*cols2 + j]);

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
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void standardVector(
            double[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int denseRows = shape1.getSize(0);
        int denseCols = shape1.getSize(1);
        int nonZeros = src2.length;

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);
        int k;

        for(int i=0; i<denseRows; i++) {
            int src1RowOffset = i*denseCols;
            T sum = dest[i];

            for(int j=0; j<nonZeros; j++) {
                k = indices[j];
                sum = sum.add(src2[j].mult(src1[src1RowOffset + k]));
            }

            dest[i] = sum;
        }
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero items in sparse matrix.
     * @param colIndices Column indices of non-zero items in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void standardVector(
            double[] src1, int[] rowIndices, int[] colIndices,
            Shape shape1, T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);;
        int row;
        int col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            dest[row] = dest[row].add(src2[col].mult(src1[i]));
        }
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void blockedVector(
            double[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        int bsize = Configurations.getBlockSize(); // Get the block size to use.

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<rows2; jj += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    int src1RowOffset = i*cols1;
                    T sum = dest[i];

                    for(int j=jj; j<jj+bsize && j<rows2; j++)
                        sum = sum.add(src2[j].mult(src1[src1RowOffset + indices[j]]));

                    dest[i] = sum;
                }
            }
        }
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandardVector(
            double[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                T sum = dest[i];
                int rowOffset = i*cols1;

                for(int j=0; j<rows2; j++) {
                    int k = indices[j];
                    sum = sum.add(src2[j].mult(src1[rowOffset + k]));
                }

                dest[i] = sum;
            }
        });
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero items in sparse matrix.
     * @param colIndices Column indices of non-zero items in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandardVector(
            double[] src1, int[] rowIndices, int[] colIndices,
            Shape shape1, T[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);;

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                T product = src2[col].mult(src1[i]);

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
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentBlockedVector(
            double[] src1, Shape shape1,
            T[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        final int bsize = Configurations.getBlockSize(); // Get the block size to use.

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        // Blocked matrix-vector multiply
        ThreadManager.concurrentBlockedOperation(rows1, bsize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii += bsize) {
                for(int jj=0; jj<rows2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        T sum = dest[i];

                        for(int j=jj; j<jj+bsize && j<rows2; j++) {
                            int k = indices[j];
                            sum = sum.add(src2[j].mult(src1[i*cols1 + k]));
                        }

                        dest[i] = sum;
                    }
                }
            }
        });
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void standardVector(
            T[] src1, Shape shape1,
            double[] src2, int[] indices,
            T[] dest) {
        int denseRows = shape1.getSize(0);
        int denseCols = shape1.getSize(1);
        int nonZeros = src2.length;

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);
        int k;

        for(int i=0; i<denseRows; i++) {
            T sum = dest[i];
            int rowOffset = i*denseCols;

            for(int j=0; j<nonZeros; j++) {
                k = indices[j];
                sum = sum.add(src1[rowOffset + k].mult(src2[j]));
            }

            dest[i] = sum;
        }
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero items in sparse matrix.
     * @param colIndices Column indices of non-zero items in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void standardVector(
            T[] src1, int[] rowIndices, int[] colIndices,
            Shape shape1, double[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        for(int i=0; i<src1.length; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];
            dest[row] = dest[row].add(src1[i].mult(src2[col]));
        }
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     */
    public static <T extends FieldElement<T>> void blockedVector(
            T[] src1, Shape shape1,
            double[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int bsize = Configurations.getBlockSize(); // Get the block size to use.
        int nnz = src2.length;

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        // Blocked matrix-vector multiply.
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<nnz; jj += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    T sum = dest[i];
                    int rowOffset = i*cols1;

                    for(int j=jj; j<jj+bsize && j<nnz; j++)
                        sum = sum.add(src1[rowOffset + indices[j]].mult(src2[j]));

                    dest[i] = sum;
                }
            }
        }
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandardVector(
            T[] src1, Shape shape1,
            double[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                T sum = dest[i];

                for(int j=0; j<rows2; j++) {
                    int k = indices[j];
                    sum = sum.add(src1[i*cols1 + k].mult(src2[j]));
                }

                dest[i] = sum;
            }
        });
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero items in sparse matrix.
     * @param colIndices Column indices of non-zero items in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentStandardVector(
            T[] src1, int[] rowIndices, int[] colIndices,
            Shape shape1, double[] src2, Shape shape2,
            T[] dest) {
        int rows1 = shape1.getSize(0);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

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
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static <T extends FieldElement<T>> void concurrentBlockedVector(
            T[] src1, Shape shape1,
            double[] src2, int[] indices,
            T[] dest) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;
        final int bsize = Configurations.getBlockSize(); // Get the block size to use.

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        // Blocked matrix-vector multiply.
        ThreadManager.concurrentBlockedOperation(rows1, bsize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii += bsize) {
                for(int jj=0; jj<rows2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        T sum = dest[i];
                        int row1Offset = i*cols1;

                        for(int j=jj; j<jj+bsize && j<rows2; j++)
                            sum = sum.add(src1[row1Offset + indices[j]].mult(src2[j]));

                        dest[i] = sum;
                    }
                }
            }
        });
    }
}
