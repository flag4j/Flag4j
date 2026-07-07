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

package org.flag4j.linalg.ops.dense_sparse.coo.real;

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;

/**
 * This class contains low level methods for computing the matrix multiplication (and matrix vector multiplication) between
 * a real dense/sparse matrix and a real sparse/dense matrix or vector.
 */
public class RealDenseSparseMatMult {

    private RealDenseSparseMatMult() {
        // Hide default constructor.
        
    }

    // TODO: Investigate if blocked algorithms provide any speedup for multiplying a sparse/dense matrix to a dense/sparse matrix.

    /**
     * Computes the matrix multiplication between a real dense matrix and a real sparse matrix using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @return The result of the matrix multiplication.
     */
    public static double[] standard(double[] src1, Shape shape1, double[] src2,
                                    int[] rowIndices, int[] colIndices, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        int row, col;
        int destStart, src1Start;

        for(int i=0; i<rows1; i++) {
            destStart = i*cols2;
            src1Start = i*cols1;

            // Loop over non-zero items of sparse matrix.
            for(int j=0; j<src2.length; j++) {
                row = rowIndices[j];
                col = colIndices[j];

                dest[destStart + col] += src1[src1Start + row]*src2[j];
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a real dense matrix using a standard algorithm.
     *
     * @param src1 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static double[] standard(double[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
                                    double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        int row, col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            int destRow = row*cols2;
            int src2Row = col*cols2;

            for(int j=0; j<cols2; j++) {
                dest[destRow + j] += src1[i]*src2[src2Row + j];
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a real sparse matrix using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @return The result of the matrix multiplication.
     */
    public static double[] concurrentStandard(double[] src1, Shape shape1, double[] src2,
                                    int[] rowIndices, int[] colIndices, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                double[] localResult = new double[cols2]; // Store the result for the local thread.
                int destRow = i*cols2;
                int src1Row = i*cols1;

                // Loop over non-zero items of sparse matrix.
                for(int j=0; j<src2.length; j++) {
                    int row = rowIndices[j];
                    int col = colIndices[j];

                    localResult[col] += src1[src1Row + row]*src2[j];
                }

                // Update the shared destination array by accumulating the local result.
                synchronized(dest) {
                    for (int j=0; j<cols2; j++) {
                        dest[destRow + j] += localResult[j];
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a real dense matrix
     * using a concurrent standard algorithm.
     *
     * @param src1 Non-zero items of the sparse matrix.
     * @param rowIndices Row indices for non-zero items of the sparse matrix.
     * @param colIndices Column indices for non-zero items of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static double[] concurrentStandard(double[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
                                              double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices[i];
                int c1 = colIndices[i];

                int destRowStart = r1 * cols2;
                int src2RowStart = c1 * cols2;

                double[] localResult = new double[cols2];
                for (int j = 0; j < cols2; j++) {
                    localResult[j] = src1[i]*src2[src2RowStart + j];
                }

                synchronized (dest) {
                    for (int j = 0; j < cols2; j++) {
                        dest[destRowStart + j] += localResult[j];
                    }
                }
            }
        });


        return dest;
    }


    // -------------------- Below are the matrix-vector multiplication algorithms --------------------

    /**
     * Computes the dense matrix sparse vector multiplication using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] standardVector(double[] src1, Shape shape1, double[] src2, int[] indices) {
        int denseRows = shape1.getSize(0);
        int denseCols = shape1.getSize(1);
        int nonZeros = src2.length;

        double[] dest = new double[denseRows];
        int k;

        for(int i=0; i<denseRows; i++) {
            double sum = dest[i];

            for(int j=0; j<nonZeros; j++) {
                k = indices[j];
                sum += src1[i*denseCols + k]*src2[j];
            }

            dest[i] = sum;
        }

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero items in sparse matrix.
     * @param colIndices Column indices of non-zero items in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] standardVector(double[] src1, int[] rowIndices, int[] colIndices,
                                          Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        double[] dest = new double[rows1];
        int row, col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];

            dest[row] += src1[i]*src2[col];
        }

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] blockedVector(double[] src1, Shape shape1, double[] src2, int[] indices) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        int bsize = Configurations.getBlockSize(); // Get the block size to use.

        double[] dest = new double[rows1];
        int k;

        // Blocked matrix-vector multiply.
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<rows2; jj += bsize) {
                // Multiply the current blocks.
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    double sum = dest[i];
                    int src1RowOffset = i*cols1;

                    for(int j=jj; j<jj+bsize && j<rows2; j++) {
                        k = indices[j];
                        sum += src1[src1RowOffset + k]*src2[j];
                    }

                    dest[i] = sum;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] concurrentStandardVector(double[] src1, Shape shape1, double[] src2, int[] indices) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        double[] dest = new double[rows1];

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int rowOffset = i*cols1;
                double sum = dest[i];

                for(int j=0; j<rows2; j++) {
                    int k = indices[j];
                    sum += src1[rowOffset + k]*src2[j];
                }

                dest[i] = sum;
            }
        });

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero items in sparse matrix.
     * @param colIndices Column indices of non-zero items in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] concurrentStandardVector(double[] src1, int[] rowIndices, int[] colIndices,
                                          Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        double[] dest = new double[rows1];

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];

                double product = src1[i]*src2[col];

                synchronized (dest) {
                    dest[row] += product;
                }
            }
        });

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero items of the sparse vector.
     * @param indices Indices of non-zero items in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] concurrentBlockedVector(double[] src1, Shape shape1, double[] src2, int[] indices) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = src2.length;

        final int bsize = Configurations.getBlockSize(); // Get the block size to use.
        double[] dest = new double[rows1];

        // Blocked matrix-vector multiply.
        ThreadManager.concurrentBlockedOperation(rows1, bsize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=bsize) {
                for(int jj=0; jj<rows2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        double sum = dest[i];

                        for(int j=jj; j<jj+bsize && j<rows2; j++) {
                            int k = indices[j];
                            sum += src1[i*cols1 + k]*src2[j];
                        }

                        dest[i] = sum;
                    }
                }
            }
        });

        return dest;
    }
}
