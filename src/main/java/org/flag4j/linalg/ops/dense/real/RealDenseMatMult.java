/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;

/**
 * This class contains several low level methods for computing matrix-matrix multiplications. <br>
 * <b>WARNING:</b> These methods do not perform any sanity checks.
 */
public final class RealDenseMatMult {

    private RealDenseMatMult() {
        // Hide default constructor.
    }


    /**
     * Computes the matrix multiplication between two real dense matrices using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] standard(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];
        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*rows2;
            destIndexStart = i*cols2;

            for(int j=0; j<cols2; j++) {
                src2Index = j;
                src1Index = src1IndexStart;
                destIndex = destIndexStart + j;
                end = src1Index + rows2;
                double sum = 0;

                while(src1Index<end) {
                    sum += src1[src1Index++]*src2[src2Index];
                    src2Index += cols2;
                }

                dest[destIndex] += sum;
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two real dense matrices using the standard algorithm with j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] reordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        double src1Val;
        int src2Index;
        int destIndex;
        int src1Start;
        int destIndexStart;
        int end;

        for(int i=0; i<rows1; i++) {
            src1Start = i*cols1;
            destIndexStart = i*cols2;

            for(int k=0; k<cols1; k++) {
                src2Index = k*cols2;
                destIndex = destIndexStart;
                end = src2Index + cols2;
                src1Val = src1[src1Start + k];

                while(src2Index < end) {
                    dest[destIndex++] += src1Val*src2[src2Index++];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] blocked(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);
        int cols1 = shape1.getSize(1);

        double[] dest = new double[rows1 * cols2];
        int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int src1Start, destStart, stopIndex;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);
            for(int jj = 0; jj<cols2; jj+=blockSize) {
                jBound = Math.min(jj + blockSize, cols2);
                for(int kk = 0; kk<cols1; kk+=blockSize) {
                    kBound = Math.min(kk + blockSize, cols1);

                    // Multiply current blocks.
                    for(int i=ii; i<iBound; i++) {
                        src1Start = i*cols1 + kk;
                        stopIndex = src1Start+(kBound-kk);
                        destStart = i*cols2;

                        for (int j=jj; j<jBound; j++) {
                            destIndex = destStart + j;
                            src1Index = src1Start;
                            src2Index = kk*cols2 + j;
                            double sum = 0;

                            while(src1Index < stopIndex) {
                                sum += src1[src1Index++] * src2[src2Index];
                                src2Index+=cols2;
                            }

                            dest[destIndex] += sum;
                        }
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a blocked algorithm with the j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] blockedReordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols2 = shape2.getSize(1);
        int cols1 = shape1.getSize(1);

        double[] dest = new double[rows1*cols2];
        int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int destStart, src1Start, stopIndex;
        int destIndex, src2Index;
        double src1Value;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);

            for(int kk = 0; kk<cols1; kk+=blockSize) {
                kBound = Math.min(kk + blockSize, cols1);

                for(int jj = 0; jj<cols2; jj+=blockSize) {
                    jBound = Math.min(jj + blockSize, cols2);

                    // Multiply current blocks.
                    for(int i=ii; i<iBound; i++) {
                        destStart = i*cols2;
                        src1Start = i*cols1;
                        stopIndex = destStart+jBound;
                        destStart+=jj;

                        for (int k=kk; k<kBound; k++) {
                            destIndex = destStart;
                            src1Value = src1[src1Start + k];
                            src2Index = k*cols2 + jj;

                            while(destIndex<stopIndex) {
                                dest[destIndex++] += src1Value*src2[src2Index++];
                            }
                        }
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentStandard(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentOperation(rows1, (startRow, endRow) -> {
            for (int i = startRow; i < endRow; i++) {
                int src1IndexStart = i * cols1;
                int destIndexStart = i * cols2;

                for (int j = 0; j < cols2; j++) {
                    int src2Index = j;
                    int src1Index = src1IndexStart;
                    int destIndex = destIndexStart + j;
                    double sum = 0;

                    for(int k = 0; k < cols1; k++) {
                        sum += src1[src1Index++] * src2[src2Index];
                        src2Index += cols2;
                    }

                    dest[destIndex] += sum;
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm with j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentReordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentOperation(rows1, (startRow, endRow) -> {
            for(int i = startRow; i<endRow; i++) {
                int src1IndexStart = i*rows2;
                int destIndexStart = i*cols2;

                for(int k=0; k<rows2; k++) {
                    int src2Index = k*cols2;
                    int destIndex = destIndexStart;
                    int end = src2Index + cols2;
                    double src1Value = src1[src1IndexStart++];

                    while(src2Index<end) {
                        dest[destIndex++] += src1Value*src2[src2Index++];
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentBlocked(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (ii, iBound) -> {
            for (int jj = 0; jj < cols2; jj += blockSize) {
                int jBound = Math.min(jj + blockSize, cols2);

                for (int kk = 0; kk < cols1; kk += blockSize) {
                    int kBound = Math.min(kk + blockSize, cols1);

                    // Multiply current blocks.
                    for (int i = ii; i < iBound; i++) {
                        int src1Start = i * cols1 + kk;
                        int stopIndex = src1Start + (kBound - kk);
                        int destStart = i * cols2;

                        for (int j = jj; j < jBound; j++) {
                            int destIndex = destStart + j;
                            int src1Index = src1Start;
                            int src2Index = kk * cols2 + j;
                            double sum = 0;

                            while (src1Index < stopIndex) {
                                sum += src1[src1Index++] * src2[src2Index];
                                src2Index += cols2;
                            }

                            dest[destIndex] += sum;
                        }
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of a blocked
     * algorithm with the j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentBlockedReordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int cols2 = shape2.getSize(1);

        double[] dest = new double[rows1*cols2];
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (blockStart, blockEnd) -> {
            for(int ii=blockStart; ii<blockEnd; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int kk = 0; kk<cols1; kk+=blockSize) {
                    int kBound = Math.min(kk + blockSize, cols1);

                    for(int jj = 0; jj<cols2; jj+=blockSize) {
                        int jBound = Math.min(jj + blockSize, cols2);

                        // Multiply current blocks.
                        for(int i=ii; i<iBound; i++) {
                            int destStart = i*cols2;
                            int src1Start = i*cols1;
                            int stopIndex = destStart+jBound;
                            destStart += jj;

                            for (int k=kk; k<kBound; k++) {
                                int destIndex = destStart;
                                double src1Value = src1[src1Start + k];
                                int src2Index = k*cols2 + jj;

                                while(destIndex<stopIndex) {
                                    dest[destIndex++] += src1Value*src2[src2Index++];
                                }
                            }
                        }
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] standardVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = shape2.getSize(0);

        double[] dest = new double[rows1];
        int src1Index, src2Index;

        for(int i=0; i<rows1; i++) {
            src1Index = i*cols1;
            src2Index = 0;
            double sum = dest[i];

            while(src2Index<rows2) {
                sum += src1[src1Index++]*src2[src2Index++];
            }

            dest[i] += sum;
        }

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] blockedVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = shape2.getSize(0);

        double[] dest = new double[rows1];
        int blockSize = Configurations.getBlockSize();
        int iBound, kBound;
        int src1Index, src2Index;

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii+blockSize, rows1);

            for(int kk=0; kk<rows2; kk+=blockSize) {
                kBound = Math.min(kk+blockSize, rows2);

                // Multiply the current blocks
                for(int i=ii; i<iBound; i++) {
                    src1Index = i*cols1 + kk;
                    src2Index = kk;
                    double sum = 0;

                    while(src2Index<kBound) {
                        sum += src1[src1Index++]*src2[src2Index++];
                    }

                    dest[i] += sum;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentStandardVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = shape2.getSize(0);

        double[] dest = new double[rows1];

        ThreadManager.concurrentOperation(rows1, (rowStart, rowEnd) -> {
            for(int i=rowStart; i<rowEnd; i++) {
                int src1Index = i*cols1;
                int src2Index = 0;
                double sum = 0;

                while(src2Index<rows2) {
                    sum += src1[src1Index++]*src2[src2Index++];
                }

                dest[i] += sum;
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentBlockedVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);
        int rows2 = shape2.getSize(0);

        double[] dest = new double[rows1];
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (rowStart, rowEnd) -> {
            for(int ii=rowStart; ii<rowEnd; ii+=blockSize) {
                int iBound = Math.min(ii+blockSize, rows1);

                for(int kk=0; kk<rows2; kk+=blockSize) {
                    int kBound = Math.min(kk+blockSize, rows2);

                    // Multiply the current blocks
                    for(int i=ii; i<iBound; i++) {
                        int src1Index = i*cols1 + kk;
                        int src2Index = kk;
                        double sum = 0;

                        while(src2Index<kBound) {
                            sum += src1[src1Index++]*src2[src2Index++];
                        }

                        dest[i] += sum;
                    }
                }
            }
        });

        return dest;
    }
}
