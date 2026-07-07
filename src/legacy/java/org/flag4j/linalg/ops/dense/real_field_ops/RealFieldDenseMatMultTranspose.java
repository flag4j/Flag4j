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

package org.flag4j.linalg.ops.dense.real_field_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.numbers.FieldElement;

import java.util.Arrays;

/**
 * <p>This class contains several low level methods for computing matrix-matrix multiplications with a transpose for a
 * real dense matrix and a dense field matrix.
 *
 * <p><b>Warning:</b> These methods do not perform any sanity checks.
 */
public final class RealFieldDenseMatMultTranspose {

    private RealFieldDenseMatMultTranspose() {
        // Hide default constructor.
        
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix. This method may be significantly faster than computing the
     * transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTranspose(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);
        
        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*cols2;
            destIndexStart = i*rows2;
            end = src1IndexStart + cols2;

            for(int j=0; j<rows2; j++) {
                src1Index = src1IndexStart;
                src2Index = j*cols2;
                destIndex = destIndexStart + j;
                T sum = dest[destIndex];

                while(src1Index<end)
                    sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                dest[destIndex] = sum;
            }
        }
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix using a blocked algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTransposeBlocked(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int src1Start, destStart, end;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);

            for(int jj = 0; jj<rows2; jj+=blockSize) {
                jBound = Math.min(jj + blockSize, rows2);

                for(int kk = 0; kk<cols2; kk+=blockSize) {
                    kBound = Math.min(kk + blockSize, cols2);

                    // Multiply the blocks
                    for(int i=ii; i<iBound; i++) {
                        destStart = i*rows2;
                        src1Start = i*cols2 + kk;
                        end = src1Start + kBound - kk;

                        for(int j=jj; j<jBound; j++) {
                            destIndex = destStart + j;
                            src1Index = src1Start;
                            src2Index = j*cols2 + kk;
                            T sum = dest[destIndex];

                            while(src1Index<end)
                                sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                            dest[destIndex] = sum;
                        }
                    }
                }
            }
        }
    }



    /**
     * Multiplies a matrix to the transpose of a second matrix using a concurrent algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTransposeConcurrent(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*cols2;
                int destIndexStart = i*rows2;
                int end = src1IndexStart + cols2;

                for(int j=0; j<rows2; j++) {
                    int src1Index = src1IndexStart;
                    int src2Index = j*cols2;
                    int destIndex = destIndexStart + j;
                    T sum = dest[destIndex];

                    while(src1Index<end)
                        sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                    dest[destIndex] = sum;
                }
            }
        });
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix using a concurrent implementation of a blocked algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTransposeBlockedConcurrent(
            T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int jj = 0; jj<rows2; jj+=blockSize) {
                    int jBound = Math.min(jj + blockSize, rows2);

                    for(int kk = 0; kk<cols2; kk+=blockSize) {
                        int kBound = Math.min(kk + blockSize, cols2);

                        // Multiply the blocks
                        for(int i=ii; i<iBound; i++) {
                            int destStart = i*rows2;
                            int src1Start = i*cols2 + kk;
                            int end = src1Start + kBound - kk;

                            for(int j=jj; j<jBound; j++) {
                                int destIndex = destStart + j;
                                int src1Index = src1Start;
                                int src2Index = j*cols2 + kk;
                                T sum = dest[destIndex];

                                while(src1Index<end)
                                    sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                                dest[destIndex] = sum;
                            }
                        }
                    }
                }
            }
        });
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix. This method may be significantly faster than computing the
     * transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTranspose(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src2[0].getZero() : null);

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*cols2;
            destIndexStart = i*rows2;
            end = src1IndexStart + cols2;

            for(int j=0; j<rows2; j++) {
                src1Index = src1IndexStart;
                src2Index = j*cols2;
                destIndex = destIndexStart + j;
                T sum = dest[destIndex];

                while(src1Index<end)
                    sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

                dest[destIndex] = sum;
            }
        }
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix using a blocked algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTransposeBlocked(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src2[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int src1Start, destStart, end;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);

            for(int jj = 0; jj<rows2; jj+=blockSize) {
                jBound = Math.min(jj + blockSize, rows2);

                for(int kk = 0; kk<cols2; kk+=blockSize) {
                    kBound = Math.min(kk + blockSize, cols2);

                    // Multiply the blocks
                    for(int i=ii; i<iBound; i++) {
                        destStart = i*rows2;
                        src1Start = i*cols2 + kk;
                        end = src1Start + kBound - kk;

                        for(int j=jj; j<jBound; j++) {
                            destIndex = destStart + j;
                            src1Index = src1Start;
                            src2Index = j*cols2 + kk;
                            T sum = dest[destIndex];

                            while(src1Index<end)
                                sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

                            dest[destIndex] = sum;
                        }
                    }
                }
            }
        }
    }



    /**
     * Multiplies a matrix to the transpose of a second matrix using a concurrent algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTransposeConcurrent(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*cols2;
                int destIndexStart = i*rows2;
                int end = src1IndexStart + cols2;

                for(int j=0; j<rows2; j++) {
                    int src1Index = src1IndexStart;
                    int src2Index = j*cols2;
                    int destIndex = destIndexStart + j;
                    T sum = dest[destIndex];

                    while(src1Index<end)
                        sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

                    dest[destIndex] = sum;
                }
            }
        });
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix using a concurrent implementation of a blocked algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in (modified). Must have length
     * {@code shape1.get(0)*shape1.get(0)}.
     */
    public static <T extends FieldElement<T>> void multTransposeBlockedConcurrent(
            double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.getSize(0);
        int rows2 = shape2.getSize(0);
        int cols2 = shape2.getSize(1);

        Arrays.fill(dest, (src1.length > 0) ? src2[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int jj = 0; jj<rows2; jj+=blockSize) {
                    int jBound = Math.min(jj + blockSize, rows2);

                    for(int kk = 0; kk<cols2; kk+=blockSize) {
                        int kBound = Math.min(kk + blockSize, cols2);

                        // Multiply the blocks
                        for(int i=ii; i<iBound; i++) {
                            int destStart = i*rows2;
                            int src1Start = i*cols2 + kk;
                            int end = src1Start + kBound - kk;

                            for(int j=jj; j<jBound; j++) {
                                int destIndex = destStart + j;
                                int src1Index = src1Start;
                                int src2Index = j*cols2 + kk;
                                T sum = dest[destIndex];

                                while(src1Index<end)
                                    sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

                                dest[destIndex] = sum;
                            }
                        }
                    }
                }
            }
        });
    }
}
