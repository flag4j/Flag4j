/*
 * MIT License
 *
 * Copyright (c) 2022-2026. Jacob Watters
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
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;


/**
 * This class contains several low-level algorithms for computing the transpose of real dense tensors.
 */
public final class RealDenseTranspose {

    private RealDenseTranspose() {
        // Hide constructor for utility class.
    }


    /**
     * Transposes tensor along specified axes using a standard transpose algorithm. In this context, transposing a
     * tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The transpose of the tensor along the specified axes.
     */
    public static double[] standard(double[] src, Shape shape, int axis1, int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        double[] dest = new double[shape.numel().intValue()];
        Shape destShape = shape.swapAxes(axis1, axis2);
        int[] destIndices;

        for(int i=0; i<src.length; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
            dest[destShape.get1DIndex(destIndices)] = src[i]; // Apply transpose for the element
        }

        return dest;
    }


    /**
     * Computes the transpose of a tensor. That is, interchanges the axes of the tensor so that it matches
     * the specified axes permutation.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of the tensor along the specified axes.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static double[] standard(double[] src, Shape shape, int[] axes) {
        ValidateParameters.ensurePermutation(axes);
        ValidateParameters.ensureAllEqual(shape.getRank(), axes.length);
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        double[] dest = new double[shape.numel().intValue()];
        Shape destShape = shape.permuteAxes(axes);
        int[] destIndices;

        for(int i=0; i<src.length; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.permute(destIndices, axes); // Compute destination indices.
            dest[destShape.get1DIndex(destIndices)] = src[i]; // Apply transpose for the element
        }

        return dest;
    }


    /**
     * Computes the transpose of a tensor using a concurrent implementation. That is, interchanges the axes of the
     * tensor so that it matches the specified axes permutation.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of the tensor along the specified axes.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static double[] standardConcurrent(double[] src, Shape shape, int[] axes) {
        ValidateParameters.ensurePermutation(axes);
        ValidateParameters.ensureAllEqual(shape.getRank(), axes.length);
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        double[] dest = new double[shape.numel().intValue()];
        Shape destShape = shape.permuteAxes(axes);

        ThreadManager.concurrentOperation(src.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int[] destIndices = shape.getNdIndices(i);
                ArrayUtils.permuteUnsafe(destIndices, axes); // Compute destination indices.
                dest[destShape.get1DIndex(destIndices)] = src[i]; // Apply transpose for the element
            }
        });

        return dest;
    }


    /**
     * Transposes tensor along specified axes using a standard concurrent transpose algorithm.
     * In this context, transposing a tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The transpose of the tensor along the specified axes.
     */
    public static double[] standardConcurrent(double[] src, Shape shape, int axis1, int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        double[] dest = new double[shape.numel().intValue()];
        Shape destShape = shape.swapAxes(axis1, axis2);

        // Compute transpose concurrently.
        ThreadManager.concurrentOperation(src.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int[] destIndices = shape.getNdIndices(i);
                ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
                dest[destShape.get1DIndex(destIndices)] = src[i]; // Apply transpose for the element
            }
        });

        return dest;
    }


    /**
     * Transposes a matrix using the standard algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The transpose of the matrix.
     */
    public static double[] standardMatrix(double[] src, int numRows, int numCols) {
        double[] dest = new double[numRows*numCols];

        int destIndex, srcIndex, end;

        for (int i=0; i<numCols; i++) {
            srcIndex = i;
            destIndex = i*numRows;
            end = destIndex + numRows;

            while (destIndex < end) {
                dest[destIndex++] = src[srcIndex];
                srcIndex += numCols;
            }
        }

        return dest;
    }


    /**
     * Transposes a matrix using a blocked algorithm. To get or set the block size see
     * {@link Configurations#getBlockSize()} or {@link Configurations#setBlockSize(int)}.
     * @param src Source data to compute transpose of.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The transpose of this tensor along specified axes
     */
    public static double[] blockedMatrix(double[] src, int numRows, int numCols) {
        double[] dest = new double[numRows*numCols];
        int blockSize = Configurations.getBlockSize();
        int srcIndexStart, destIndexStart;
        int srcIndex, destIndex, srcIndexEnd, destIndexEnd;
        int blockHeight;

        for(int ii=0; ii<numRows; ii+=blockSize) {
            blockHeight = Math.min(ii+blockSize, numRows) - ii;
            srcIndexStart = ii*numCols;
            destIndexStart = ii;

            for(int jj=0; jj<numCols; jj+=blockSize) {
                srcIndexEnd = srcIndexStart + Math.min(numCols-jj, blockSize);

                while(srcIndexStart<srcIndexEnd) {
                    srcIndex = srcIndexStart;
                    destIndex = destIndexStart;
                    destIndexEnd = destIndex + blockHeight;

                    while(destIndex<destIndexEnd) {
                        dest[destIndex++] = src[srcIndex];
                        srcIndex+=numCols;
                    }

                    destIndexStart += numRows;
                    srcIndexStart++;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the transpose of a matrix using a standard concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The transpose of the source matrix.
     */
    public static double[] standardMatrixConcurrent(double[] src, int numRows, int numCols) {
        double[] dest = new double[src.length];

        // Compute transpose concurrently.
        ThreadManager.concurrentOperation(numCols, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int srcIndex = i;
                int destIndex = i*numRows;
                int end = destIndex + numRows;

                while (destIndex < end) {
                    dest[destIndex++] = src[srcIndex];
                    srcIndex += numCols;
                }
            }
        });

        return dest;
    }


    /**
     * Computes the transpose of a matrix using a blocked concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The transpose of the source matrix.
     */
    public static double[] blockedMatrixConcurrent(double[] src, int numRows, int numCols) {
        double[] dest = new double[src.length];
        int blockSize = Configurations.getBlockSize();

        // Compute transpose concurrently.
        ThreadManager.concurrentBlockedOperation(numRows, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int blockHeight = Math.min(ii+blockSize, numRows) - ii;
                int srcIndexStart = ii*numCols;
                int destIndexStart = ii;

                for(int jj=0; jj<numCols; jj+=blockSize) {
                    int srcIndexEnd = srcIndexStart + Math.min(numCols-jj, blockSize);

                    while(srcIndexStart<srcIndexEnd) {
                        int srcIndex = srcIndexStart;
                        int destIndex = destIndexStart;
                        int destIndexEnd = destIndex + blockHeight;

                        while(destIndex<destIndexEnd) {
                            dest[destIndex++] = src[srcIndex];
                            srcIndex+=numCols;
                        }

                        destIndexStart += numRows;
                        srcIndexStart++;
                    }
                }
            }
        });

        return dest;
    }


    // ------------------------------------ Integer Transpose ------------------------------------
    /**
     * Transposes a matrix using the standard algorithm.
     * @param src Entries of the matrix to transpose.
     * @return The transpose of the matrix.
     */
    public static int[][] standardIntMatrix(int[][] src) {
        if(src.length == 0) return new int[0][0];

        int rows = src.length;
        int cols = src[0].length;
        int[][] dest = new int[cols][rows];

        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                dest[j][i] = src[i][j];

        return dest;
    }


    /**
     * Transposes an integer matrix using a blocked algorithm.
     * @param src Entries of the matrix to transpose.
     * @return The transpose of the matrix.
     */
    public static int[][] blockedIntMatrix(int[][] src) {
        int[][] dest = new int[src[0].length][src.length];
        int blockSize = Configurations.getBlockSize();

        int iBound;
        int jBound;

        for(int ii=0; ii<src[0].length; ii+=blockSize) {
            iBound = Math.min(src[0].length, ii+blockSize);

            for(int jj=0; jj<src.length; jj+=blockSize) {
                jBound = Math.min(src.length, jj+blockSize);

                for(int i=ii; i<iBound; i++) {
                    for(int j=jj; j<jBound; j++) {
                        dest[i][j] = src[j][i];
                    }
                }
            }
        }

        return dest;
    }
}
