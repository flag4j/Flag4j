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

package org.flag4j.linalg.ops.dense;

import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ArrayUtils;

import static org.flag4j.util.ArrayBuilder.getOrCreateArray;

/**
 * This utility class provides methods for computing the transpose of a dense tensor.
 */
public final class DenseTranspose {

    private DenseTranspose() {
        // Hide default constructor for utility class.
    }


    /**
     * Transposes tensor along specified axes using a standard transpose algorithm. In this context, transposing a
     * tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     */
    public static Object[] standard(Object[] src, Shape shape, int axis1, int axis2, Object[] dest) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        dest = getOrCreateArray(dest, () -> new Object[src.length]);
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
     * Transposes tensor along specified axes using a standard concurrent transpose algorithm.
     * In this context, transposing a tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     */
    public static <T extends Object> Object[] standardConcurrent(Object[] src, Shape shape, int axis1, int axis2, Object[] dest) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank "
                    + shape.getRank() + " tensor.");
        }

        dest = getOrCreateArray(dest, () -> new Object[src.length]);
        Shape destShape = shape.swapAxes(axis1, axis2);
        Object[] finalDest = dest; // Copy to an effectively final variable for lambda expression.

        // Compute transpose concurrently
        ThreadManager.concurrentOperation(src.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int[] destIndices = shape.getNdIndices(i);
                ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
                finalDest[destShape.get1DIndex(destIndices)] = src[i]; // Apply transpose for the element
            }
        });

        return dest;
    }


    /**
     * Computes the transpose of a tensor. That is, interchanges the axes of the tensor so that it matches
     * the specified axes permutation.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static <T extends Object> Object[] standard(Object[] src, Shape shape, int[] axes, Object[] dest) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        dest = getOrCreateArray(dest, () -> new Object[src.length]);
        Shape destShape = shape.permuteAxes(axes);
        int[] destIndices;

        for(int i=0; i<src.length; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.permuteUnsafe(destIndices, axes); // Compute destination indices.
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
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static double[] standard(double[] src, Shape shape, int[] axes, double[] dest) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        dest = ArrayBuilder.getOrCreateArray(dest, src.length);
        Shape destShape = shape.permuteAxes(axes);
        int[] destIndices;

        for(int i=0; i<src.length; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.permuteUnsafe(destIndices, axes); // Compute destination indices.
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
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static <T extends Object> Object[] standardConcurrent(Object[] src, Shape shape, int[] axes, Object[] dest) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        dest = getOrCreateArray(dest, () -> new Object[src.length]);
        Shape destShape = shape.permuteAxes(axes);
        Object[] finalDest = dest; // Copy to an effectively final variable for lambda expression.

        // Compute transpose concurrently
        ThreadManager.concurrentOperation(src.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int[] destIndices = shape.getNdIndices(i);
                ArrayUtils.permuteUnsafe(destIndices, axes); // Compute destination indices.
                finalDest[destShape.get1DIndex(destIndices)] = src[i]; // Apply transpose for the element
            }
        });

        return dest;
    }



    /**
     * Transposes a matrix using the standard algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     */
    public static <T extends Object> Object[] standardMatrix(Object[] src, int numRows, int numCols, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src.length]);

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
     * @param src Source matrix in the transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     */
    public static Object[] blockedMatrix(Object[] src, int numRows, int numCols, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src.length]);
        int blockSize = Configurations.getBlockSize();
        int blockRowEnd;
        int blockColEnd;
        int srcIndex, destIndex, end;

        for(int i=0; i<numCols; i+=blockSize) {
            for(int j=0; j<numRows; j+=blockSize) {
                blockRowEnd = Math.min(j+blockSize, numRows);
                blockColEnd = Math.min(i+blockSize, numCols);

                // Transpose the block beginning at (i, j)
                for(int blockI=i; blockI<blockColEnd; blockI++) {
                    srcIndex = blockI;
                    destIndex = blockI*numRows;
                    end = destIndex + blockRowEnd;

                    while (destIndex < end) {
                        dest[destIndex++] = src[srcIndex];
                        srcIndex += numCols;
                    }
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
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     */
    public static Object[] standardMatrixConcurrent(Object[] src, int numRows, int numCols, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src.length]);
        Object[] finalDest = dest; // Copy to an effectively final variable for lambda expression.

        // Compute transpose concurrently
        ThreadManager.concurrentOperation(numCols, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int srcIndex = i;
                int destIndex = i*numRows;
                int end = destIndex + numRows;

                while(destIndex < end) {
                    finalDest[destIndex++] = src[srcIndex];
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
     * @param dest Array to store the transpose result in. May be {@code null}. If not null, must satisfy:
     * {@code dest.length >= src.length}.
     * @return If {@code dest != null}, a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     */
    public static Object[] blockedMatrixConcurrent(Object[] src, int numRows, int numCols, Object[] dest) {
        dest = getOrCreateArray(dest, () -> new Object[src.length]);
        int blockSize = Configurations.getBlockSize();
        Object[] finalDest = dest; // Copy to an effectively final variable for lambda expression.

        // Compute transpose concurrently
        ThreadManager.concurrentBlockedOperation(numCols, blockSize, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                for(int j=0; j<numRows; j+=blockSize) {
                    int blockRowEnd = Math.min(j+blockSize, numRows);
                    int blockColEnd = Math.min(i+blockSize, numCols);

                    // Transpose the block beginning at (i, j)
                    for(int blockI=i; blockI<blockColEnd; blockI++) {
                        int srcIndex = blockI;
                        int destIndex = blockI*numRows;
                        int end = destIndex + blockRowEnd;

                        while(destIndex < end) {
                            finalDest[destIndex++] = src[srcIndex];
                            srcIndex += numCols;
                        }
                    }
                }
            }
        });

        return dest;
    }
}
