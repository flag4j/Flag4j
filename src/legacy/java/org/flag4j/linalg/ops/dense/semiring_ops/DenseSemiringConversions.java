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

package org.flag4j.linalg.ops.dense.semiring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for converting dense {@link SemiringElement} tensors to another type of tensor.
 */
public final class DenseSemiringConversions {

    private DenseSemiringConversions() {
        // Hide default constructor for utility class.
    }


    /**
     * <p>Converts a dense matrix to an equivalent sparse COO matrix.
     * <p>This should only be done for matrices which are known to be sparse.
     * <p>If no estimate is known, then using {@link #toCoo(Shape, SemiringElement[])} will estimate the sparsity as 0.9.
     * @param shape Shape of the matrix.
     * @param entries Entries of the matrix.
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern it is better to underestimate the sparsity.
     * @return A sparse COO matrix containing the non-zero values of the specified dense matrix.
     */
    public static <T extends SemiringElement<T>> SparseMatrixData<T> toCoo(
            Shape shape, T[] entries, double estimatedSparsity) {
        ValidateParameters.ensureInRange(estimatedSparsity, 0.0, 1.0, "estimatedSparsity");
        int estimatedSize = (int) (entries.length*(1.0-estimatedSparsity));
        List<T> cooEntries = new ArrayList<>(estimatedSize);
        List<Integer> rowIndices = new ArrayList<>(estimatedSize);
        List<Integer> colIndices = new ArrayList<>(estimatedSize);

        final int rows = shape.getSize(0);
        final int cols = shape.getSize(1);

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                T val = entries[rowOffset + j];

                if(!val.isZero()) {
                    cooEntries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new SparseMatrixData<T>(shape, cooEntries, rowIndices, colIndices);
    }


    /**
     * <p>Converts a dense matrix to an equivalent sparse COO matrix.
     * <p>This should only be done for matrices which are known to be sparse.
     * <p>This method will estimate the sparsity of the matrix at 99%. If a more accurate estimation is known, providing it to
     * {@link #toCoo(Shape, SemiringElement[], double)} <em>may</em> provide a slight speedup or reduce excess memory consumption. If the
     * sparsity is not known, it is recommended to simply use this method.
     *
     * @param shape Shape of the matrix.
     * @param entries Entries of the matrix.
     * @return A sparse COO matrix containing the non-zero values of the specified dense matrix.
     * @see #toCoo(Shape, SemiringElement[], double)
     */
    public static <T extends SemiringElement<T>> SparseMatrixData<T> toCoo(
            Shape shape, T[] entries) {
        return toCoo(shape, entries, 0.9);
    }


    /**
     * <p>Converts a dense tensor to an equivalent sparse COO tensor.
     * <p>This should only be done for tensors which are known to be sparse.
     * <p>If no estimate is known, then using {@link #toCooTensor(Shape, SemiringElement[])} will estimate the sparsity as 0.9.
     * @param shape Shape of the tensor.
     * @param entries Entries of the tensor.
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern it is better to underestimate the sparsity.
     * @return A sparse COO tensor containing the non-zero values of the specified dense tensor.
     */
    public static <T extends SemiringElement<T>> SparseTensorData<T> toCooTensor(
            Shape shape, T[] entries, double estimatedSparsity) {
        ValidateParameters.ensureInRange(estimatedSparsity, 0.0, 1.0, "estimatedSparsity");
        int estimatedSize = (int) (entries.length*(1.0-estimatedSparsity));
        List<T> cooEntries = new ArrayList<>(estimatedSize);
        List<int[]> cooIndices = new ArrayList<>(estimatedSize);

        for(int i=0, size=entries.length; i<size; i++) {
            T val = entries[i];

            if(!val.isZero()) {
                cooEntries.add(val);
                cooIndices.add(shape.getNdIndices(i));
            }
        }

        return new SparseTensorData<T>(shape, cooEntries, cooIndices);
    }


    /**
     * <p>Converts a dense tensor to an equivalent sparse COO tensor.
     * <p>This should only be done for matrices which are known to be sparse.
     * <p>This method will estimate the sparsity of the tensor at 99%. If a more accurate estimation is known, providing it to
     * {@link #toCoo(Shape, SemiringElement[], double)} <em>may</em> provide a slight speedup or reduce excess memory consumption. If the
     * sparsity is not known, it is recommended to simply use this method.
     *
     * @param shape Shape of the tensor.
     * @param entries Entries of the tensor.
     * @return A sparse COO tensor containing the non-zero values of the specified dense tensor.
     * @see #toCoo(Shape, SemiringElement[], double)
     */
    public static <T extends SemiringElement<T>> SparseTensorData<T> toCooTensor(
            Shape shape, T[] entries) {
        return toCooTensor(shape, entries, 0.9);
    }
}
