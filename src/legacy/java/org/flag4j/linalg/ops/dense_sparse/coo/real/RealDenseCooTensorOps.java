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

import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains methods to apply common binary ops to a real dense matrix and to a real sparse matrix.
 */
public class RealDenseCooTensorOps {

    private RealDenseCooTensorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Adds a real dense tensor to a real sparse tensor.
     * @param src1 First tensor in sum.
     * @param src2 Second tensor in sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static Tensor add(Tensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int[] indices;
        Tensor dest = src1.copy();

        for(int i=0, size=src2.nnz; i<size; i++) {
            indices = src2.indices[i];
            dest.data[dest.shape.get1DIndex(indices)] += src2.data[i];
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a real sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooTensor elemMult(Tensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int index;
        double[] destEntries = new double[src2.nnz];
        int[][] destIndices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy2D(src2.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            index = src2.shape.get1DIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.data[index]*src2.data[i];
        }

        return CooTensor.unsafeMake(src2.shape, destEntries, destIndices);
    }


    /**
     * Computes the element-wise division between a real dense tensor and a real sparse tensor.
     * @param src1 Real sparse tensor.
     * @param src2 Real dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooTensor elemDiv(CooTensor src1, Tensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        int index;
        double[] destEntries = new double[src1.nnz];
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy2D(src1.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            index = src2.shape.get1DIndex(src1.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.data[index]/src2.data[i];
        }

        return CooTensor.unsafeMake(src2.shape, destEntries, destIndices);
    }


    /**
     * Subtracts a real sparse tensor from a real dense tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static Tensor sub(Tensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Tensor dest = src1.copy();

        for(int i=0, size=src2.nnz; i<size; i++)
            dest.data[dest.shape.get1DIndex(src2.indices[i])] -= src2.data[i];
        
        return dest;
    }


    /**
     * Subtracts a real dense tensor from a real sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static Tensor sub(CooTensor src1, Tensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Tensor dest = src2.mult(-1);

        for(int i=0, size=src1.nnz; i<size; i++)
            dest.data[src1.shape.get1DIndex(src1.indices[i])] += src1.data[i];
        

        return dest;
    }


    /**
     * Adds a real dense tensor to a real sparse tensor and stores the result in the first tensor.
     * @param src1 First tensor in sum. Also, storage of result.
     * @param src2 Second tensor in sum.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static void addEq(Tensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++)
            src1.data[src1.shape.get1DIndex(src2.indices[i])] += src2.data[i];
    }


    /**
     * Subtracts a real sparse tensor from a real dense tensor and stores the result in the dense tensor.
     * @param src1 First tensor in difference. Also, storage of result.
     * @param src2 Second tensor in difference.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static void subEq(Tensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++)
            src1.data[src1.shape.get1DIndex(src2.indices[i])] -= src2.data[i];
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static Tensor add(CooTensor src1, double b) {
        Tensor sum = new Tensor(src1.shape, b);

        for(int i=0, size=src1.nnz; i<size; i++)
            sum.data[src1.shape.get1DIndex(src1.indices[i])] += src1.data[i];

        return sum;
    }


    /**
     * Subtracts a scalar from each entry of a real sparse COO tensor.
     * @param src1 Sparse tensor in difference.
     * @param b Scalar in difference.
     * @return A dense tensor which is the difference of {@code src1} and {@code b} such that {@code b} is subtracted from each
     * element of {@code src1}.
     */
    public static Tensor sub(CooTensor src1, double b) {
        Tensor sum = new Tensor(src1.shape, b);

        for(int i=0, size=src1.nnz; i<size; i++)
            sum.data[src1.shape.get1DIndex(src1.indices[i])] -= src1.data[i];

        return sum;
    }
}
