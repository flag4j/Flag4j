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

package org.flag4j.linalg.ops.dense_sparse.coo.real_complex;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains methods to apply common binary ops to a real/complex dense matrix and to a complex/real sparse matrix.
 */
public final class RealComplexDenseCooOps {

    private RealComplexDenseCooOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise division between a real dense tensor and a complex sparse tensor.
     * @param src1 Real sparse tensor.
     * @param src2 complex dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemDiv(CooTensor src1, CTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] destEntries = new Complex128[src1.nnz];
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy2D(src1.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.get1DIndex(src1.indices[i]); // Get the index of non-zero entry.
            destEntries[i] = new Complex128(src1.data[index]).div(src2.data[i]);
        }

        return CooCTensor.unsafeMake(src2.shape, destEntries, destIndices);
    }


    /**
     * Computes element-wise sum between a real dense tensor to a sparse COO complex tensor.
     * @param shape1 Shape of the first tensor.
     * @param src1 Entries of the first tensor.
     * @param shape2 Shape of the COO tensor.
     * @param src2 Non-zero data of the COO tensor.
     * @param indices Non-zero indices of the COO tensor.
     * @param dest Array to store the dense result of the element-wise sum. Must be at least as large as {@code src1}.
     * May be {@code null} or the same array as {@code src1}.
     * @throws IllegalArgumentException If {@code !shape1.equals(shape2)}.
     */
    public static void add(Shape shape1, double[] src1,
                           Shape shape2, Complex128[] src2, int[][] indices,
                           Complex128[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        ArrayConversions.toComplex128(src1, dest);

        for(int i=0, size=src2.length; i<size; i++) {
            int idx = shape2.get1DIndex(indices[i]);
            dest[idx] = src2[i].add(dest[idx]);
        }
    }


    /**
     * Subtracts a sparse complex tensor from a real dense tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensor sub(Tensor src1, CooCTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] entries = ArrayConversions.toComplex128(src1.data, null);

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.get1DIndex(src2.indices[i]);
            entries[idx] = entries[idx].sub(src2.data[i]);
        }

        return new CTensor(src1.shape, entries);
    }


    /**
     * Computes the element-wise multiplication between a complex dense tensor and a real sparse matrix.
     * @param src1 First tensor in the element-wise multiplication.
     * @param src2 Second tensor in the element-wise multiplication.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemMult(CTensor src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        Complex128[] destEntries = new Complex128[src2.nnz];

        int[][] indices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy2D( src2.indices, indices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            destEntries[i] = src1.data[src2.shape.get1DIndex(src2.indices[i])].mult(src2.data[i]);
        }

        return CooCTensor.unsafeMake(src2.shape, destEntries, indices);
    }


    /**
     * Computes the element-wise multiplication between a complex dense tensor and a real sparse matrix.
     * @param src1 First tensor in the element-wise multiplication.
     * @param src2 Second tensor in the element-wise multiplication.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemMult(Tensor src1, CooCTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] destEntries = new Complex128[src2.nnz];

        int[][] indices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy2D(src2.indices, indices);

        for(int i=0, size=destEntries.length; i<size; i++)
            destEntries[i] = src2.data[i].mult(src1.data[src2.shape.get1DIndex(src2.indices[i])]);

        return CooCTensor.unsafeMake(src2.shape, destEntries, indices);
    }


    /**
     * Subtracts a real dense tensor from a complex sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensor sub(CooCTensor src1, Tensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        Complex128[] entries = ArrayConversions.toComplex128(
                RealOps.scalMult(src2.data, -1.0, null), null);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            entries[idx] = src1.data[i].add(entries[idx].re);
        }

        return new CTensor(src1.shape, entries);
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensor add(CooTensor src1, Complex128 b) {
        CTensor sum = new CTensor(src1.shape, b);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            sum.data[idx].add(src1.data[i]);
        }

        return sum;
    }


    /**
     * Subtracts a scalar from a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensor sub(CooTensor src1, Complex128 b) {
        CTensor sum = new CTensor(src1.shape, b);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            sum.data[idx].add(src1.data[i]);
        }

        return sum;
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensor sub(CooCTensor src1, double b) {
        CTensor sum = new CTensor(src1.shape, b);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            sum.data[idx] = sum.data[idx].sub(src1.data[i]);
        }

        return sum;
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensor add(CooCTensor src1, double b) {
        CTensor sum = new CTensor(src1.shape, b);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            sum.data[idx] = sum.data[idx].add(src1.data[i]);
        }

        return sum;
    }
}