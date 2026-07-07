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

package org.flag4j.linalg.ops.dense_sparse.coo.field_ops;


import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldTensor;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldTensor;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This class contains methods to apply common ops to a dense/sparse field matrix and to a sparse/dense field matrix.
 */
public final class DenseCooFieldTensorOps {

    private DenseCooFieldTensorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes element-wise addition of a dense tensor with a sparse COO tensor.
     * @param src1 Dense tensor in sum.
     * @param src2 Sparse COO tensor in sum.
     * @return The result of the element-wise subtraction.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldTensor<?, T> add(
            AbstractDenseFieldTensor<?, T> src1,
            AbstractCooFieldTensor<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src1.data.clone();
        
        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.get1DIndex(src2.indices[i]);
            destEntries[idx] = destEntries[idx].add(src2.data[i]);
        }

        return src1.makeLikeNDArray(src1.shape, destEntries);
    }


    /**
     * Computes element-wise addition of a dense tensor with a sparse COO tensor.
     * @param src1 Dense tensor in sum.
     * @param src2 Sparse COO tensor in sum.
     * @return The result of the element-wise addition.
     */
    public static <T extends FieldElement<T>> void addEq(
            AbstractDenseFieldTensor<?, T> src1,
            AbstractCooFieldTensor<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            src1.data[src2.shape.get1DIndex(src2.indices[i])] =
                    src1.data[src2.shape.get1DIndex(src2.indices[i])].add(src2.data[i]);
        }
    }


    /**
     * Computes the element-wise tensor a complex sparse tensor from a complex dense tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise tensor subtraction.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldTensor<?, T> sub(
            AbstractDenseFieldTensor<?, T> src1, 
            AbstractCooFieldTensor<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src1.data.clone();

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.get1DIndex(src2.indices[i]);
            destEntries[idx] =
                    destEntries[idx].sub(src2.data[i]);
        }

        return src1.makeLikeNDArray(src1.shape, destEntries);
    }


    /**
     * Subtracts a complex dense tensor from a complex sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldTensor<?, T> sub(
            AbstractCooFieldTensor<?, ?, T> src1, AbstractDenseFieldTensor<?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldTensor<?, T> dest = src2.mult(-1);

        for(int i=0, size=src1.nnz; i<size; i++) {
            dest.data[src1.shape.get1DIndex(src1.indices[i])] =
                    dest.data[src1.shape.get1DIndex(src1.indices[i])].add(src1.data[i]);
        }

        return dest;
    }


    /**
     * Computes element-wise subtraction of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise subtraction.
     */
    public static <T extends FieldElement<T>> void subEq(AbstractDenseFieldTensor<?, T> src1, AbstractCooFieldTensor<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            src1.data[src2.shape.get1DIndex(src2.indices[i])] =
                    src1.data[src2.shape.get1DIndex(src2.indices[i])].sub(src2.data[i]);
        }
    }


    /**
     * Computes the element-wise tensor multiplication between a complex dense tensor and a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return THe result of the element-wise tensor multiplication.
     */
    public static <T extends FieldElement<T>> AbstractCooFieldTensor<?, ?, T> elemMult(
            AbstractDenseFieldTensor<?, T> src1, AbstractCooFieldTensor<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src1.makeEmptyDataArray(src2.nnz);
        int[][] indices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy2D(src2.indices, indices);

        for(int i=0, size=destEntries.length; i<size; i++)
            destEntries[i] = src1.data[src2.shape.get1DIndex(src2.indices[i])].mult((T) src2.data[i]);

        return src2.makeLikeTensor(src2.shape, destEntries, indices);
    }


    /**
     * Adds a scalar to a complex sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldTensor<?, T> add(AbstractCooFieldTensor<?, ?, T> src1, FieldElement<T> b) {
        T[] sumEntries = src1.makeEmptyDataArray(src1.shape.numelIntValueExact());
        Arrays.fill(sumEntries, b);
        AbstractDenseFieldTensor<?, T> sum = src1.makeLikeDenseTensor(src1.shape, sumEntries);

        for(int i=0, size=src1.nnz; i<size; i++) {
            sum.data[src1.shape.get1DIndex(src1.indices[i])] =
                    sum.data[src1.shape.get1DIndex(src1.indices[i])].add(src1.data[i]);
        }

        return sum;
    }


    /**
     * Subtracts a scalar from a sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldTensor<?, T> sub(AbstractCooFieldTensor<?, ?, T> src1, FieldElement<T> b) {
        T[] sumEntries = src1.makeEmptyDataArray(src1.shape.numelIntValueExact());
        Arrays.fill(sumEntries, b);
        AbstractDenseFieldTensor<?, T> sum = src1.makeLikeDenseTensor(src1.shape, sumEntries);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            sum.data[idx].add(src1.data[i]);
        }

        return sum;
    }


    /**
     * Computes the element-wise division between a complex dense tensor and a complex sparse tensor.
     * @param src1 Complex sparse tensor.
     * @param src2 Complex dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends FieldElement<T>> AbstractCooFieldTensor<?, ?, T> elemDiv(
            AbstractCooFieldTensor<?, ?, T> src1,
            AbstractDenseFieldTensor<?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src1.makeEmptyDataArray(src1.nnz);
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy2D(src1.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.get1DIndex(src1.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.data[index].div(src2.data[i]);
        }

        return src1.makeLikeTensor(src2.shape, destEntries, destIndices);
    }
}
