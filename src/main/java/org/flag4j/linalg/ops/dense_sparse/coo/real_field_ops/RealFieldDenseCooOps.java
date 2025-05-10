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


import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldTensor;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains methods to apply common binary ops to a real/field dense matrix and to a field/real sparse matrix.
 */
public final class RealFieldDenseCooOps {

    private RealFieldDenseCooOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a complex sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of element-wise multiplication between the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> AbstractCooFieldTensor<?, ?, T> elemMult(Tensor src1, AbstractCooFieldTensor<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        T[] destEntries = (T[]) new Field[src2.nnz];
        int[][] destIndices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy2D(src2.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.get1DIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src2.data[i].mult(src1.data[index]);
        }

        return src2.makeLikeTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a complex sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Complex sparse tensor.
     * @param destEntries Array to store non-zero data resulting from the element-wise product.
     * Assumed to have length {@code src2.nnz}.
     * @param destIndices Array to store non-zero indices resulting from the element-wise product.
     * Assumed to have size {@code src2.nnz}&times;{@code src2.getRank()}.
     * @return The result of element-wise multiplication between the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> void elemMult(
            AbstractDenseFieldTensor<?, T> src1, CooTensor src2,
            T[] destEntries, int[][] destIndices) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        ArrayUtils.deepCopy2D(src2.indices, destIndices); // Copy over indices.

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.get1DIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.data[index].mult(src2.data[i]);
        }
    }


    /**
     * Computes the element-wise division between a real dense tensor and a real sparse tensor.
     * @param src1 Real sparse tensor.
     * @param src2 Real dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> AbstractCooFieldTensor<?, ?, T> elemDiv(AbstractCooFieldTensor<?, ?, T> src1, Tensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = (T[]) new Field[src1.nnz];
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy2D(src1.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.get1DIndex(src1.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.data[index].div(src2.data[i]);
        }

        return src1.makeLikeTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Adds a dense complex tensor to a real sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result of the tensor addition.
     */
    public static <T extends Field<T>> AbstractDenseFieldTensor<?, T> add(
            AbstractDenseFieldTensor<?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src1.data.clone();

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src2.indices[i]);
            destEntries[idx] = destEntries[idx].add(src2.data[i]);
        }

        return src1.makeLikeNDArray(src2.shape, destEntries);
    }


    /**
     * Adds a dense complex tensor to a real sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result of the tensor addition.
     */
    public static <T extends Field<T>> AbstractDenseFieldTensor<?, T> sub(AbstractDenseFieldTensor<?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src1.data.clone();

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src2.indices[i]);
            destEntries[idx] = destEntries[idx].sub(src2.data[i]);
        }

        return src1.makeLikeNDArray(src1.shape, destEntries);
    }


    /**
     * Computes element-wise addition between a complex dense tensor and a real sparse tensor. The result is stored
     * in the complex dense tensor.
     * @param src1 The complex dense tensor. Also, the storage for the computation.
     * @param src2 The real sparse tensor.
     */
    public static <T extends Field<T>> void addEq(AbstractDenseFieldTensor<?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.get1DIndex(src2.indices[i]);
            src1.data[idx] = src1.data[idx].add(src2.data[i]);
        }
    }


    /**
     * Computes element-wise subtraction between a complex dense tensor and a real sparse tensor. The result is stored
     * in the complex dense tensor.
     * @param src1 The complex dense tensor. Also, the storage for the computation.
     * @param src2 The real sparse tensor.
     */
    public static <T extends Field<T>> void subEq(AbstractDenseFieldTensor<?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.get1DIndex(src2.indices[i]);
            src1.data[idx] = src1.data[idx].sub(src2.data[i]);
        }
    }


    /**
     * Subtracts a complex dense tensor from a real sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static <T extends Field<T>> AbstractDenseFieldTensor<?, T> sub(CooTensor src1, AbstractDenseFieldTensor<?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldTensor<?, T> dest = src2.mult(-1);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.get1DIndex(src1.indices[i]);
            dest.data[idx] = dest.data[idx].add(src1.data[i]);
        }

        return dest;
    }
}
