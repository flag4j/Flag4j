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

package org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops;


import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldVector;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.Arrays;

/**
 * This class provides low level methods for computing ops between a real/field dense/sparse vector and a
 * field/real sparse/dense vector.
 */
public final class RealFieldDenseCooVectorOps {

    private RealFieldDenseCooVectorOps() {
        // Hide default constructor in utility class.
    }


    /**
     * Computes the vector inner product between a real dense vector and a complex sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero data).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of data in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> T inner(double[] src1, T[] src2, int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        T innerProd = (src2.length > 0) ? src2[0].getZero() : null;

        for(int i=0; i<src2.length; i++) {
            int index = indices[i];
            innerProd = innerProd.add(src2[i].conj().mult(src1[index]));
        }

        return innerProd;
    }


    /**
     * Computes the vector inner product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero data).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of data in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> T inner(T[] src1, double[] src2, int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        T innerProd = (src1.length > 0) ? src1[0].getZero() : null;

        for(int i=0; i<src2.length; i++) {
            int index = indices[i];
            innerProd = innerProd.add(src1[index].mult(src2[i]));
        }

        return innerProd;
    }


    /**
     * Computes the vector inner product between a complex dense vector and a real sparse vector.
     * @param src1 Non-zero data of the sparse vector.
     * @param src2 Entries of the dense vector.
     * @param indices Indices of non-zero data in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero data).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of data in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> T inner(double[] src1, int[] indices, int sparseSize, T[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src2.length, sparseSize);
        T innerProd = (src2.length > 0) ? src2[0].getZero() : null;

        for(int i=0; i<src1.length; i++)
            innerProd = innerProd.add(src2[indices[i]].conj().mult(src1[i]));

        return innerProd;
    }


    /**
     * Computes the vector outer product between a real dense vector and a complex sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero data.
     * @param dest Array to store the result of the vector outer product in (modified). Must have length
     * {@code src1.length*sparseSize}.
     */
    public static <T extends Field<T>> void outerProduct(double[] src1, T[] src2, int[] indices, int sparseSize, T[] dest) {
        T zero = (src2.length > 0) ? src2[0].getZero() : null;
        Arrays.fill(dest, zero);

        for(int i=0, size=src1.length; i<size; i++) {
            int destIdx = i*src1.length;
            double val1 = src1[i];

            for(int j=0, size2=src2.length; j<size2; j++)
                dest[destIdx + indices[j]] = src2[j].conj().mult(val1);
        }
    }


    /**
     * Computes the vector outer product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero data.
     * @param dest Array to store the result of the vector outer product (modified). Must have length {@code sparseSize*src1.length}.
     */
    public static <T extends Field<T>> void outerProduct(T[] src1, double[] src2, int[] indices, int sparseSize, T[] dest) {
        T zero = (src1.length > 0) ? src1[0].getZero() : null;
        Arrays.fill(dest, zero);

        for(int i=0, size=src1.length; i<size; i++) {
            int destIdx = i*sparseSize;
            T val1 = src1[i];

            for(int j=0, size2=src2.length; j<size2; j++)
                dest[destIdx + indices[j]] = val1.mult(src2[j]);
        }
    }


    /**
     * Computes the vector outer product between a complex dense vector and a real sparse vector.
     * @param src1 Non-zero data of the real sparse vector.
     * @param indices Indices of non-zero data of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero data.
     * @param src2 Entries of the complex dense vector.
     * @param dest Array to store the result of the vector outer product in (modified).
     * Must have length {@code src2.length*sparseSize}.
     */
    public static <T extends Field<T>> void outerProduct(double[] src1, int[] indices, int sparseSize, T[] src2, T[] dest) {
        ValidateParameters.ensureAllEqual(sparseSize, src2.length);
        T zero = (src2.length > 0) ? src2[0].getZero() : null;
        Arrays.fill(dest, zero);
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = indices[i]*src2.length;

            for(T v : src2)
                dest[destIndex++] = v.mult(src1[i]);
        }
    }


    /**
     * Computes the vector outer product between a rea; dense vector and a complex sparse vector.
     * @param src1 Non-zero data of the complex sparse vector.
     * @param indices Indices of non-zero data of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero data.
     * @param src2 Entries of the real dense vector.
     * @param dest Array to store the vector outer product in (modified). Must have length {@code src2.length*sparseSize}.
     */
    public static <T extends Field<T>> T[] outerProduct(T[] src1, int[] indices, int sparseSize, double[] src2, T[] dest) {
        T zero = (src1.length > 0) ? src1[0].getZero() : null;
        Arrays.fill(dest, zero);

        for(int i=0; i<src2.length; i++) {
            int destIdx = i*sparseSize;

            for(int j=0; j<src1.length; j++)
                dest[destIdx + indices[j]] = src1[j].mult(src2[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> AbstractDenseFieldVector<?, ?, T> add(AbstractDenseFieldVector<?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldVector<?, ?, T> dest = src1.copy();

        for(int i = 0; i<src2.data.length; i++) {
            int index = src2.indices[i];
            dest.data[index] = dest.data[index].add(src2.data[i]);
        }

        return dest;
    }


    /**
     * Subtracts a complex dense vector from a complex sparse vector.
     * @param src1 Sparse vector.
     * @param src2 Dense vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> AbstractDenseFieldVector<?, ?, T> sub(CooVector src1, AbstractDenseFieldVector<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        T[] destData = src2.makeEmptyDataArray(src2.data.length);
        FieldOps.scalMult(src2.data, -1, destData);
        AbstractDenseFieldVector<?, ?, T> dest = src2.makeLikeNDArray(src2.shape, destData);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.indices[i];
            dest.data[idx] = dest.data[idx].add(src1.data[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication of a real dense vector with a complex sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static <T extends Field<T>> AbstractCooFieldVector<?, ?, ?, ?, T> elemMult(
            Vector src1, AbstractCooFieldVector<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] dest = src2.makeEmptyDataArray(src2.nnz);

        for(int i=0, size=src2.nnz; i<size; i++)
            dest[i] = src2.data[i].mult(src1.data[src2.indices[i]]);

        return src2.makeLikeTensor(src1.shape, dest, src2.indices.clone());
    }


    /**
     * Computes the element-wise product between a dense {@link Field} vector and a real COO vector.
     * @param src1 Dense {@link Field} vector in element-wise product.
     * @param src2 Real COO vector in element-wise product.
     * @return The non-zero data of the element-wise product of {@code src1} and {@code src2}.
     * @throws NDArrayShapeException If {@code !src1.shape.equals(src2.shape)}
     */
    public static <T extends Field<T>> T[] elemMult(
            AbstractDenseFieldVector<?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] dest = src1.makeEmptyDataArray(src2.data.length);

        for(int i = 0, size=src2.data.length; i<size; i++) {
            int idx = src2.indices[i];
            dest[i] = src1.data[idx].mult(src2.data[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> AbstractDenseFieldVector<?, ?, T> sub(
            AbstractDenseFieldVector<?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldVector<?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            dest.data[index] = dest.data[index].sub(src2.data[i]);
        }

        return dest;
    }


    /**
     * Computes the vector addition between a dense complex vector and a sparse real vector. The result is stored in
     * the first vector.
     * @param src1 First vector to add. Also, where the result is stored.
     * @param src2 Second vector to add.
     * @throws IllegalArgumentException If the vectors do not have the same size.
     */
    public static <T extends Field<T>> void addEq(AbstractDenseFieldVector<?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            src1.data[index] = src1.data[index].add(src2.data[i]);
        }
    }


    /**
     * Computes the vector subtraction between a dense complex vector and a real sparse vector. The result is stored in
     * the first vector.
     * @param src1 First vector in subtraction.
     * @param src2 Second vector in subtraction.
     */
    public static <T extends Field<T>> void subEq(AbstractDenseFieldVector<?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            src1.data[index] = src1.data[index].sub(src2.data[i]);
        }
    }


    /**
     * Compute the element-wise division between a sparse vector and a dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static <T extends Field<T>> AbstractCooFieldVector<?, ?, ?, ?, T> elemDiv(
            AbstractCooFieldVector<?, ?, ?, ?, T> src1, Vector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] dest = src1.makeEmptyDataArray(src1.data.length);

        for(int i = 0, size = src1.data.length; i<size; i++)
            dest[i] = src1.data[i].div(src2.data[src1.indices[i]]);

        return src1.makeLikeTensor(src1.shape, dest, src1.indices.clone());
    }
}
