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


import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;

/**
 * This class contains low-level implementations of ops between real/field and dense/sparse matrices.
 */
public final class RealFieldDenseCooMatrixOps {

    private RealFieldDenseCooMatrixOps() {
        // Hide private constructor for utility class.
        
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> add(
            AbstractDenseFieldMatrix<?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldMatrix<?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.data[idx] = dest.data[idx].add(src2.data[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> sub(AbstractDenseFieldMatrix<?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldMatrix<?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.data[idx] = dest.data[idx].sub(src2.data[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> sub(CooMatrix src1, AbstractDenseFieldMatrix<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src2.shape, src1.shape);

        T[] destData = src2.makeEmptyDataArray(src2.data.length);
        FieldOps.scalMult(src2.data, -1, destData);
        AbstractDenseFieldMatrix<?, ?, T> dest = src2.makeLikeNDArray(src2.shape, destData);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.rowIndices[i]*src2.numCols + src1.colIndices[i];
            dest.data[idx] = dest.data[idx].add(src1.data[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> void addEq(AbstractDenseFieldMatrix<?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.data[idx] = src1.data[idx].add(src2.data[i]);
        }
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> void subEq(AbstractDenseFieldMatrix<?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.data[idx] = src1.data[idx].sub(src2.data[i]);
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> AbstractCooFieldMatrix<?, ?, ?, T> elemMult(
            Matrix src1, AbstractCooFieldMatrix<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destEntries = src2.makeEmptyDataArray(src2.nnz);

        for(int i=0; i<destEntries.length; i++) {
            int row = src2.rowIndices[i];
            int col = src2.colIndices[i];
            destEntries[i] = src2.data[i].mult(src1.data[row*src1.numCols + col]);
        }

        return src2.makeLikeTensor(src2.shape, destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between two matrices.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @param destEntries Array to store non-zero values resulting from the element-wise product. Assumed to have length
     * {@code src2.nnz}.
     * @return The result of element-wise multiplication.
     * @throws ArrayShapeException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> void elemMult(
            AbstractDenseFieldMatrix<?, ?, T> src1, CooMatrix src2,
            Field<T>[] destEntries) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            int row = src2.rowIndices[i];
            int col = src2.colIndices[i];
            destEntries[i] = src1.data[row*src1.numCols + col].mult(src2.data[i]);
        }
    }


    /**
     * Computes the element-wise division between a complex sparse matrix and a real dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * 
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from ops with sparse matrices.
     * 
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static <T extends Field<T>> AbstractCooFieldMatrix<?, ?, ?, T> elemDiv(AbstractCooFieldMatrix<?, ?, ?, T> src1, Matrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] quotient = src1.makeEmptyDataArray(src1.data.length);

        for(int i = 0; i<src1.data.length; i++) {
            int row = src1.rowIndices[i];
            int col = src1.colIndices[i];
            quotient[i] = src1.data[i].div(src2.data[row*src2.numCols + col]);
        }

        return src1.makeLikeTensor(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }
}
