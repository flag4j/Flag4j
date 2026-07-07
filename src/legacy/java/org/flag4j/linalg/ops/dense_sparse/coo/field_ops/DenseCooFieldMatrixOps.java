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

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldVector;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.Arrays;

/**
 * This class contains low-level implementations for ops between a dense and a sparse field matrix.
 */
public final class DenseCooFieldMatrixOps {

    private DenseCooFieldMatrixOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise sum of a dense matrix to a sparse COO matrix.
     * @param src1 Dense matrix in sum.
     * @param src2 Sparse COO matrix in the sum.
     * @return The element-wise sum of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldMatrix<?, ?, T> add(
            AbstractDenseFieldMatrix<?, ?, T> src1, AbstractCooFieldMatrix<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldMatrix<?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.data[idx] = dest.data[idx].add(src2.data[i]);
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
    public static <T extends FieldElement<T>> AbstractDenseFieldMatrix<?, ?, T> sub(
            AbstractDenseFieldMatrix<?, ?, T> src1, AbstractCooFieldMatrix<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        AbstractDenseFieldMatrix<?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.data[idx] = dest.data[idx].sub(src2.data[i]);
        }

        return dest;
    }


    /**
     * Subtracts a complex dense matrix from a complex sparse matrix.
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in the difference.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldMatrix<?, ?, T> sub(
            AbstractCooFieldMatrix<?, ?, ?, T> src2,
            AbstractDenseFieldMatrix<?, ?, T> src1) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] destData = src1.makeEmptyDataArray(src1.data.length);
        FieldOps.scalMult(src1.data, -1, destData);
        AbstractDenseFieldMatrix<?, ?, T> dest = src1.makeLikeNDArray(src1.shape, destData);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.data[idx] = dest.data[idx].add(src2.data[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends FieldElement<T>> void addEq(AbstractDenseFieldMatrix<?, ?, T> src1, AbstractCooFieldMatrix<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.data[idx] = src1.data[idx].add(src2.data[i]);
        }
    }


    /**
     * Subtracts a complex sparse matrix from a complex dense matrix and stores the result in the dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends FieldElement<T>> void subEq(AbstractDenseFieldMatrix<?, ?, T> src1, AbstractCooFieldMatrix<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.data[idx] = src1.data[idx].sub(src2.data[i]);
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a real sparse matrix.
     * @param shape1 Shape of the first matrix in element-wise product.
     * @param data1 Entries of the first matrix in the element-wise product.
     * @param shape2 Shape of the second matrix in the element-wise product.
     * @param data2 Non-zero items of the second matrix in the element-wise product.
     * @param rowIndices2 Non-zero row indices of the second matrix in the element-wise product.
     * @param colIndices2 Non-zero column indices of the second matrix in the element-wise product.
     * @param dest Array to store the non-zero items of the sparse COO matrix resulting from the element-wise multiplication
     * (modified). Must have same length as {@code data2}. May be the same array as {@code data2}.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}
     */
    public static <T extends FieldElement<T>> void elemMult(
            Shape shape1, T[] data1,
            Shape shape2, T[] data2, int[] rowIndices2, int[] colIndices2,
            T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        ValidateParameters.ensureArrayLengthsEq(dest.length, data2.length);

        int src1NumCols = shape1.getSize(1);

        for(int i=0, size=dest.length; i<size; i++) {
            int row = rowIndices2[i];
            int col = colIndices2[i];
            dest[i] = data1[row*src1NumCols + col].mult(data2[i]);
        }
    }


    /**
     * Computes the element-wise division between a complex sparse matrix and a complex dense matrix.
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
    public static <T extends FieldElement<T>> AbstractCooFieldMatrix<?, ?, ?, T> elemDiv(
            AbstractCooFieldMatrix<?, ?, ?, T> src1,
            AbstractDenseFieldMatrix<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        T[] quotient = src1.makeEmptyDataArray(src1.data.length);

        for(int i = 0; i<src1.data.length; i++) {
            int row = src1.rowIndices[i];
            int col = src1.colIndices[i];
            quotient[i] = src1.data[i].div(src2.data[row*src2.numCols + col]);
        }

        return src1.makeLikeNDArray(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col Vector to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of items in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldMatrix<?, ?, T> addToEachCol(
            AbstractCooFieldMatrix<?, ?, ?, T> src,
            AbstractDenseFieldVector<?, ?, T> col) {
        T[] sumEntries = src.makeEmptyDataArray(src.shape.numelIntValueExact());
        Arrays.fill(sumEntries, (col.data.length > 0) ? col.data[0].getZero() : null);
        AbstractDenseFieldMatrix<?, ?, T> sum = src.makeLikeDenseNDArray(src.shape, sumEntries);

        for(int j=0; j<sum.numCols; j++)
            sum.setCol(col.data, j);

        for(int i = 0; i<src.data.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.data[idx] = sum.data[idx].add(src.data[i]);
        }

        return sum;
    }


    /**
     * Adds a dense vector to add to each row as if the vector is a row vector.
     * @param src Source sparse matrix.
     * @param row Vector to add to each row of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each row.
     * @throws IllegalArgumentException If the number of items in the {@code col} vector does not match the number
     * of columns in the {@code src} matrix.
     */
    public static <T extends FieldElement<T>> AbstractDenseFieldMatrix<?, ?, T> addToEachRow(
            AbstractCooFieldMatrix<?, ?, ?, T> src, AbstractDenseFieldVector<?, ?, T> row) {

        T[] sumEntries = src.makeEmptyDataArray(src.shape.numelIntValueExact());
        Arrays.fill(sumEntries, (row.data.length > 0) ? row.data[0].getZero() : null);
        AbstractDenseFieldMatrix<?, ?, T> sum = src.makeLikeDenseNDArray(src.shape, sumEntries);

        for(int i=0; i<sum.numRows; i++)
            sum.setRow(row.data, i);

        for(int i = 0; i<src.data.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.data[idx] = sum.data[idx].add(src.data[i]);
        }

        return sum;
    }
}
