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

package org.flag4j.linalg.ops.dense_sparse.coo.real;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains low-level ops between a real dense and real sparse matrix.
 */
public class RealDenseSparseMatrixOps {

    private RealDenseSparseMatrixOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix in sum.
     * @param src2 Second matrix in sum.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix add(Matrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nnz; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.data[row*src1.numCols + col] += src2.data[i];
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix in difference.
     * @param src2 Second matrix in difference.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix sub(Matrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nnz; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.data[row*src1.numCols + col] -= src2.data[i];
        }

        return dest;
    }


    /**
     * Subtracts a real dense matrix from a real sparse matrix.
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in the difference.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix sub(CooMatrix src2, Matrix src1) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1.shape, RealOps.scalMult(src1.data, -1, null));

        for(int i=0; i<src2.nnz; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.data[row*src1.numCols + col] += src2.data[i];
        }

        return dest;
    }


    /**
     * Adds a real dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void addEq(Matrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nnz; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.data[row*src1.numCols + col] += src2.data[i];
        }
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in difference.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void subEq(Matrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nnz; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.data[row*src1.numCols + col] -= src2.data[i];
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a real sparse matrix.
     * @return The result of element-wise multiplication.
     * @param src1 Entries of first matrix in element-wise product.
     * @param src2 Entries of second matrix in element-wise product.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CooMatrix elemMult(Matrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int row, col;
        double[] product = new double[src2.nnz];

        for(int i=0; i<product.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            product[i] = src1.data[row*src1.numCols + col]*src2.data[i];
        }

        return CooMatrix.unsafeMake(src2.shape, product, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a real sparse matrix and a real dense matrix.
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
    public static CooMatrix elemDiv(CooMatrix src1, Matrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        double[] quotient = new double[src1.data.length];

        int row;
        int col;

        for(int i = 0; i<src1.data.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = src1.data[i] / src2.data[row*src2.numCols + col];
        }

        return CooMatrix.unsafeMake(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col Vector to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of items in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static Matrix addToEachCol(CooMatrix src, Vector col) {
        Matrix sum = new Matrix(src.numRows, src.numCols);

        for(int j=0; j<sum.numCols; j++) {
            sum.setCol(col, j);
        }

        for(int i = 0; i<src.data.length; i++) {
            sum.data[src.rowIndices[i]*src.numCols + src.colIndices[i]] += src.data[i];
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
    public static Matrix addToEachRow(CooMatrix src, Vector row) {
        Matrix sum = new Matrix(src.numRows, src.numCols);

        for(int i=0; i<sum.numRows; i++)
            sum.setRow(row, i);

        for(int i=0; i<src.nnz; i++)
            sum.data[src.rowIndices[i]*src.numCols + src.colIndices[i]] += src.data[i];

        return sum;
    }
}
