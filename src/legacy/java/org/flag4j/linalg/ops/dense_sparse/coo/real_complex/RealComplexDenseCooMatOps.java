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

package org.flag4j.linalg.ops.dense_sparse.coo.real_complex;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * This class contains low-level implementations of ops between real/complex and dense/sparse matrices.
 */
public final class RealComplexDenseCooMatOps {

    private RealComplexDenseCooMatOps() {
        // Hide private constructor for utility class.
        
    }


    /**
     * Adds a real dense matrix to a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws NDArrayShapeException If the matrices do not have the same shape.
     */
    public static CMatrix add(Matrix src1, CooCMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CMatrix dest = src1.toComplex();

        for(int i=0; i<src2.nnz; i++) {
            int row = src2.rowIndices[i];
            int col = src2.colIndices[i];
            dest.data[row*src1.numCols + col] = dest.data[row*src1.numCols + col].add(src2.data[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws NDArrayShapeException If the matrices do not have the same shape.
     */
    public static CMatrix sub(Matrix src1, CooCMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CMatrix dest = src1.toComplex();

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
     * @throws NDArrayShapeException If the matrices do not have the same shape.
     */
    public static CMatrix sub(CooCMatrix src2, Matrix src1) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CMatrix dest = new CMatrix(src1.shape, RealOps.scalMult(src1.data, -1, null));

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.data[idx] = dest.data[idx].add(src2.data[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication between a complex dense matrix and a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws NDArrayShapeException If the matrices do not have the same shape.
     */
    public static CooCMatrix elemMult(CMatrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        int row;
        int col;
        Complex128[] destEntries = new Complex128[src2.nnz];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src1.data[row*src1.numCols + col].mult(src2.data[i]);
        }

        return CooCMatrix.unsafeMake(src2.shape, destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between a complex dense matrix and a real sparse matrix.
     * @param src1 First matrix in the element-wise product.
     * @param src2 Second matrix in the element-wise product.
     * @return The result of element-wise product of {@code src1} with {@code src2}.
     * @throws NDArrayShapeException If the matrices do not have the same shape.
     */
    public static CooCMatrix elemMult(Matrix src1, CooCMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] destEntries = new Complex128[src2.nnz];

        for(int i=0, size=destEntries.length; i<size; i++) {
            int row = src2.rowIndices[i];
            int col = src2.colIndices[i];
            destEntries[i] = src2.data[i].mult(src1.data[row*src1.numCols + col]);
        }

        return CooCMatrix.unsafeMake(src2.shape, destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a real sparse matrix and a complex dense matrix.
     *
     * <p>If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     * either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * 
     *
     * <p>If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     * zero. This is done to realize computational benefits from ops with sparse matrices.
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws NDArrayShapeException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static CooCMatrix elemDiv(CooMatrix src1, CMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] quotient = new Complex128[src1.data.length];
        int row;
        int col;

        for(int i = 0; i<src1.data.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = new Complex128(src1.data[i]).div(src2.data[row*src2.numCols + col]);
        }

        return CooCMatrix.unsafeMake(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }
}
