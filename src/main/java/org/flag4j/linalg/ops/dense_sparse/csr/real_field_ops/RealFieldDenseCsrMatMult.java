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

package org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractCsrFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;

import java.util.Arrays;

/**
 * This class contains low-level implementations of real-complex sparse-dense matrix multiplication where the sparse matrix
 * is in CSR format.
 */
public final class RealFieldDenseCsrMatMult {

    private RealFieldDenseCsrMatMult() {
        // Hide default constructor for utility method.
    }


    /**
     * <p>Computes the matrix multiplication between a real sparse CSR matrix and a complex dense matrix.
     * <p>WARNING: If the first matrix is very large but not very sparse, this method may be slower than converting the
     * first matrix to a dense matrix and computing a dense-dense matrix multiplication.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws ArrayShapeException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> standard(
            CsrMatrix src1, AbstractDenseFieldMatrix<?, ?, T> src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        T[] destEntries = src2.makeEmptyDataArray(src1.numRows*src2.numCols);
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;
        int cols2 = src2.numCols;

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*src2.numCols;
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int innerStop = rowOffset + cols2;

            for(int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.data[aIndex];
                int src2Idx = aCol*src2.numCols;
                int destIdx = rowOffset;

                while(destIdx < innerStop) {
                    destEntries[destIdx] = destEntries[destIdx++].add(src2.data[src2Idx++].mult(aVal));
                }
            }
        }

        return src2.makeLikeNDArray(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between a real sparse CSR matrix and the transpose of a complex dense matrix.
     * WARNING: This method is likely slower than {@code standard(src1, src2.T())} unless
     * {@code src1} has many more columns than rows and is very sparse.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication. Will be implicitly transposed.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same number of rows.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> standardTranspose(
            CsrMatrix src1, AbstractDenseFieldMatrix<?, ?, T> src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureAllEqual(src1.numCols, src2.numCols);

        T[] destEntries = src2.makeEmptyDataArray(src1.numRows*src2.numRows);
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;
        int rows2 = src2.numRows;
        int src2RowOffset;
        int destRowOffset;
        int start;
        int stop;

        for(int k=0; k<rows2; k++) {
            src2RowOffset = k*src2.numCols;

            for(int i=0; i<rows1; i++) {
                destRowOffset = i*src2.numRows + k;
                start = src1.rowPointers[i];
                stop = src1.rowPointers[i+1];

                while(start < stop) {
                    destEntries[destRowOffset] = destEntries[destRowOffset].add(
                            src2.data[src2RowOffset + src1.colIndices[start]].mult(src1.data[start++]));
                }
            }
        }

        return src2.makeLikeNDArray(new Shape(src1.numRows, src2.numRows), destEntries);
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a real sparse CSR matrix.
     * WARNING: If the second matrix is very large but not very sparse, this method may be slower than converting the
     * second matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link Matrix#mult(Matrix)}.
     * @param src1 First matrix in the matrix multiplication (dense matrix).
     * @param src2 Second matrix in the matrix multiplication (sparse CSR matrix).
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws ArrayShapeException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> standard(
            Matrix src1, 
            AbstractCsrFieldMatrix<?, ?, ?, T> src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        T[] destEntries = src2.makeEmptyDataArray(src1.numRows * src2.numCols);
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;
        int cols1 = src1.numCols;
        int cols2 = src2.numCols;

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            for(int j=0; j<cols1; j++) {
                double src1Val = src1.data[src1RowOffset + j];
                int start = src2.rowPointers[j];
                int stop = src2.rowPointers[j + 1];

                for(int aIndex=start; aIndex<stop; aIndex++) {
                    int aCol = src2.colIndices[aIndex];
                    T aVal = src2.data[aIndex];
                    destEntries[rowOffset + aCol] = destEntries[rowOffset + aCol].add(aVal.mult(src1Val));
                }
            }
        }

        return src2.makeLikeDenseTensor(new Shape(rows1, cols2), destEntries);
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a real sparse CSR matrix.
     * WARNING: If the second matrix is very large but not very sparse, this method may be slower than converting the
     * second matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link Matrix#mult(Matrix)}.
     * @param src1 First matrix in the matrix multiplication (dense matrix).
     * @param src2 Second matrix in the matrix multiplication (sparse CSR matrix).
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws ArrayShapeException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> standard(
            AbstractDenseFieldMatrix<?, ?, T> src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        T[] destEntries = src1.makeEmptyDataArray(src1.numRows * src2.numCols);
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;
        int cols1 = src1.numCols;
        int cols2 = src2.numCols;

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            for(int j=0; j<cols1; j++) {
                T src1Val = src1.data[src1RowOffset + j];
                int start = src2.rowPointers[j];
                int stop = src2.rowPointers[j + 1];

                for(int aIndex=start; aIndex<stop; aIndex++) {
                    int aCol = src2.colIndices[aIndex];
                    double aVal = src2.data[aIndex];
                    destEntries[rowOffset + aCol] = destEntries[rowOffset + aCol].add(src1Val.mult(aVal));
                }
            }
        }

        return src1.makeLikeNDArray(new Shape(rows1, cols2), destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex dense vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the length of
     * {@code src2}.
     */
    public static <T extends Field<T>> AbstractDenseFieldVector<?, ?, T> standardVector(
            CsrMatrix src1, AbstractDenseFieldVector<?, ?, T> src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);

        T[] destEntries = src2.makeEmptyDataArray(src1.numRows);
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;

        for (int i = 0; i < rows1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex = start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.data[aIndex];
                destEntries[i] = destEntries[i].add(src2.data[aCol].mult(aVal));
            }
        }

        return src2.makeLikeTensor(destEntries);
    }


    /**
     * Computes the matrix multiplication between a complex sparse CSR matrix and a real dense matrix.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws ArrayShapeException If {@code src1} does not have the same number of columns as
     * {@code src2} has rows.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> standard(
            AbstractCsrFieldMatrix<?, ?, ?, T> src1, Matrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);
        
        T[] destEntries = src1.makeEmptyDataArray(src1.numRows*src2.numCols);
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;
        int cols2 = src2.numCols;

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*src2.numCols;
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int innerStop = rowOffset + cols2;

            for(int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                T aVal = src1.data[aIndex];
                int src2Idx = aCol*src2.numCols;
                int destIdx = rowOffset;

                while(destIdx < innerStop)
                    destEntries[destIdx] = destEntries[destIdx++].add(aVal.mult(src2.data[src2Idx++]));
            }
        }

        return src1.makeLikeDenseTensor(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex dense vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the length of
     * {@code src2}.
     */
    public static <T extends Field<T>> AbstractDenseFieldVector<?, ?, T> standardVector(
            AbstractCsrFieldMatrix<?, ?, ?, T> src1, Vector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);
        
        T[] destEntries = (T[]) new Field[src1.numRows];
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;

        for (int i = 0; i < rows1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex = start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                T aVal = src1.data[aIndex];

                destEntries[i] = destEntries[i].add(aVal.mult(src2.data[aCol]));
            }
        }

        return src1.makeLikeDenseTensor(new Shape(1, src1.numRows), destEntries).toVector();
    }
}
