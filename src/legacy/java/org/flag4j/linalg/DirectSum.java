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

package org.flag4j.linalg;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ArrayJoiner;
import org.flag4j.util.ArrayUtils;


/**
 * Utility class for commuting the direct sum of two matrices.
 */
public final class DirectSum {

    private DirectSum() {
        // Hide default constructor for utility class.
    }

    // ------------------------------ Real Dense Matrix ------------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static Matrix directSum(Matrix A, Matrix B) {
        Matrix sum = new Matrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, i*sum.numCols, A.numCols);

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++)
            System.arraycopy(B.data, i*B.numCols, sum.data, (i + A.numRows)*sum.numCols + A.numCols, B.numCols);

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static Matrix directSum(Matrix A, CooMatrix B) {
        Matrix sum = new Matrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, i*sum.numCols, A.numCols);

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nnz; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.data[(row+A.numRows)*sum.numCols + (col+A.numCols)] = B.data[i];
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrix directSum(Matrix A, CMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int rowOffset = i*A.numCols;
            int sumRowOffset = i*sum.numCols;

            for(int j=0; j<A.numCols; j++)
                sum.data[sumRowOffset + j] = new Complex128(A.data[rowOffset + j]);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++)
            System.arraycopy(B.data, i*B.numCols, sum.data, (i+A.numRows)*sum.numCols+(A.numCols), B.numCols);

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrix directSum(Matrix A, CooCMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int rowOffset = i*A.numCols;
            int sumRowOffset = i*sum.numCols;

            for(int j=0; j<A.numCols; j++)
                sum.data[sumRowOffset + j] = new Complex128(A.data[rowOffset + j]);
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nnz; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.data[(row+A.numRows)*sum.numCols + (col+A.numCols)] = B.data[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static Matrix invDirectSum(Matrix A, Matrix B) {
        Matrix sum = new Matrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, (i+B.numRows)*sum.numCols, A.numCols);

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++)
            System.arraycopy(B.data, i*B.numCols, sum.data, i*sum.numCols+A.numCols, B.numCols);

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static Matrix invDirectSum(Matrix A, CooMatrix B) {
        Matrix sum = new Matrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, (i+B.numRows)*sum.numCols, A.numCols);

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nnz; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.data[row*sum.numCols + col + A.numCols] = B.data[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrix invDirectSum(Matrix A, CMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int aRowOffset = i*A.numCols;
            int sumRowOffset = (i+B.numRows)*sum.numCols;

            for(int j=0; j<A.numCols; j++)
                sum.data[sumRowOffset + j] = new Complex128(A.data[aRowOffset + j]);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++)
                sum.data[i*sum.numCols + j + A.numCols] = B.data[i*B.numCols + j];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrix invDirectSum(Matrix A, CooCMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int aRowOffset = i*A.numCols;
            int sumRowOffset = (i+B.numRows)*sum.numCols;

            for(int j=0; j<A.numCols; j++)
                sum.data[sumRowOffset + j] = new Complex128(A.data[aRowOffset + j]);
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nnz; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.data[row*sum.numCols + col + A.numCols] = B.data[i];
        }

        return sum;
    }
    // -------------------------------------------------------------------------------


    // ---------------------------- Complex Dense Matrix -----------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrix directSum(CMatrix A, Matrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols , sum.data, i*sum.numCols, A.numCols);

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            int bRowOffset = i*B.numCols;
            int sumRowOffset = (i+A.numRows)*sum.numCols + A.numCols;

            for(int j=0; j<B.numCols; j++)
                sum.data[sumRowOffset + j] = new Complex128(B.data[bRowOffset + j]);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrix directSum(CMatrix A, CooMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, i*sum.numCols, A.numCols);

        // Copy over second matrix.
        int row;
        int col;
        for(int i=0; i<B.nnz; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.data[(row+A.numRows)*sum.numCols + (col+A.numCols)] = new Complex128(B.data[i]);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrix directSum(CMatrix A, CMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, i*sum.numCols, A.numCols);

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++)
            System.arraycopy(B.data, i*B.numCols, sum.data, (i+A.numRows)*sum.numCols + A.numCols, B.numCols);

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrix directSum(CMatrix A, CooCMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, i*sum.numCols, A.numCols);

        // Copy over second matrix.
        int row;
        int col;
        for(int i=0; i<B.nnz; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.data[(row+A.numRows)*sum.numCols + (col+A.numCols)] = B.data[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrix invDirectSum(CMatrix A, Matrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, (i+B.numRows)*sum.numCols, A.numCols);

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            int sumRowOffset = i*sum.numCols + A.numCols;
            int bRowOffset = i*B.numCols;

            for(int j=0; j<B.numCols; j++)
                sum.data[sumRowOffset + j] = new Complex128(B.data[bRowOffset + j]);
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrix invDirectSum(CMatrix A, CooMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++)
            System.arraycopy(A.data, i*A.numCols, sum.data, (i+B.numRows)*sum.numCols, A.numCols);

        // Copy over second matrix.
        for(int i=0; i<B.nnz; i++) {
            int row = B.rowIndices[i];
            int col = B.colIndices[i];

            sum.data[row*sum.numCols + col + A.numCols] = new Complex128(B.data[i]);
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrix invDirectSum(CMatrix A, CMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int sumRowOffset = (i+B.numRows)*sum.numCols;
            int aRowOffset = i*A.numCols;

            for(int j=0; j<A.numCols; j++)
                sum.data[sumRowOffset + j] = A.data[aRowOffset + j];
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            int bRowOffset = i*B.numCols;
            int sumRowOffset = i*sum.numCols + A.numCols;

            for(int j=0; j<B.numCols; j++)
                sum.data[sumRowOffset + j] = B.data[bRowOffset + j];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrix invDirectSum(CMatrix A, CooCMatrix B) {
        CMatrix sum = new CMatrix(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int sumRowOffset = (i+B.numRows)*sum.numCols;
            int aRowOffset = i*A.numCols;

            for(int j=0; j<A.numCols; j++)
                sum.data[sumRowOffset + j] = A.data[aRowOffset + j];
        }

        // Copy over second matrix.
        for(int i=0; i<B.nnz; i++) {
            int row = B.rowIndices[i];
            int col = B.colIndices[i];

            sum.data[row*sum.numCols + col + A.numCols] = B.data[i];
        }

        return sum;
    }
    // -------------------------------------------------------------------------------


    // ------------------------------- Real COO Matrix -------------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooMatrix directSum(CooMatrix A, Matrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        System.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            indices = B.shape.getNdIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return CooMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooMatrix directSum(CooMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        System.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return CooMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooMatrix A, CMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        ArrayUtils.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        System.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            indices = B.shape.getNdIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        ArrayUtils.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        System.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooMatrix invDirectSum(CooMatrix A, Matrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        System.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayBuilder.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayJoiner.repeat(B.numRows, ArrayBuilder.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return CooMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooMatrix invDirectSum(CooMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        System.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return CooMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooMatrix A, CMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        ArrayUtils.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayBuilder.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayJoiner.repeat(B.numRows, ArrayBuilder.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        ArrayUtils.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }
    // -------------------------------------------------------------------------------


    // ------------------------------ Complex COO Matrix -----------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, Matrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        ArrayUtils.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            indices = B.shape.getNdIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        ArrayUtils.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, CMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        System.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            indices = B.shape.getNdIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(A.data, 0, destEntries, 0, A.data.length);
        System.arraycopy(B.data, 0, destEntries, A.data.length, B.data.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i = 0; i<B.data.length; i++) {
            destIdx = i+A.data.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, Matrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        ArrayUtils.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        System.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayBuilder.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayJoiner.repeat(B.numRows, ArrayBuilder.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        System.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        ArrayUtils.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        System.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, CMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        Complex128[] destEntries = new Complex128[A.data.length + B.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy items from both matrices.
        System.arraycopy(B.data, 0, destEntries, 0, B.data.length);
        System.arraycopy(A.data, 0, destEntries, B.data.length, A.data.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayBuilder.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayJoiner.repeat(B.numRows, ArrayBuilder.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return CooCMatrix.unsafeMake(destShape, destEntries, destRowIndices, destColIndices);
    }
}
