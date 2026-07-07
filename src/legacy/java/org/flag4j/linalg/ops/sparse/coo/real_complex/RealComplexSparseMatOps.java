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

package org.flag4j.linalg.ops.sparse.coo.real_complex;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class has low-level implementations for ops between a real sparse matrix and a complex sparse matrix.
 */
public final class RealComplexSparseMatOps {

    private RealComplexSparseMatOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Adds a real sparse matrix to a complex sparse matrix.
     * This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the sum.
     * @param src2 Second matrix in the sum.
     * @return The sum of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static CooCMatrix add(CooCMatrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<Complex128> values = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.data.length || src2Counter < src2.data.length) {

            if(src1Counter >= src1.data.length || src2Counter >= src2.data.length) {
                add1 = src2Counter >= src2.data.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                values.add(src1.data[src1Counter].add(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                values.add(src1.data[src1Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                values.add(new Complex128(src2.data[src2Counter]));
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new CooCMatrix(
                src1.shape,
                values,
                rowIndices,
                colIndices
        );
    }


    /**
     * Adds a double all items (including zero values) of a complex sparse matrix.
     * @param src Sparse matrix to add double value to.
     * @param a Double value to add to the sparse matrix.
     * @return
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} items in the matrix (including zero items).
     */
    public static CMatrix add(CooCMatrix src, double a) {
        Complex128[] sum = new Complex128[src.totalEntries().intValueExact()];
        ArrayBuilder.fill(sum, a);

        int row;
        int col;

        for(int i = 0; i<src.data.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col] = src.data[i];
        }

        return new CMatrix(src.shape, sum);
    }


    /**
     * Adds a complex number to all items (including zero values) of a real sparse matrix.
     * @param src Sparse matrix to add double value to.
     * @param a Complex value to add to the sparse matrix.
     * @return The result of the matrix addition.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} items in the matrix (including zero items).
     */
    public static CMatrix add(CooMatrix src, Complex128 a) {
        Complex128[] sum = new Complex128[src.totalEntries().intValueExact()];
        Arrays.fill(sum, a);

        int row;
        int col;

        for(int i = 0; i<src.data.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum[idx] = sum[idx].add(src.data[i]);
        }

        return new CMatrix(src.shape, sum);
    }


    /**
     * Subtracts a real sparse matrix from a complex sparse matrix. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the difference.
     * @param src2 Second matrix in the difference.
     * @return The difference of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static CooCMatrix sub(CooCMatrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<Complex128> values = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.data.length || src2Counter < src2.data.length) {

            if(src1Counter >= src1.data.length || src2Counter >= src2.data.length) {
                add1 = src2Counter >= src2.data.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                values.add(src1.data[src1Counter].sub(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                values.add(src1.data[src1Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                values.add(new Complex128(-src2.data[src2Counter]));
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new CooCMatrix(
                src1.shape,
                values,
                rowIndices,
                colIndices
        );
    }


    /**
     * Subtracts a complex sparse matrix from a real sparse matrix. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the difference.
     * @param src2 Second matrix in the difference.
     * @return The difference of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static CooCMatrix sub(CooMatrix src1, CooCMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<Complex128> values = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.data.length || src2Counter < src2.data.length) {

            if(src1Counter >= src1.data.length || src2Counter >= src2.data.length) {
                add1 = src2Counter >= src2.data.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                values.add(new Complex128(src1.data[src1Counter]).sub(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                values.add(new Complex128(src1.data[src1Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.data[src2Counter].negate());
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new CooCMatrix(
                src1.shape,
                values,
                rowIndices,
                colIndices
        );
    }


    /**
     * Adds a double all items (including zero values) of a complex sparse matrix.
     * @param src Sparse matrix to add double value to.
     * @param a Double value to add to the sparse matrix.
     * @return The result of subtracting the double value from all items of the sparse matrix.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} items in the matrix (including zero items).
     */
    public static CMatrix sub(CooCMatrix src, double a) {
        return add(src, -a);
    }


    /**
     * Adds a complex number to all items (including zero values) of a real sparse matrix.
     * @param src Sparse matrix to add double value to.
     * @param a Complex value to add to the sparse matrix.
     * @return The result of the matrix addition.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} items in the matrix (including zero items).
     */
    public static CMatrix sub(CooMatrix src, Complex128 a) {
        return add(src, a.negate());
    }


    /**
     * Multiplies two sparse matrices element-wise. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the element-wise multiplication.
     * @param src2 Second matrix in the element-wise multiplication.
     * @return The element-wise product of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static CooCMatrix elemMult(CooCMatrix src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<Complex128> product = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                product.add(src1.data[src1Counter].mult(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.

                if(src1.colIndices[src1Counter] < src2.colIndices[src2Counter]) {
                    src1Counter++;
                } else {
                    src2Counter++;
                }
            } else {
                if(src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter]) {
                    src1Counter++;
                } else {
                    src2Counter++;
                }
            }
        }

        return new CooCMatrix(
                src1.shape,
                product,
                rowIndices,
                colIndices
        );
    }


    /**
     * Adds a sparse vector to each column of a sparse matrix as if the vector is a column vector.
     * @param src The source sparse matrix.
     * @param col Sparse vector to add to each column of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code col} vector added to each row of the matrix.
     */
    public static CMatrix addToEachCol(CooMatrix src, CooCVector col) {
        ValidateParameters.ensureAllEqual(src.numRows, col.size);
        Complex128[] destEntries = new Complex128[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i = 0; i<src.data.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = new Complex128(src.data[i]);
        }

        // Add values from sparse column.
        for(int i = 0; i<col.data.length; i++) {
            int idx = col.indices[i]*src.numCols;
            int end = idx + src.numCols;
            Complex128 value = col.data[i];

            while(idx < end) {
                destEntries[idx] = destEntries[idx++].add(value);
            }
        }

        return new CMatrix(src.shape, destEntries);
    }


    /**
     * Adds a sparse vector to each row of a sparse matrix as if the vector is a row vector.
     * @param src The source sparse matrix.
     * @param row Sparse vector to add to each row of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code row} vector added to each row of the matrix.
     */
    public static CMatrix addToEachRow(CooMatrix src, CooCVector row) {
        ValidateParameters.ensureAllEqual(src.numCols, row.size);
        Complex128[] destEntries = new Complex128[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i = 0; i<src.data.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = new Complex128(src.data[i]);
        }

        // Add values from sparse column.
        for(int i = 0; i<row.data.length; i++) {
            int idx = 0;
            int colIdx = row.indices[i];
            Complex128 value = row.data[i];

            while(idx < destEntries.length) {
                destEntries[idx + colIdx] = destEntries[idx + colIdx].add(value);
                idx += src.numCols;
            }
        }

        return new CMatrix(src.shape, destEntries);
    }


    /**
     * Adds a sparse vector to each column of a sparse matrix as if the vector is a column vector.
     * @param src The source sparse matrix.
     * @param col Sparse vector to add to each column of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code col} vector added to each row of the matrix.
     */
    public static CMatrix addToEachCol(CooCMatrix src, CooVector col) {
        ValidateParameters.ensureAllEqual(src.numRows, col.size);
        Complex128[] destEntries = new Complex128[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i = 0; i<src.data.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = src.data[i];
        }

        // Add values from sparse column.
        for(int i = 0; i<col.data.length; i++) {
            int idx = col.indices[i]*src.numCols;
            int end = idx + src.numCols;
            Complex128 value = new Complex128(col.data[i]);

            while(idx < end) {
                destEntries[idx] = destEntries[idx++].add(value);
            }
        }

        return new CMatrix(src.numRows, src.numCols, destEntries);
    }


    /**
     * Adds a sparse vector to each row of a sparse matrix as if the vector is a row vector.
     * @param src The source sparse matrix.
     * @param row Sparse vector to add to each row of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code row} vector added to each row of the matrix.
     */
    public static CMatrix addToEachRow(CooCMatrix src, CooVector row) {
        ValidateParameters.ensureAllEqual(src.numCols, row.size);
        Complex128[] destEntries = new Complex128[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i = 0; i<src.data.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = src.data[i];
        }

        // Add values from sparse column.
        for(int i = 0; i<row.data.length; i++) {
            int idx = 0;
            int colIdx = row.indices[i];
            Complex128 value = new Complex128(row.data[i]);

            while(idx < destEntries.length) {
                destEntries[idx + colIdx] = destEntries[idx + colIdx].add(value);
                idx += src.numCols;
            }
        }

        return new CMatrix(src.numRows, src.numCols, destEntries);
    }
}
