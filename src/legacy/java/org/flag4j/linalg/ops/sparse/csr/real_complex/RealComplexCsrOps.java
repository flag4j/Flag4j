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

package org.flag4j.linalg.ops.sparse.csr.real_complex;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * This class contains low-level implementations for element-wise ops on real/complex CSR matrices.
 */
public final class RealComplexCsrOps {

    private RealComplexCsrOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrix CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to dense matrix and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to {@code src1} and {@code src2}.
     * @param uOpp Unary operator for use with binary ops which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return The result of applying the specified binary operation to {@code src1} and {@code src2}
     * element-wise.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static CsrCMatrix applyBinOpp(CsrMatrix src1, CsrCMatrix src2,
                                         BiFunction<Double, Complex128, Complex128> opp,
                                         UnaryOperator<Complex128> uOpp) {
        // TODO: The JavaDoc is not correct. The uOpp is not applied in that way. Either update the docs or how the binary operation
        //  is computed. This is an issue in multiple csr ops utility classes.
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        List<Complex128> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) {
                    dest.add(opp.apply(src1.data[rowPtr1], src2.data[rowPtr2]));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                } else if(col1 < col2) {
                    dest.add(new Complex128(src1.data[rowPtr1]));
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    if(uOpp!=null) dest.add(uOpp.apply(src2.data[rowPtr2]));
                    else dest.add(src2.data[rowPtr2]);
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(new Complex128(src1.data[rowPtr1]));
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                if(uOpp!=null) dest.add(uOpp.apply(src2.data[rowPtr2]));
                else dest.add(src2.data[rowPtr2]);
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrix(src1.shape,
                dest.toArray(Complex128[]::new),
                rowPointers,
                ArrayConversions.fromIntegerList(colIndices)
        );
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrix CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to dense matrices and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to {@code src1} and {@code src2}.
     * @param uOpp Unary operator for use with binary ops which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return The result of applying the specified binary operation to {@code src1} and {@code src2}
     * element-wise.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static CsrCMatrix applyBinOpp(CsrCMatrix src1, CsrMatrix src2,
                                         BiFunction<Complex128, Double, Complex128> opp,
                                         UnaryOperator<Double> uOpp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        List<Complex128> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) {
                    double val2 = uOpp==null ? src2.data[rowPtr2] : uOpp.apply(src2.data[rowPtr2]);
                    dest.add(opp.apply(src1.data[rowPtr1], val2));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                } else if(col1 < col2) {
                    dest.add(src1.data[rowPtr1]);
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    if(uOpp!=null) dest.add(new Complex128(uOpp.apply(src2.data[rowPtr2])));
                    else dest.add(new Complex128(src2.data[rowPtr2]));
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(src1.data[rowPtr1]);
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                if(uOpp!=null) dest.add(new Complex128(uOpp.apply(src2.data[rowPtr2])));
                else dest.add(new Complex128(src2.data[rowPtr2]));
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrix(src1.shape,
                dest.toArray(Complex128[]::new),
                rowPointers,
                ArrayConversions.fromIntegerList(colIndices)
        );
    }


     /**
     * Computes the element-wise multiplication between a complex sparse matrix and a real sparse matrix. <br><br>
     *
     * @param src1 The first matrix in the element-wise multiplication.
     * @param src2 The second matrix in the element-wise multiplication.
     * @return The result of the element-wise multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static CsrCMatrix elemMult(CsrCMatrix src1, CsrMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        List<Complex128> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) { // Only values at the same indices need to be multiplied.
                    dest.add(src1.data[rowPtr1].mult(src2.data[rowPtr2]));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrix(src1.shape,
                dest.toArray(Complex128[]::new),
                rowPointers,
                ArrayConversions.fromIntegerList(colIndices)
        );
    }


    /**
     * Computes the element-wise sum of two matrices.
     * @param a First matrix in sum.
     * @param b Second matrix in sum.
     * @return The element-wise sum of {@code a} and {@code b}.
     */
    public static CsrCMatrix add(CsrCMatrix a, CsrMatrix b) {
        return applyBinOpp(a, b, (Complex128 x, Double y)->x.add(y), null);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CsrCMatrix sub(CsrCMatrix a, CsrMatrix b) {
        return applyBinOpp(a, b, Complex128::add, (Double x)->-x);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CsrCMatrix sub(CsrMatrix a, CsrCMatrix b) {
        return applyBinOpp(a, b, (Double x, Complex128 y)->new Complex128(x-y.re, -y.im), (Complex128 x)->x.negate());
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param src1 CSR matrix to add vector to each column of.
     * @param src2 Vector to add to each column of this matrix.
     * @return The result of adding the vector src2 to each column of this matrix.
     */
    public static CMatrix addToEachCol(CsrCMatrix src1, Vector src2) {
        ValidateParameters.ensureAllEqual(src1.numRows, src2.size);
        CMatrix sum = src2.repeat(src1.numCols, 1).toComplex();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = rowOffset + src1.colIndices[j];
                sum.data[idx] = sum.data[idx].add(src1.data[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param src1 CSR matrix to add vector to each column of.
     * @param src2 Vector to add to each column of this matrix.
     * @return The result of adding the vector src2 to each column of this matrix.
     */
    public static CMatrix addToEachCol(CsrCMatrix src1, CooVector src2) {
        ValidateParameters.ensureAllEqual(src1.numRows, src2.size);
        CMatrix sum = src2.repeat(src1.numCols, 1).toComplex().toDense();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = rowOffset + src1.colIndices[j];
                sum.data[idx] = sum.data[idx].add(src1.data[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param src1 CSR matrix to add vector to each row of.
     * @param src2 Vector to add to each row of this matrix.
     * @return The result of adding the vector src2 to each row of this matrix.
     */
    public static CMatrix addToEachRow(CsrCMatrix src1, Vector src2) {
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);
        CMatrix sum = src2.repeat(src1.numRows, 0).toComplex();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = i*sum.numCols + src1.colIndices[j];
                sum.data[idx] = sum.data[idx].add(src1.data[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param src1 CSR matrix to add vector to each row of.
     * @param src2 Vector to add to each row of this matrix.
     * @return The result of adding the vector src2 to each row of this matrix.
     */
    public static CMatrix addToEachRow(CsrCMatrix src1, CooVector src2) {
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);
        CMatrix sum = src2.repeat(src1.numRows, 0).toComplex().toDense();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = i*sum.numCols + src1.colIndices[j];
                sum.data[idx] = sum.data[idx].add(src1.data[j]);
            }
        }

        return sum;
    }
}
