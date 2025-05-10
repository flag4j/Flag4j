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

package org.flag4j.linalg.ops.dense_sparse.csr.field_ops;

import org.flag4j.arrays.backend.field_arrays.AbstractCsrFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;


/**
 * This class contains low-level ops which act on a dense and a sparse CSR field matrix.
 */
public final class DenseCsrFieldOps {

    private DenseCsrFieldOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary ops which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> applyBinOpp(
            AbstractCsrFieldMatrix<?, ?, ?, T> src1, AbstractDenseFieldMatrix<?, ?, T> src2,
            BinaryOperator<T> opp, UnaryOperator<T> uOpp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        T[] dest;
        if(uOpp == null) dest = Arrays.copyOf(src2.data, src2.data.length);
        else dest = ArrayUtils.applyTransform(src2.data, uOpp);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply((T) src1.data[j], (T) dest[idx]);
            }
        }

        return src2.makeLikeNDArray(src2.shape, dest);
    }



    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> applyBinOpp(
            AbstractDenseFieldMatrix<?, ?, T> src1, 
            AbstractCsrFieldMatrix<?, ?, ?, T> src2, 
            BinaryOperator<T> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        T[] dest = Arrays.copyOf(src2.data, src2.data.length);

        for(int i=0; i<src2.nnz; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[i];
                dest[idx] = opp.apply(src1.data[idx], src2.data[j]);
            }
        }

        return src1.makeLikeNDArray(src2.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply element-wise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary ops which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> applyBinOpp(
            AbstractCsrFieldMatrix<?, ?, ?, T> src1, double b,
            BinaryOperator<T> opp,
            UnaryOperator<Double> uOpp) {
        T[] dest = (T[]) new Field[src1.shape.totalEntriesIntValueExact()];
        T bWrapped = (src1.getZeroElement() != null) ? src1.getZeroElement().add(b) : null;

        // Apply unary operator if specified.
        if(uOpp != null) Arrays.fill(dest, (src1.getZeroElement() != null) ? src1.getZeroElement().add(uOpp.apply(b)) : null);
        else Arrays.fill(dest, bWrapped);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply(src1.data[j], dest[idx]);
            }
        }

        return src1.makeLikeDenseTensor(src1.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply element-wise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary ops which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> applyBinOpp(
            AbstractCsrFieldMatrix<?, ?, ?, T> src1, T b,
            BinaryOperator<T> opp, UnaryOperator<T> uOpp) {
        T[] dest = (T[]) new Field[src1.totalEntries().intValueExact()];

        // Apply unary operator if specified.
        if(uOpp != null) Arrays.fill(dest, uOpp.apply(b));
        else Arrays.fill(dest, b);

        for(int i=0; i<src1.nnz; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply(src1.data[j], dest[idx]);
            }
        }

        return src1.makeLikeDenseTensor(src1.shape, dest);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static <T extends Field<T>> AbstractCsrFieldMatrix<?, ?, ?, T> applyBinOppToSparse(
            AbstractDenseFieldMatrix<?, ?, T> src1, AbstractCsrFieldMatrix<?, ?, ?, T> src2, BinaryOperator<T> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src2.rowPointers.clone();
        int[] colIndices = src2.colIndices.clone();
        Field<T>[] entries = new Field[src2.data.length];

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[j];
                entries[idx] = opp.apply((T) src1.data[idx], (T) src2.data[j]);
            }
        }

        return src2.makeLikeTensor(src1.shape, (T[]) entries, rowPointers, colIndices);
    }


    /**
     * Computes the element-wise sum of two matrices.
     * @param a First matrix in sum.
     * @param b Second matrix in sum.
     * @return The element-wise sum of {@code a} and {@code b}.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> add(
            AbstractCsrFieldMatrix<?, ?, ?, T> a, AbstractDenseFieldMatrix<?, ?, T> b) {
        return applyBinOpp(a, b, T::add, null);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> sub(
            AbstractCsrFieldMatrix<?, ?, ?, T> a, AbstractDenseFieldMatrix<?, ?, T> b) {
        return applyBinOpp(a, b, T::add, T::addInv);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> sub(
            AbstractDenseFieldMatrix<?, ?, T> a, AbstractCsrFieldMatrix<?, ?, ?, T> b) {
        return applyBinOpp(a, b, (T x, T y)->y.sub(x));
    }
}
