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

package org.flag4j.linalg.ops.dense_sparse.csr.real_complex;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * This class contains low-level ops which act on a real/complex dense and a complex/real
 * sparse {@link CsrCMatrix CSR matrix}.
 */
public final class RealComplexCsrDenseOps {

    private RealComplexCsrDenseOps() {
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
    public static CMatrix applyBinOpp(CsrCMatrix src1, Matrix src2,
                                      BiFunction<Complex128, Double, Complex128> opp,
                                      UnaryOperator<Double> uOpp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);  // Ensure both matrices are same shape.

        Complex128[] dest;
        if(uOpp == null) dest = ArrayConversions.toComplex128(src2.data, null);
        else dest = ArrayUtils.applyTransform(src2.data.clone(), (Double a)->new Complex128(uOpp.apply(a)));

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply(src1.data[j], dest[idx].re);
            }
        }

        return new CMatrix(src2.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(Matrix src1, CsrCMatrix src2, BiFunction<Double, Complex128, Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        Complex128[] dest = ArrayConversions.toComplex128(src1.data, null);

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[i];

                dest[idx] = opp.apply(src1.data[idx], src2.data[j]);
            }
        }

        return new CMatrix(src2.shape, dest);
    }

    // -------------------------------------------------------------------------------


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
    public static CMatrix applyBinOpp(CsrMatrix src1, Complex128 b,
                                         BiFunction<Double, Complex128, Complex128> opp,
                                         UnaryOperator<Complex128> uOpp) {
        Complex128[] dest = new Complex128[src1.totalEntries().intValueExact()];
        if(uOpp != null) b = uOpp.apply(b);  // Apply unary operator if specified.
        Arrays.fill(dest, b);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];

                dest[idx] = opp.apply(src1.data[j], dest[idx]);
            }
        }

        return new CMatrix(src1.shape, dest);
    }


    /**
     * Computes the element-wise sum of two matrices.
     * @param a First matrix in sum.
     * @param b Second matrix in sum.
     * @return The element-wise sum of {@code a} and {@code b}.
     */
    public static CMatrix add(CsrCMatrix a, Matrix b) {
        return applyBinOpp(a, b, Complex128::add, null);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(CsrCMatrix a, Matrix b) {
        return applyBinOpp(a, b, Complex128::add, (Double x)->-x);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(Matrix a, CsrCMatrix b) {
        return applyBinOpp(a, b, (Double x, Complex128 y)->y.sub(x));
    }
}
