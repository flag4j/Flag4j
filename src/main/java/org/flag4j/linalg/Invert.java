/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.chol.Cholesky;
import org.flag4j.linalg.decompositions.chol.ComplexCholesky;
import org.flag4j.linalg.decompositions.chol.RealCholesky;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.decompositions.svd.SVD;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;
import org.flag4j.linalg.solvers.exact.ComplexExactSolver;
import org.flag4j.linalg.solvers.exact.RealExactSolver;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.ComplexForwardSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealForwardSolver;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * This class provides methods for computing the inverse of a matrix. Specialized methods are provided for inverting triangular,
 * diagonal, and symmetric positive-definite matrices.
 */
public final class Invert {

    private Invert() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the inverse of this matrix. This is done by computing the {@link LU LU decomposition} of
     * this matrix, inverting <span class="latex-inline">L</span> using a back-solve algorithm, then solving
     * <span class="latex-inline">UA<sup>-1</sup> = L<sup>-1</sup></span>
     * for <span class="latex-inline">A<sup>-1</sup></span>.
     *
     * @param src Matrix to compute inverse of.
     * @return The inverse of this matrix.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular (i.e., not invertible).
     */
    public static Matrix inv(Matrix src) {
        return new RealExactSolver().solveIdentity(src);
    }


    /**
     * Computes the inverse of this matrix. This is done by computing the {@link LU LU decomposition} of
     * this matrix, inverting <span class="latex-inline">L</span> using a back-solve algorithm, then solving
     * <span class="latex-inline">UA<sup>-1</sup> = L<sup>-1</sup></span>
     * for <span class="latex-inline">A<sup>-1</sup></span>.
     *
     * @param src Matrix to compute inverse of.
     * @return The inverse of this matrix.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular (i.e., not invertible).
     */
    public static CMatrix inv(CMatrix src) {
        return new ComplexExactSolver().solveIdentity(src);
    }


    /**
     * Inverts an upper triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * upper triangular.
     * @param src Upper triangular matrix to compute the inverse of.
     * @return The inverse of the upper triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e., has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static Matrix invTriU(Matrix src) {
        return new RealBackSolver(false).solveIdentity(src); // If the matrix is singular, it will be caught here.
    }


    /**
     * Inverts a lower triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * lower triangular.
     * @param src Lower triangular matrix to compute the inverse of.
     * @return The inverse of the lower triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e., has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static Matrix invTriL(Matrix src) {
        return new RealForwardSolver().solveIdentity(src); // If the matrix is singular, it will be caught here.
    }


    /**
     * Inverts a diagonal matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * diagonal.
     * @param src Diagonal matrix to compute the inverse of.
     * @return The inverse of the diagonal matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e., has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static Matrix invDiag(Matrix src) {
        ValidateParameters.ensureSquareMatrix(src.shape);
        Matrix inverse = new Matrix(src.shape);

        double value;
        int step = src.numCols+1;
        double det = 1;

        for(int i = 0; i<src.data.length; i+=step) {
            value = src.data[i];
            det *= value;
            inverse.data[i] = 1.0/value;
        }

        if(Math.abs(det) <= EPS_F64*Math.max(src.numRows, src.numCols)) {
            throw new SingularMatrixException("Could not invert.");
        }

        return inverse;
    }


    /**
     * Inverts an upper triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * upper triangular.
     * @param src Upper triangular matrix to compute the inverse of.
     * @return The inverse of the upper triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e., has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static CMatrix invTriU(CMatrix src) {
        return new ComplexBackSolver().solveIdentity(src); // If matrix is singular, it will be caught here.
    }


    /**
     * Inverts a lower triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * lower triangular and will treat it as such even if it is not triangular.
     * @param src Lower triangular matrix to compute the inverse of.
     * @return The inverse of the lower triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e., has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static CMatrix invTriL(CMatrix src) {
        return new ComplexForwardSolver().solveIdentity(src); // If matrix is singular, it will be caught here.
    }


    /**
     * Inverts a diagonal matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * diagonal.
     * @param src Diagonal matrix to compute the inverse of.
     * @return The inverse of the diagonal matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e., has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static CMatrix invDiag(CMatrix src) {
        ValidateParameters.ensureSquareMatrix(src.shape);

        CMatrix inverse = new CMatrix(src.shape);

        Complex128 value;
        int step = src.numCols+1;
        Complex128 det = Complex128.ONE;

        for(int i = 0; i<src.data.length; i+=step) {
            value = src.data[i];
            det = det.mult(value);
            inverse.data[i] = value.multInv();
        }

        if(det.mag() <= EPS_F64*Math.max(src.numRows, src.numCols))
            throw new SingularMatrixException("Could not invert.");

        return inverse;
    }


    /**
     * Inverts a symmetric positive-definite matrix.
     * @param src positive-definite matrix. It will <em>not</em> be verified if {@code src} is actually symmetric positive-definite.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     * @see #invSymPosDef(Matrix, boolean)
     */
    public static Matrix invSymPosDef(Matrix src) {
        return invSymPosDef(src, false);
    }


    /**
     * Inverts a symmetric positive-definite matrix.
     * @param src positive-definite matrix.
     * @param checkPosDef Flag indicating if a check should be made to see if {@code src} is actually symmetric
     *                    positive-definite. <b>WARNING</b>: Checking if the matrix is positive-definite can be very computationally
     *                    expensive.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     */
    public static Matrix invSymPosDef(Matrix src, boolean checkPosDef) {
        Cholesky<Matrix> chol = new RealCholesky(checkPosDef).decompose(src);
        RealBackSolver backSolver = new RealBackSolver(false);
        RealForwardSolver forwardSolver = new RealForwardSolver();

        // Compute the inverse of lower triangular matrix L.
        Matrix Linv = forwardSolver.solveIdentity(chol.getL());

        return backSolver.solveLower(chol.getLH(), Linv); // Compute inverse of src.
    }


    /**
     * Inverts a Hermitian positive-definite matrix.
     * @param src positive-definite matrix. It will <em>not</em> be verified if {@code src} is actually Hermitian positive-definite.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     * @see #invSymPosDef(Matrix, boolean)
     */
    public static CMatrix invHermPosDef(CMatrix src) {
        return invHermPosDef(src, false);
    }


    /**
     * Inverts a Hermitian positive-definite matrix.
     * @param src positive-definite matrix.
     * @param checkPosDef Flag indicating if a check should be made to see if {@code src} is actually Hermitian
     *                    positive-definite. <b>WARNING</b>: Checking if the matrix is positive-definite can be very computationally
     *                    expensive.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     */
    public static CMatrix invHermPosDef(CMatrix src, boolean checkPosDef) {
        Cholesky<CMatrix> chol = new ComplexCholesky(checkPosDef).decompose(src);
        ComplexBackSolver backSolver = new ComplexBackSolver();
        ComplexForwardSolver forwardSolver = new ComplexForwardSolver();

        // Compute the inverse of lower triangular matrix L.
        CMatrix Linv = forwardSolver.solveIdentity(chol.getL());

        return backSolver.solveLower(chol.getLH(), Linv); // Compute inverse of src.
    }


    // ------------------------------------------- Pseudo-inverses -------------------------------------------

    /**
     * Computes the pseudo-inverse of this matrix. That is, for a matrix <span class="latex-inline">A</span>,
     * computes the Moore–Penrose <span class="latex-inline">A<sup>+</sup></span> such that the following hold:
     * <ol>
     *   <li><span class="latex-inline">AA<sup>+</sup>A=A</span>.</li>
     *   <li><span class="latex-inline">A<sup>+</sup>AA<sup>+</sup>=A<sup>+</sup></span>.</li>
     *   <li><span class="latex-inline">AA<sup>+</sup></span> is Hermitian.</li>
     *   <li><span class="latex-inline">A<sup>+</sup>A</span> is also Hermitian.</li>
     * </ol>
     *
     * @return The Moore–Penrose pseudo-inverse of this matrix.
     */
    public static Matrix pInv(Matrix src) {
        SVD<Matrix> svd = new RealSVD().decompose(src);
        Matrix sInv = Invert.invDiag(svd.getS());

        return svd.getV().mult(sInv).mult(svd.getU().T());
    }


    /**
     * Computes the pseudo-inverse of this matrix. That is, for a matrix <span class="latex-inline">A</span>,
     * computes the Moore–Penrose <span class="latex-inline">A<sup>+</sup></span> such that the following hold:
     * <ol>
     *   <li><span class="latex-inline">AA<sup>+</sup>A=A</span>.</li>
     *   <li><span class="latex-inline">A<sup>+</sup>AA<sup>+</sup>=A<sup>+</sup></span>.</li>
     *   <li><span class="latex-inline">AA<sup>+</sup></span> is Hermitian.</li>
     *   <li><span class="latex-inline">A<sup>+</sup>A</span> is also Hermitian.</li>
     * </ol>
     *
     * @return The Moore–Penrose pseudo-inverse of this matrix.
     */
    public static CMatrix pInv(CMatrix src) {
        SVD<CMatrix> svd = new ComplexSVD().decompose(src);
        Matrix sInv = Invert.invDiag(svd.getS());

        return svd.getV().mult(sInv).mult(svd.getU().H());
    }

    // -------------------------------- End Pseudo-inverses --------------------------------

    /**
     * Checks if matrices are inverses of each other. This method rounds values near zero to zero when checking
     * if the two matrices are inverses to account for floating point precision loss.
     *
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return {@code true} if matrix src2 is an inverse of this matrix; {@code false} otherwise. Otherwise, returns false.
     */
    public static boolean isInv(Matrix src1, Matrix src2) {
        boolean result;

        if(!src1.isSquare() || !src2.isSquare() || !src1.shape.equals(src2.shape))
            result = false;
        else
            result = src1.mult(src2).isCloseToI();

        return result;
    }


    /**
     * Checks if matrices are inverses of each other.
     *
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return {@code true} if matrix src2 is an inverse (approximately) of this matrix; {@code false} otherwise. Otherwise, returns false.
     */
    public static boolean isInv(CMatrix src1, CMatrix src2) {
        boolean result;

        if(!src1.isSquare() || !src2.isSquare() || !src1.shape.equals(src2.shape)) {
            result = false;
        } else {
            CMatrix prod = src1.mult(src2);
            CMatrix I = CMatrix.I(src1.shape);
            result = RingProperties.allClose(prod.data, I.data);
        }

        return result;
    }
}
