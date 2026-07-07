/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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
import org.flag4j.linalg.decompositions.chol.ComplexCholesky;
import org.flag4j.linalg.decompositions.chol.RealCholesky;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>Utility class for checking the positive-(semi-)definiteness of a real or complex matrix.
 *
 * <p>A matrix <span class="latex-inline">M</span> is positive-definite iff
 * <span class="latex-inline">Re[x<sup>H</sup>Mx] > 0</span> for any vector <span class="latex-inline">x</span> or equivalently, if
 * all eigenvalues are real and strictly greater than zero.
 *
 * <p>In the case where <span class="latex-inline">M</span> is real, this simplifies to
 * <span class="latex-inline">x<sup>T</sup>Mx > 0</span>.
 *
 * <p>Similarly, a matrix <span class="latex-inline">M</span> is positive-semi-definite iff
 * <span class="latex-inline">Re[x<sup>H</sup>Mx] >= 0</span> for any vector <span class="latex-inline">x</span> or equivalently, if
 * all eigenvalues are real and greater than or equal to than zero.
 */
public final class PositiveDefiniteness {

    private PositiveDefiniteness() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if the matrix is positive-definite. A matrix <span class="latex-inline">M</span> is positive-definite iff
     * <span class="latex-inline">x<sup>T</sup>Mx > 0</span> for any vector <span class="latex-inline">x</span> or equivalently, if
     * all eigenvalues are strictly greater than zero.
     *
     * @param src Matrix to check if it is positive-definite.
     * @return {@code true} if the matrix is positive-definite; {@code false} otherwise.
     * @see #isPosSemiDef(Matrix)
     */
    public static boolean isPosDef(Matrix src) {
        boolean result;
        double tol = 1.0E-8; // Tolerance for considering eigenvalues positive.

        if(src.isSymmetric())
            result = Eigen.getEigenValues(src).minAbs() > tol;
        else
            result = false;

        return result;
    }


    /**
     * Checks if the matrix is positive-definite. A matrix <span class="latex-inline">M</span> is positive-definite iff
     * <span class="latex-inline">Re[x<sup>T</sup>Mx] > 0</span> for any vector <span class="latex-inline">x</span> or equivalently,
     * if all eigenvalues are strictly greater than zero.
     *
     * @param src Matrix to check if it is positive-definite.
     * @return {@code true} if the matrix is positive-definite; {@code false} otherwise.
     * @see #isPosSemiDef(CMatrix)
     */
    public static boolean isPosDef(CMatrix src) {
        boolean result;
        double tol = 1.0E-8; // Tolerance for considering eigenvalues positive.

        if(src.isHermitian())
            result = Eigen.getEigenValues(src).toReal().min() > tol;
        else
            result = false;

        return result;
    }


    /**
     * Checks if the matrix is symmetric positive-definite. A matrix <span class="latex-inline">M</span> is symmetric
     * positive-definite iff the matrix is symmetric and
     * <span class="latex-inline">x<sup>H</sup>Mx > 0</span> for any vector <span class="latex-inline">x</span> or
     * equivalently, if all eigenvalues are real and strictly greater than zero.
     *
     * @param src Matrix to check if it is positive-definite.
     * @return {@code true} if the matrix is positive-definite; {@code false} otherwise.
     * @see #isPosSemiDef(Matrix)
     */
    public static boolean isSymmPosDef(Matrix src) {
        boolean result = true;

        try {
            new RealCholesky(true).decompose(src);
        } catch(LinearAlgebraException | IllegalArgumentException e) {
            result = false; // Could not compute Cholesky decomposition. Matrix is not symmetric positive-definite.
        }

        return result;
    }


    /**
     * Checks if the matrix is Hermitian positive-definite. A matrix <span class="latex-inline">M</span> is Hermitian
     * positive-definite iff the matrix is Hermitian and
     * <span class="latex-inline">Re[x<sup>H</sup>Mx] > 0</span> for any vector <span class="latex-inline">x</span> or
     * equivalently, if all eigenvalues are real and strictly greater than zero.
     *
     * @param src Matrix to check if it is positive-definite.
     * @return {@code true} if the matrix is positive-definite; {@code false} otherwise.
     * @see #isPosSemiDef(Matrix)
     */
    public static boolean isHermPosDef(CMatrix src) {
        boolean result = true;

        try {
            new ComplexCholesky(true).decompose(src);
        } catch(LinearAlgebraException | IllegalArgumentException e) {
            result = false; // Could not compute Cholesky decomposition. Matrix is not Hermitian positive-definite.
        }

        return result;
    }


    /**
     * Checks if the matrix is positive semi-definite. A matrix <span class="latex-inline">M</span> is positive-semi-definite iff
     * <span class="latex-inline">x<sup>T</sup>Mx >= 0</span> for any vector <span class="latex-inline">x</span>, or equivalently, if
     * all eigenvalues are greater than or equal to zero.
     *
     * @param src Matrix to check if it is positive semi-definite.
     * @return {@code true} if the matrix is positive semi-definite; {@code false} otherwise.
     * @see #isPosSemiDef(Matrix)
     */
    public static boolean isPosSemiDef(Matrix src) {
        boolean result;
        double tol = -1.0E-8; // Tolerance for considering eigenvalues non-negative.

        if(src.isSymmetric())
            result = Eigen.getEigenValues(src).toReal().min() > tol;
        else
            result = false;

        return result;
    }


    /**
     * Checks if the matrix is positive semi-definite. A matrix <span class="latex-inline">M</span> is positive-semi-definite iff
     * <span class="latex-inline">Re[x<sup>T</sup>Mx] >= 0</span> for any vector <span class="latex-inline">x</span>, or
     * equivalently, if
     * all eigenvalues are greater than or equal to zero.
     *
     * @param src Matrix to check if it is positive semi-definite.
     * @return {@code true} if the matrix is positive semi-definite; {@code false} otherwise.
     * @see #isPosSemiDef(CMatrix)
     */
    public static boolean isPosSemiDef(CMatrix src) {
        boolean result;
        double tol = -1.0E-8; // Tolerance for considering eigenvalues non-negative.

        if(src.isHermitian())
            result = Eigen.getEigenValues(src).toReal().min() > tol;
        else
            result = false;

        return result;
    }
}
