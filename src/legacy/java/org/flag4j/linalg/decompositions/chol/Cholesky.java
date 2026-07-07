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

package org.flag4j.linalg.decompositions.chol;


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.linalg.decompositions.Decomposition;


/**
 * <p>An abstract base class for Cholesky decomposition of symmetric (or Hermitian) positive-definite matrices.
 *
 * <p>The Cholesky decomposition factorizes a symmetric/Hermitian, positive-definite matrix <span class="latex-inline">A</span> as:
 *
 * <span class="latex-display"><pre>
 *     A = LL<sup>H</sup></pre></span>
 * where <span class="latex-inline">L</span> is a lower triangular matrix.
 * The decomposition is primarily used for efficient numerical solutions to linear systems, computing matrix inverses,
 * and generating samples from multivariate normal distributions.
 *
 * <h2>Hermitian Verification:</h2>
 * <p>This class provides an option to explicitly check whether the input matrix is Hermitian. If {@code enforceHermitian} is set
 * to {@code true}, the implementation will verify that <span class="latex-inline">A</span>
 * satisfies <span class="latex-inline">A = A<sup>H</sup></span> before performing decomposition.
 * If set to {@code false}, the matrix is assumed to be Hermitian, no explicit check will be performed, and only the lower-diagonal
 * entries of <span class="latex-inline">A</span> are accessed.
 *
 * <h2>positive-definiteness Check:</h2>
 * <p>To ensure numerical stability, the algorithm verifies that all diagonal entries of <span class="latex-inline">L</span>
 * are positive.
 * A tolerance threshold, {@code posDefTolerance}, is used to determine whether a diagonal entry is considered
 * non-positive, indicating that the matrix is <em>not</em> positive-definite. This threshold can be adjusted using
 * {@link #setPosDefTolerance(double)}.
 *
 * <h2>Usage:</h2>
 * <p>A typical workflow using a concrete implementation of Cholesky decomposition follows these steps:
 * <ol>
 *     <li>Instantiate a subclass of {@code Cholesky}.</li>
 *     <li>Call {@link #decompose(MatrixMixin)} to compute the decomposition.</li>
 *     <li>Retrieve the factorized matrices using {@link #getL()} or {@link #getLH()}.</li>
 * </ol>
 *
 * @param <T> The type of matrix on which the Cholesky decomposition is performed, extending {@link MatrixMixin}.
 *
 * @see Decomposition
 * @see MatrixMixin
 * @see #setPosDefTolerance(double)
 */
public abstract class Cholesky<T extends MatrixMixin<T, ?, ?, ?>> extends Decomposition<T> {

    /**
     * Error message to display when the matrix to be decomposed is not symmetric positive-definite.
     */
    protected static final String SYM_POS_DEF_ERR = "Matrix is not symmetric positive-definite.";

    /**
     * Flag indicating if an explicit check should be made that the matrix to be decomposed is Hermitian.
     * <ul>
     *     <li>If {@code true}, the matrix will be explicitly verified to be Hermitian.</li>
     *     <li>If {@code false}, <em>no</em> check will be made to verify the matrix is Hermitian, and it will be assumed to be.</li>
     * </ul>
     */
    protected boolean enforceHermitian;
    /**
     * Tolerance for determining if an entry along the diagonal of <span class="latex-inline">L</span> is not positive-definite.
     */
    protected double posDefTolerance = 1.0e-14;


    /**
     * Constructs a Cholesky decomposer.
     * @param enforceHermitian Flat indicating if an explicit check should be made that the matrix to be decomposed is Hermitian.
     * <ul>
     *     <li>If {@code true}, the matrix will be explicitly verified to be Hermitian.</li>
     *     <li>If {@code false}, <em>no</em> check will be made to verify the matrix is Hermitian, and it will be assumed to be.</li>
     * </ul>
     * @param inPlace Flag indicating if the decomposition should be done in-place.
     * <ul>
     *     <li>If {@code true}, the decomposition will be done in-place overwriting the original matrix.</li>
     *     <li>If {@code false}, the decomposition will be done out-of-place leaving the original matrix unmodified.</li>
     * </ul>
     */
    protected Cholesky(boolean enforceHermitian) {
        this.enforceHermitian = enforceHermitian;
    }


    /**
     * <p>Sets the tolerance for determining if the matrix being decomposed is positive-definite.
     * <p>The matrix being decomposed will be considered to <em>not</em> be positive-definite if any diagonal entry of
     * <span class="latex-inline">L</span>
     * is {@code <= tol}. By default, this value is {@code 1.0e-14}.
     * @param tol Tolerance to use. Must be non-negative.
     * @throws IllegalArgumentException If {@code tol < 0}.
     */
    public void setPosDefTolerance(double tol) {
        if(tol < 0)
            throw new IllegalArgumentException("tolerance must be non-negative but got tol=" + tol + ".");
        this.posDefTolerance = tol;
    }


    /**
     * The lower triangular matrix, <span class="latex-inline">L</span>, resulting from the Cholesky decomposition
     * <span class="latex-inline">A = LL<sup>H</sup></span>.
     */
    protected T L;


    /**
     * Gets the L matrix computed by the Cholesky decomposition <span class="latex-inline">A = LL<sup>H</sup></span>.
     * @return The <span class="latex-inline">L</span> matrix from the Cholesky decomposition
     * <span class="latex-inline">A = LL<sup>H</sup></span>.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not been called on this instance.
     */
    public T getL() {
        ensureHasDecomposed();
        return L;
    }


    /**
     * Gets the <span class="latex-inline">L<sup>H</sup></span> matrix computed by the Cholesky decomposition
     * <span class="latex-inline">A = LL<sup>H</sup></span>.
     * @return The <span class="latex-inline">L<sup>H</sup></span> matrix from the Cholesky decomposition
     * <span class="latex-inline">A = LL<sup>H</sup></span>.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not been called on this instance.
     */
    public T getLH() {
        ensureHasDecomposed();
        return L.H();
    }
}
