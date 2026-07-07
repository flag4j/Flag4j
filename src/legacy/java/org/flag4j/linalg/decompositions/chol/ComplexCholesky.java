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


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>An abstract base class for Cholesky decomposition of Hermitian (or Hermitian) positive-definite matrices.
 *
 * <p>The Cholesky decomposition factorizes a Hermitian/Hermitian, positive-definite matrix <span class="latex-inline">A</span> as:
 * <span class="latex-display"><pre>
 *     A = LL<sup>H</sup></pre></span>
 * where <span class="latex-inline">L</span> is a lower triangular matrix.
 * The decomposition is primarily used for efficient numerical solutions to linear systems, computing matrix inverses,
 * and generating samples from multivariate normal distributions.
 *
 * <h2>Hermitian Verification:</h2>
 * <p>This class provides an option to explicitly check whether the input matrix is Hermitian. If {@code enforceHermitian} is set
 * to {@code true}, the implementation will verify that <span class="latex-inline">A</span> satisfies 
 * <span class="latex-inline">A = A<sup>H</sup></span> before performing decomposition.
 * If set to {@code false}, the matrix is assumed to be Hermitian, no explicit check will be performed, and only the lower-diagonal
 * entries of <span class="latex-inline">A</span> are accessed.
 *
 * <h2>positive-definiteness Check:</h2>
 * <p>To ensure numerical stability, the algorithm verifies that all diagonal entries of <span class="latex-inline">L</span>
 * are positive. A tolerance threshold, {@code posDefTolerance}, is used to determine whether a diagonal entry is considered
 * non-positive, indicating that the matrix is <em>not</em> positive-definite. This threshold can be adjusted using
 * {@link #setPosDefTolerance(double)}.
 *
 * <h2>Usage:</h2>
 * <p>A typical workflow using a concrete implementation of Cholesky decomposition follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code ComplexCholesky}.</li>
 *     <li>Call {@link #decompose(CMatrix)} to compute the decomposition.</li>
 *     <li>Retrieve the factorized matrices using {@link #getL()} or {@link #getLH()}.</li>
 * </ol>
 *
 * @see RealCholesky
 * @see Decomposition
 * @see CMatrix
 * @see #setPosDefTolerance(double)
 */
public class ComplexCholesky extends Cholesky<CMatrix> {

    /**
     * <p>Constructs a complex Cholesky decomposer.
     */
    public ComplexCholesky() {
        super(true);
    }


    /**
     * Constructs a complex Cholesky decomposer.
     *
     * @param enforceHermitian Flag indicating if an explicit check should be made that the matrix to be decomposed is Hermitian.
     * <ul>
     *     <li>If {@code true}, the matrix will be explicitly verified to be Hermitian.</li>
     *     <li>If {@code false}, <em>no</em> check will be made to verify the matrix is Hermitian, and it will be assumed to be.</li>
     * </ul>
     */
    public ComplexCholesky(boolean enforceHermitian) {
        super(enforceHermitian);
    }


    /**
     * Decompose a matrix into <span class="latex-inline">A = LL<sup>H</span> where <span class="latex-inline">L</span> is a
     * lower triangular matrix and <span class="latex-inline">L<sup>H</sup></span> is
     * the conjugate transpose of <span class="latex-inline">L</span>.
     *
     * @param src The source matrix to decompose. Must be Hermitian positive-definite.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code src.numRows != src.numCols} or {@code src} is not Hermitian and
     * {@link #ComplexCholesky(boolean) enforceHermitian} was set to true when this decomposer was instantiated.
     * @throws LinearAlgebraException If {@code src} is not positive-definite.
     */
    @Override
    public ComplexCholesky decompose(CMatrix src) {
        if(enforceHermitian && !src.isHermitian())
            throw new IllegalArgumentException(SYM_POS_DEF_ERR);
        else
            ValidateParameters.ensureSquareMatrix(src.shape);

        L = src.getTriL();
        double posDefTolerance = Math.max(L.numRows*Flag4jConstants.EPS_F64, super.posDefTolerance);
        Complex128 sum;

        int lIndex1;
        int lIndex2;
        int lIndex3;

        for(int i=0; i<L.numCols; i++) {
            lIndex1 = i*L.numCols;

            for(int j=0; j<=i; j++) {
                sum = Complex128.ZERO;
                lIndex2 = j*L.numCols;
                lIndex3 = lIndex1 + j;

                for(int k=0; k<j; k++)
                    sum = sum.add(L.data[lIndex1 + k].mult(L.data[lIndex2 + k].conj()));

                if(i==j) {
                    Complex128 diag = L.data[lIndex3].sub(sum);
                    if(diag.re <= 0 || diag.mag() <= posDefTolerance)
                        throw new LinearAlgebraException(SYM_POS_DEF_ERR);

                    L.data[lIndex3] = diag.sqrt();
                } else if(!L.data[j*(L.numCols + 1)].isZero()) {
                        L.data[lIndex3] = (L.data[lIndex3].sub(sum)).div(L.data[lIndex2 + j]);
                }
            }
        }

        this.hasDecomposed = true;
        return this;
    }
}
