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

package org.flag4j.linalg.solvers.exact;


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.lu.RealLU;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealForwardSolver;

/**
 * <p>Solves a well determined system of equations <span class="latex-inline">Ax = b</span> or <span class="latex-inline">AX = B</span>
 * in an exact sense by using a
 * {@link RealLU LU decomposition}
 * where <span class="latex-inline">A</span>, <span class="latex-inline">B</span>, and <span class="latex-inline">X</span>
 * are matrices, and <span class="latex-inline">x</span> and <span class="latex-inline">b</span> are vectors.
 *
 * <p>If the system is not well determined, i.e., <span class="latex-inline">A</span> is not square or not full rank, then use a
 * {@link org.flag4j.linalg.solvers.lstsq.RealLstsqSolver least-squares solver}.
 *
 * <h2>Usage:</h2>
 * <p>A single system may be solved by calling either {@link ExactSolver#solve(MatrixMixin, VectorMixin)} or
 * {@link ExactSolver#solve(MatrixMixin, VectorMixin)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form <span class="latex-inline">Ax = b</span>
 * or <span class="latex-inline">AX = B</span>
 * for the same coefficient matrix <span class="latex-inline">A</span> but numerous constant vectors/matrices
 * <span class="latex-inline">b</span> or <span class="latex-inline">B</span>. To do this, the workflow
 * would be as follows:
 * <ol>
 *     <li>Create an instance of {@code RealExactSolver}.</li>
 *     <li>Call {@link ExactSolver#decompose(MatrixMixin) decompse(A)} once on the coefficient matrix <span
 *     class="latex-inline">A</span>.</li>
 *     <li>Call {@link ExactSolver#solve(VectorMixin) solve(b)} or {@link ExactSolver#solve(MatrixMixin) solve(B)} as many times as
 *     needed to solve
 *     each
 *     system for with the various <span class="latex-inline">b</span> vectors and/or
 *     <span class="latex-inline">B</span> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link ExactSolver#decompose(MatrixMixin) decompse(A)}
 * will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link ExactSolver#solve(MatrixMixin, VectorMixin)}</li>
 *     <li>{@link ExactSolver#solve(MatrixMixin, VectorMixin)}</li>
 * </ul>
 *
 * <p>Specialized solvers are provided for inversion using {@link ExactSolver#solveIdentity(MatrixMixin)}. This should be preferred
 * over calling on of the other solve methods and providing an identity matrix explicitly.
 *
 * @param <T> The type of the coefficient matrix in the linear system.
 * @param <U> The type of vector in the linear system.
 */
public class RealExactSolver extends ExactSolver<Matrix, Vector> {

    /**
     * Constructs an exact LU solver where the coefficient matrix is real dense.
     */
    public RealExactSolver() {
        super(new RealLU(),
                new RealForwardSolver(true, false),
                new RealBackSolver(false));
    }


    /**
     * Permute the rows of a vector using the row permutation matrix from the LU decomposition.
     *
     * @param b Vector to permute the rows of.
     * @return A vector which is the result of applying the row permutation from the LU decomposition
     * to the vector {@code b}.
     */
    @Override
    protected Vector permuteRows(Vector b) {
        return rowPermute.leftMult(b);
    }


    /**
     * Permute the rows of a matrix using the row permutation matrix from the LU decomposition.
     *
     * @param B matrix to permute the rows of.
     * @return A matrix which is the result of applying the row permutation from the LU decomposition
     * to the matrix {@code B}.
     */
    @Override
    protected Matrix permuteRows(Matrix B) {
        return rowPermute.leftMult(B);
    }
}
