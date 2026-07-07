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

package org.flag4j.linalg.solvers.lstsq;


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.linalg.decompositions.unitary.UnitaryDecomposition;
import org.flag4j.linalg.solvers.LinearMatrixSolver;


/**
 * <p>An abstract solver for linear systems of the form <span class="latex-inline">Ax = b</span> or
 * <span class="latex-inline">AX = B</span>
 * in a <em>least-squares</em> sense. The system may be under-, well-, or over-determined.
 * Specifically, this solver minimizes the sum of squared residuals:
 * <ul>
 *   <li><span class="latex-inline">||Ax - b||<sub>2</sub><sup>2</sup></span> for vector-based problems, or</li>
 *   <li><span class="latex-inline">||AX - B||<sub>F</sub><sup>2</sup></span> for matrix-based where problems</li>
 * </ul>
 * where <span class="latex-inline">||&middot;||<sub>2</sub></span> is the Euclidean vector norm and
 * <span class="latex-inline">||&middot;||<sub>F</sub></span>
 * is the Frobenius norm. This is equivalent to solving the normal equations given by:
 * <span class="latex-inline">A<sup>H</sup>Ax = A<sup>H</sup>b</span> <em>or</em>
 * <span class="latex-inline">A<sup>H</sup>AX = A<sup>H</sup>B</span>
 *
 * <h2>Usage:</h2>
 * <p>A single system may be solved by calling either {@link #solve(MatrixMixin, VectorMixin)} or
 * {@link #solve(MatrixMixin, VectorMixin)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form
 * <span class="latex-inline">Ax = b</span> or <span class="latex-inline">AX = B</span>
 * for the same coefficient matrix <span class="latex-inline">A</span> but numerous constant vectors/matrices <span class="latex-inline">b</span> or
 * <span class="latex-inline">B</span>. To do this, the workflow would be as follows:
 * <ol>
 *     <li>Create a concrete instance of {@code LstsqSolver}.</li>
 *     <li>Call {@link #decompose(MatrixMixin) decompse(A)} once on the coefficient matrix <span class="latex-inline">A</span>.</li>
 *     <li>Call {@link #solve(VectorMixin) solve(b)} or {@link #solve(MatrixMixin) solve(B)} as many times as needed to solve each
 *     system for with the various <span class="latex-inline">b</span> vectors and/or <span class="latex-inline">B</span> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link #decompose(MatrixMixin) decompse(A)} will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
 *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <p>Minimizing the sum of squared residuals is achieved by computing a QR decomposition of the coefficient
 * matrix <span class="latex-inline">A</span>:
 * <span class="latex-display"><blockquote>
 *    A = QR
 * </blockquote></span>
 * where <span class="latex-inline">Q</span> is a unitary/orthonormal matrix and <span class="latex-inline">R</span>
 * is an upper triangular matrix. The normal equations then reduces to:
 * <span class="latex-replace"><blockquote>
 *   A<sup>H</sup>Ax = A<sup>H</sup>b <em>or</em> A<sup>H</sup>AX = A<sup>H</sup>B <br>
 *
 *   &Implies; (QR)<sup>H</sup>QRx = (QR)<sup>H</sup>b
 *   &nbsp; <em>or</em> &nbsp; (QR)<sup>H</sup>QRX = (QR)<sup>H</sup>B  <br>
 *
 *   &Implies; R<sup>H</sup>Q<sup>H</sup>QRx = R<sup>H</sup>Q<sup>H</sup>b
 *   &nbsp; <em>or</em> &nbsp; R<sup>H</sup>Q<sup>H</sup>QRX = R<sup>H</sup>Q<sup>H</sup>B <br>
 *
 *   &Implies; R<sup>H</sup>Rx = R<sup>H</sup>Q<sup>H</sup>b
 *   &nbsp; <em>or</em> &nbsp; R<sup>H</sup>RX = R<sup>H</sup>Q<sup>H</sup>B
 *   since <span class="latex-inline">Q</span> is unitary/orthonormal.<br>
 *
 *   &Implies; Rx = Q<sup>H</sup>b
 *   &nbsp; <em>or</em> &nbsp; RX = Q<sup>H</sup>B
 * </blockquote></span>
 *
 * <!-- LATEX: \[
 * \begin{alignat*}{6}
 *    & &A^HAx &= A^Hb &&\; \text{ or }\; &A^HAX &= A^HB \\
 *    &\implies &(QR)^HQRx &= (QR)^Hb &&\; \text{ or }\; &(QR)^HQRX &= (QR)^HB \\
 *    &\implies &R^HQ^HQRx &= R^HQ^Hb &&\; \text{ or }\; &R^HQ^HQRX &= R^HQ^HB \\
 *    &\implies &R^HRx &= R^HQ^Hb &&\; \text{ or }\; &R^HRX &= R^HQ^HB \quad \text{ since } Q \text{ is unitary.} \\
 *    &\implies &Rx &= Q^Hb &&\; \text{ or }\; & RX &= Q^HB
 * \end{alignat*}
 * \] -->
 *
 * which is easily solved by back-substitution on <span class="latex-inline">R</span>. In the real case, 
 * <span class="latex-inline">Q<sup>H</sup></span> simply
 * becomes <span class="latex-inline">Q<sup>T</sup></span>.
 *
 * @param <T> The matrix type in the linear system.
 * @param <U> The vector type in the linear system.
 */
public abstract class LstsqSolver<T extends MatrixMixin<T, ?, U, ?>, U extends VectorMixin<U, T, ?, ?>>
        implements LinearMatrixSolver<T, U> {


    /**
     * Solver for system with an upper triangular coefficient matrix.
     */
    protected final LinearMatrixSolver<T, U> backSolver;
    /**
     * Decomposer to compute the QR decomposition for using the least-squares solver.
     */
    protected final UnitaryDecomposition<T, ?> qr;
    /**
     * The Hermitian transpose of the unitary matrix, <span class="latex-inline">Q</span>, from the QR decomposition.
     */
    protected T Qh;
    /**
     * The upper triangular matrix, <span class="latex-inline">R</span>, from the QR decomposition.
     */
    protected T R;

    /**
     * Constructs a least-squares solver with a specified decomposer to use in the QR decomposition.
     * @param qr The QR decomposer to use in the solver.
     * @param backSolver The solver to solve the upper triangular system resulting from the QR decomposition
     *                   which is equivalent to solving the normal equations
     */
    protected LstsqSolver(UnitaryDecomposition<T, ?> qr, LinearMatrixSolver<T, U> backSolver) {
        this.qr = qr;
        this.backSolver = backSolver;
    }


    /**
     * Solves the linear system given by <span class="latex-inline">Ax = b</span> in a least-squares sense.
     *
     * <p><strong>Note</strong>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <span class="latex-inline">A</span>, in the linear system.
     * @param b Constant vector, <span class="latex-inline">b</span>, in the linear system.
     * @return The least-squares solution to <span class="latex-inline">c</span> in the linear system 
     * <span class="latex-inline">Ax = b</span>.
     */
    @Override
    public U solve(T A, U b) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, (U) Qh.mult(b));
    }


    /**
     * Solves the linear system of equation given by <span class="latex-inline">AX = B</span> for the matrix 
     * <span class="latex-inline">X</span> in a 
     * least-squares sense.
     *
     * <p><strong>Note</strong>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <span class="latex-inline">A</span>, in the linear system.
     * @param B Constant matrix, <span class="latex-inline">B</span>, in the linear system.
     * @return The solution to <span class="latex-inline">X</span> in the linear system <span class="latex-inline">AX = B</span>.
     */
    @Override
    public T solve(T A, T B) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, (T) Qh.mult(B));
    }


    /**
     * Solves the linear system given by <span class="latex-inline">Ax = b</span> in a least-squares sense.
     *
     * @param b Constant vector, <span class="latex-inline">b</span>, in the linear system.
     * @return The solution to <span class="latex-inline">x</span> in the linear system <span class="latex-inline">Ax = b</span>
     * for the last <span class="latex-inline">A</span> passed to
     * {@link #decompose(MatrixMixin)}.
     * @throws IllegalStateException If no coefficient matrix has been specified for this solver by first calling one of the following:
     * <ul>
     *     <li>{@link #decompose(MatrixMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
     * </ul>
     */
    public U solve(U b) {
        if (Qh == null) {
            throw new IllegalStateException("Coefficient matrix has not been specified for this solver." +
                    "\nMust call decompose(...) or a solve(...) which accepts a coefficient matrix first.");
        }

        return backSolver.solve(R, (U) Qh.mult(b));
    }


    /**
     * Solves the set of linear system of equations given by <span class="latex-inline">AX = B</span> for the matrix 
     * <span class="latex-inline">X</span> in a least-squares sense.
     *
     * @param B Constant matrix, <span class="latex-inline">B</span>, in the linear system.
     * @return The solution to <span class="latex-inline">X</span> in the linear system <span class="latex-inline">AX = B</span> 
     * for the last <span class="latex-inline">A</span> passed to
     * {@link #decompose(MatrixMixin)}.
     * @throws IllegalStateException If no coefficient matrix has been specified for this solver by first calling one of the following:
     * <ul>
     *     <li>{@link #decompose(MatrixMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
     * </ul>
     */
    public T solve(T B) {
        if (Qh == null) {
            throw new IllegalStateException("Coefficient matrix has not been specified for this solver." +
                    "\nMust call decompose(...) or a solve(...) which accepts a coefficient matrix first.");
        }

        return backSolver.solve(R, (T) Qh.mult(B));
    }


    /**
     * <p>Computes (or updates) the QR decomposition of the given matrix <span class="latex-inline">A</span>
     * for use in subsequent solves.
     *
     * <p>Subclasses may override this method to customize how the QR decomposition is computed or updated.
     *
     * @param A Coefficient matrix to decompose.
     */
    public void decompose(T A) {
        qr.decompose(A);
        Qh = qr.getQ().H();
        R = qr.getUpper();
    }
}
