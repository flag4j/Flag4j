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
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.decompositions.qr.ComplexQR;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;

/**
 * <p>Instances of this class solve complex linear systems of the form <span class="latex-inline">Ax = b</span> or 
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
 * <p>A single system may be solved by calling either {@link LstsqSolver#solve(MatrixMixin, VectorMixin)} or
 * {@link LstsqSolver#solve(MatrixMixin, VectorMixin)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form
 * <span class="latex-inline">Ax = b</span> or
 * <span class="latex-inline">AX = B</span>
 * for the same coefficient matrix <span class="latex-inline">A</span> but numerous constant vectors/matrices
 * <span class="latex-inline">b</span> or <span class="latex-inline">B</span>.
 * To do this, the workflow would be as follows:
 * <ol>
 *     <li>Create an instance of {@code RealLstsqSolver}.</li>
 *     <li>Call {@link LstsqSolver#decompose(MatrixMixin) decompse(A)} once on the coefficient matrix <span
 *     class="latex-inline">A</span>.</li>
 *     <li>Call {@link LstsqSolver#solve(VectorMixin) solve(b)} or {@link LstsqSolver#solve(MatrixMixin) solve(B)} as many times as
 *     needed to solve each
 *     system for with the various <span class="latex-inline">b</span> vectors and/or 
 *     <span class="latex-inline">B</span> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link LstsqSolver#decompose(MatrixMixin)
 * decompse(A)} will override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link LstsqSolver#solve(MatrixMixin, VectorMixin)}</li>
 *     <li>{@link LstsqSolver#solve(MatrixMixin, MatrixMixin)}</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <p>Minimizing the sum of squared residuals is achieved by computing a QR decomposition of the coefficient
 * matrix <span class="latex-inline">A</span>:
 * <span class="latex-display"><blockquote>
 *    A = QR
 * </blockquote></span>
 * where <span class="latex-inline">Q</span> is a unity matrix and <span class="latex-inline">R</span>
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
 *   since <span class="latex-inline">Q</span> is unity.<br>
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
 *    &\implies &R^HRx &= R^HQ^Hb &&\; \text{ or }\; &R^HRX &= R^HQ^HB \quad \text{ since } Q \text{ is unity.} \\
 *    &\implies &Rx &= Q^Hb &&\; \text{ or }\; & RX &= Q^HB
 * \end{alignat*}
 * \] -->
 *
 * which is easily solved by back-substitution on <span class="latex-inline">R</span>.
 *
 * @see ComplexLstsqSolver
 */
public class ComplexLstsqSolver extends LstsqSolver<CMatrix, CVector> {


    /**
     * Creates a solver for solving complex linear systems of the form <strong>Ax = b</strong> or <strong>AX = B</strong>
     * in a <em>least-squares</em> sense. The system may be under-, well-, or over-determined.
     * The solution is computed by minimizing the sum of squared residuals:
     *
     * <ul>
     *   <li><strong>||Ax - b||<sub>2</sub><sup>2</sup></strong> for vector-based problems, or</li>
     *   <li><strong>||AX - B||<sub>F</sub><sup>2</sup></strong> for matrix-based where problems</li>
     * </ul>
     */
    public ComplexLstsqSolver() {
        super(new ComplexQR(), new ComplexBackSolver());
    }
}
