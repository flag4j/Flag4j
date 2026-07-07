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

package org.flag4j.linalg.solvers.exact.triangular;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.linalg.solvers.LinearMatrixSolver;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * Base class for solvers which solve a linear system of equations <span class="latex-inline">Ux = b</span> or
 * <span class="latex-inline">UX = B</span> where <span class="latex-inline">U</span> is an
 * upper triangular matrix. This system is solved in an exact sense.
 *
 * @param <T> Type of matrix to decompose.
 * @param <U> Vector type equivalent of matrix.
 * @param <V> Type of internal storage for the matrix and vector.
 */
public abstract class BackSolver<T extends MatrixMixin<T, ?, U, ?>, U extends VectorMixin<U, T, ?, ?>, V>
        implements LinearMatrixSolver<T, U> {

    // TODO: Investigate alternative methods for determining if the matrix is singular (or near singular).
    //  Since the coefficient matrix is upper-triangular there is no need to compute the determinant explicitly,
    //  we need only check if any individual value along the diagonal is near-zero.

    /**
     * For storing matrix results.
     */
    protected T X;
    /**
     * For storing vector results.
     */
    protected U x;
    /**
     * For temporary storage of matrix columns to help improve cache performance.
     */
    protected V xCol;
    /**
     * Flag indicating if determinant should be computed.
     */
    protected final boolean enforceTriU;
    /**
     * Flag indicating if an explicit check should be made that the matrix is singular (or near singular).
     * <ul>
     *     <li>If {@code true}, an explicit singularity check will be made.</li>
     *     <li>If {@code false}, <em>no</em> check will be made.</li>
     * </ul>
     */
    protected boolean checkSingular = true;


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public BackSolver(boolean enforceTriU) {
        this.enforceTriU = enforceTriU;
    }


    /**
     * Sets a flag indicating if an explicit check should be made that the coefficient matrix is singular.
     *
     * @param checkSingular Flag indicating if an explicit check should be made that the matrix is singular (or near singular).
     * <ul>
     *     <li>If {@code true}, an explicit singularity check will be made.</li>
     *     <li>If {@code false}, <em>no</em> check will be made.</li>
     * </ul>
     *
     * @return A reference to this back solver instance.
     */
    protected BackSolver<T, U, V> setCheckSingular(boolean checkSingular) {
        this.checkSingular = checkSingular;
        return this;
    }


    /**
     * Ensures passed parameters are valid for the back solver.
     * @param coeff Coefficient matrix in the linear system.
     * @param constantRows Shape of the constant vector or matrix.
     * @throws IllegalArgumentException If coeff is not square,  {@code coeff.numRows()!=constantRows}, or if {@code enforceTriU} is
     * true and {@code coeff} is not upper triangular.
     */
    protected void checkParams(T coeff, Shape constantShape) {
        ValidateParameters.ensureSquare(coeff.getShape());

        if(coeff.numRows() != constantShape.getSize(0)) {
            throw new IllegalArgumentException("Expecting coefficient matrix rows to match " +
                    "constant vector/matrix entries/rows " +
                    "\nbut got shapes: " + coeff.getShape() + ", " + constantShape + ".");
        }

        if(enforceTriU && !coeff.isTriU())
            throw new IllegalArgumentException("Expecting matrix U to be upper triangular.");
    }


    /**
     * Checks if the coefficient matrix is singular based on the computed determinant.
     * @param detAbs Absolute value of computed determinant.
     * @param numRows Number of rows in the coefficient matrix.
     * @param numCols Number of columns in the coefficient matrix.
     */
    protected void checkSingular(double detAbs, int numRows, int numCols) {
        if(detAbs <= Flag4jConstants.EPS_F64*Math.max(numRows, numCols) || Double.isNaN(detAbs))
            throw new SingularMatrixException("Could not solve system.");
    }
}
