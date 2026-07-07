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


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.linalg.solvers.LinearMatrixSolver;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * This solver solves linear systems of equations where the coefficient matrix is lower triangular.
 * That is, solves the systems <span class="latex-inline">Lx = b</span> or <span class="latex-inline">LX = B</span>
 * where <span class="latex-inline">L</span> is a lower triangular
 * matrix. This is accomplished using a simple forward substitution.
 *
 * @param <T> Type of coefficient matrix.
 * @param <U> Vector type equivalent to the coefficient matrix.
 * @param <V> Type of the internal storage datastructures in the matrix and vector.
 */
public abstract class ForwardSolver<T extends MatrixMixin<T, ?, U, ?>, U extends VectorMixin<U, T, ?, ?>, V>
        implements LinearMatrixSolver<T, U> {

    // TODO: Investigate alternative methods for determining if the matrix is singular (or near singular).
    //  Since the coefficient matrix is upper-triangular there is no need to compute the determinant explicitly,
    //  we need only check if any individual value along the diagonal is near-zero.

    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    protected static final double RANK_CONDITION = Flag4jConstants.EPS_F64;
    /**
     * Flag indicating if lower-triangular matrices passed to this solver will be unit lower-triangular (true) or simply
     * lower-triangular (false).
     */
    protected final boolean isUnit;
    /**
     * Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular. If false, the matrix will
     * simply be assumed to be lower triangular.
     */
    protected final boolean enforceLower;
    /**
     * Storage for solution in solves which return a matrix.
     */
    T X;
    /**
     * Storage for solution in solves which return a vector.
     */
    U x;
    /**
     * Temporary storage for columns of the solution matrix. This can be used to improve cache performance when columns need to
     * be traveled.
     */
    protected V xCol;


    /**
     * Creates a solver for solving a lower-triangular system.
     * @param isUnit Flag indicating if coefficient matrices passed will be unit lower-triangular or simply lower-triangular in
     *               general.
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    protected ForwardSolver(boolean isUnit, boolean enforceLower) {
        this.isUnit = isUnit;
        this.enforceLower = enforceLower;
    }


    /**
     * Solves a linear system <span class="latex-inline">LX = P</span> for
     * <span class="latex-inline">X</span> where <span class="latex-inline">L</span> is a lower triangular matrix and
     * <span class="latex-inline">P</span> is a permutation matrix.
     * @param L Lower triangular coefficient matrix.
     * @param P Constant permutation matrix.
     * @return The solution of <span class="latex-inline">X</span> for the linear system <span class="latex-inline">LX = P</span>.
     */
    public abstract T solve(T L, PermutationMatrix P);


    /**
     * Ensures passed parameters are valid for the back solver.
     * @param coeff Coefficient matrix in the linear system.
     * @param constantRows Number of rows in the constant vector or matrix.
     * @throws IllegalArgumentException If coeff is not square,  {@code coeff.numRows()!=constantRows}, or if {@code enforceTriU} is
     * true and {@code coeff} is not upper triangular.
     */
    protected void checkParams(T coeff, int constantRows) {
        ValidateParameters.ensureSquare(coeff.getShape());
        ValidateParameters.ensureAllEqual(coeff.numRows(), constantRows);

        if(enforceLower && !coeff.isTriL())
            throw new IllegalArgumentException("Expecting matrix L to be lower triangular.");
    }


    /**
     * Checks if the coefficient matrix is singular based on the computed determinant.
     * @param detAbs Absolute value of computed determinant.
     * @param numRows Number of rows in the coefficient matrix.
     * @param numCols Number of columns in the coefficient matrix.
     */
    protected void checkSingular(double detAbs, int numRows, int numCols) {
        if(detAbs <= RANK_CONDITION*Math.max(numRows, numCols) || Double.isNaN(detAbs))
            throw new SingularMatrixException("Could not solve.");
    }
}
