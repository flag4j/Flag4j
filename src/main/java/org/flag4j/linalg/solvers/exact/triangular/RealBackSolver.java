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

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.exceptions.SingularMatrixException;


/**
 * This solver solves linear systems of the form <span class="latex-inline">Ux = b</span> or
 * <span class="latex-inline">UX = B</span> where <span class="latex-inline">U</span> is an
 * upper triangular matrix. This system is solved in an exact sense.
 *
 * @see ComplexBackSolver
 */
public class RealBackSolver extends BackSolver<Matrix, Vector, double[]> {

    /**
     * For computing determinant of coefficient matrix during solve.
     */
    protected double det;

    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices. By default, an explicit check
     * will be made that the coefficient matrix is upper triangular. To toggle this, use {@link #RealBackSolver(boolean)}.
     */
    public RealBackSolver() {
        super(true);
    }


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public RealBackSolver(boolean enforceTriU) {
        super(enforceTriU);
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
    @Override
    public RealBackSolver setCheckSingular(boolean checkSingular) {
        super.setCheckSingular(checkSingular);
        return this;
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public double getDet() {
        return det;
    }


    /**
     * Solves the linear system of equations given by <span class="latex-inline">Ux = b</span> where the coefficient matrix
     * <span class="latex-inline">U</span> is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *          this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution to x in the linear system <span class="latex-inline">Ux = b</span>.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e., has a zero on the principle diagonal).
     */
    @Override
    public Vector solve(Matrix U, Vector b) {
        checkParams(U, b.shape);

        double sum, diag;
        int uIndex;
        int n = b.size;
        x = new Vector(U.numRows);
        det = U.data[n*n-1];

        x.data[n-1] = b.data[n-1]/det;

        for(int i=n-2; i>=0; i--) {
            sum = 0;
            uIndex = i*U.numCols;

            diag = U.data[i*(n+1)];
            det*=diag;

            for(int j=i+1; j<n; j++) {
                sum += U.data[uIndex + j]*x.data[j];
            }

            x.data[i] = (b.data[i]-sum)/diag;
        }

        if(checkSingular) checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return x;
    }


    /**
     * Solves the linear system of equations given by <span class="latex-inline">UX = B</span> where the coefficient matrix
     * <span class="latex-inline">U</span> is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param B Matrix of constants in the linear system.
     * @return The solution to X in the linear system <span class="latex-inline">UX = B</span>.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e., has a zero on the principle diagonal).
     */
    @Override
    public Matrix solve(Matrix U, Matrix B) {
        checkParams(U, B.shape);

        double sum, diag;
        int uIndex, xIndex;
        int n = B.numRows;
        double uValue = U.data[n*n-1];
        int rowOffset = (n-1)*B.numCols;
        X = new Matrix(B.shape);
        det = U.data[n*n-1];

        xCol = new double[n];

        for(int j=0; j<B.numCols; j++) {
            X.data[rowOffset + j] = B.data[rowOffset + j]/uValue;

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=n-2; i>=0; i--) {
                sum = 0;
                uIndex = i*U.numCols;
                xIndex = i*X.numCols + j;
                diag = U.data[i*(n+1)];

                if(j==0) det *= diag;

                for(int k=i+1; k<n; k++)
                    sum += U.data[uIndex + k]*xCol[k];

                double value = (B.data[xIndex] - sum) / diag;

                X.data[xIndex] = xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves the linear system of equations given by <span class="latex-inline">UX = I</span> where the coefficient matrix
     * <span class="latex-inline">U</span>
     * is an {@link Matrix#isTriU() upper triangular} matrix and I is the {@link Matrix#isI() identity}
     * matrix of appropriate size. This essentially inverts the upper triangular matrix since
     * <span class="latex-inline">UU<sup>-1</sup> = I</span>.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @return The solution to X in the linear system <span class="latex-inline">UX = B</span>.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e., has a zero on the principle diagonal).
     */
    public Matrix solveIdentity(Matrix U) {
        checkParams(U, U.shape);

        double sum, diag;
        int uIndex, xIndex;
        int n = U.numRows;
        X = new Matrix(U.shape);
        det = U.data[n*n-1];

        xCol = new double[n];
        X.data[X.data.length-1] = 1.0/det;

        for(int j=0; j<n; j++) {
            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=n-2; i>=0; i--) {
                sum = (i == j) ? 1 : 0;
                uIndex = i*n;
                xIndex = uIndex + j;
                uIndex += i+1;
                diag = U.data[i*(n+1)];

                if(j==0) det *= diag;

                for(int k=i+1; k<n; k++)
                    sum -= U.data[uIndex++]*xCol[k];

                double value = sum / diag;
                X.data[xIndex] = xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves a special case of the linear system <span class="latex-inline">UX = L</span> for <span class="latex-inline">X</span>
     * where the coefficient matrix <span class="latex-inline">U</span>
     * is an {@link Matrix#isTriU() upper triangular} matrix and the constant matrix <span class="latex-inline">L</span> is
     * {@link Matrix#isTriL() lower triangular}.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param L Lower triangular constant matrix. This is not explicit checked. If {@code L} is not lower triangular, values above
     *          the principle diagonal will be ignored and the result will still be correctly computed.
     * @return The result of solving the linear system <span class="latex-inline">UX = L</span> for the matrix
     * <span class="latex-inline">X</span>.
     */
    public Matrix solveLower(Matrix U, Matrix L) {
        checkParams(U, L.shape);

        double sum, diag;
        int uIndex, xIndex;
        int n = L.numRows;
        double uValue = U.data[U.data.length-1];
        int rowOffset = (n-1)*n;
        X = new Matrix(L.shape);
        det = uValue;

        xCol = new double[n];

        for(int j=0; j<n; j++) {
            X.data[rowOffset] = L.data[rowOffset++]/uValue;

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=L.numCols-2; i>=0; i--) {
                sum = 0;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                diag = U.data[i*(n+1)];

                if(j==0) det *= diag;

                for(int k=i+1; k<n; k++)
                    sum += U.data[uIndex + k]*xCol[k];

                double value = (L.data[xIndex] - sum) / diag;
                X.data[xIndex] = xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }
}
