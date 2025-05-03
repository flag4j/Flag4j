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


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.SingularMatrixException;


/**
 * This solver solves linear systems of the form <span class="latex-inline">Ux = b</span> or
 * <span class="latex-inline">UX = B</span> where <span class="latex-inline">U</span> is an
 * upper triangular matrix. This system is solved in an exact sense.
 *
 * @see RealBackSolver
 */
public class ComplexBackSolver extends BackSolver<CMatrix, CVector, Complex128[]> {
    
    /**
     * For computing determinant of coefficient matrix during solve.
     */
    protected Complex128 det;


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices. Note, by default no check will
     * be made to ensure the coefficient matrix is upper triangular. If you would like to enforce this, see
     * {@link #ComplexBackSolver(boolean)}.
     */
    public ComplexBackSolver() {
        super(true);
    }


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public ComplexBackSolver(boolean enforceTriU) {
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
    public ComplexBackSolver setCheckSingular(boolean checkSingular) {
        super.setCheckSingular(checkSingular);
        return this;
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public Complex128 getDet() {
        return det;
    }


    /**
     * Solves the linear system of equations given by <span class="latex-inline">Ux = b</span> where the coefficient matrix
     * <span class="latex-inline">U</span> is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *          this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution to <span class="latex-inline">x</span></> in the linear system <span class="latex-inline">Ux = b</span>.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e., has a zero on the principle diagonal).
     */
    @Override
    public CVector solve(CMatrix U, CVector b) {
        checkParams(U, b.shape);

        Complex128 sum;
        int uIndex;
        int n = b.size;
        x = new CVector(U.numRows);
        det = U.data[n*n-1];

        x.data[n-1] = b.data[n-1].div(det);

        for(int i=n-2; i>-1; i--) {
            sum = Complex128.ZERO;
            uIndex = i*U.numCols;
            Complex128 diag = U.data[i*(n+1)];
            det = det.mult(diag);

            for(int j=i+1; j<n; j++)
                sum = sum.add(U.data[uIndex + j].mult(x.data[j]));

            x.data[i] = (b.data[i].sub(sum)).div(diag);
        }

        if(checkSingular) checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return x;
    }


    /**
     * Solves the linear system of equations given by <span class="latex-inline">UX = B</span> where the coefficient matrix
     * <span class="latex-inline">U</span> is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param B Matrix of constants in the linear system.
     * @return The solution to <span class="latex-inline">X</span> in the linear system <span class="latex-inline">UX = B</span>.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e., has a zero on the principle diagonal).
     */
    @Override
    public CMatrix solve(CMatrix U, CMatrix B) {
        checkParams(U, B.shape);

        Complex128 sum, diag;
        int uIndex, xIndex;
        int n = B.numRows;
        X = new CMatrix(B.shape);
        det = U.data[U.data.length-1];

        xCol = new Complex128[n];

        for(int j=0; j<B.numCols; j++) {
            X.data[(n-1)*X.numCols + j] = B.data[(n-1)*X.numCols + j].div(U.data[n*n-1]);
            det = det.mult(U.data[j*(n+1)]);

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=n-2; i>-1; i--) {
                sum = Complex128.ZERO;
                uIndex = i*U.numCols;
                xIndex = i*X.numCols + j;
                diag = U.data[i*(n+1)];

                if(j==0) det = det.mult(diag);

                for(int k=i+1; k<n; k++)
                    sum = sum.add(U.data[uIndex + k].mult(xCol[k]));

                Complex128 value = B.data[xIndex].sub(sum).div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

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
    public CMatrix solveIdentity(CMatrix U) {
        checkParams(U, U.shape);

        Complex128 sum, diag;
        int uIndex, xIndex;
        int n = U.numRows;
        X = new CMatrix(U.shape);
        det = U.data[U.data.length-1];

        xCol = new Complex128[n];
        X.data[X.data.length-1] = det.multInv();

        for(int j=0; j<U.numCols; j++) {

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=n-2; i>-1; i--) {
                sum = (i == j) ? Complex128.ONE : Complex128.ZERO;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                uIndex += i+1;
                diag = U.data[i*(n+1)];

                if(j==0) det = det.mult(diag);

                for(int k=i+1; k<n; k++) {
                    sum = sum.sub(U.data[uIndex++].mult(xCol[k]));
                }

                Complex128 value = sum.div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

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
    public CMatrix solveLower(CMatrix U, CMatrix L) {
        checkParams(U, L.shape);

        Complex128 sum, diag;
        int uIndex, xIndex;
        int n = L.numRows;
        Complex128 uValue = U.data[n*n-1];
        int rowOffset = (n-1)*n;
        X = new CMatrix(L.shape);
        det = U.data[U.data.length-1];

        xCol = new Complex128[n];

        for(int j=0; j<n; j++) {
            X.data[rowOffset] = L.data[rowOffset++].div(uValue);

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=L.numCols-2; i>=0; i--) {
                sum = Complex128.ZERO;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                diag = U.data[i*(n+1)];

                if(j==0) det = det.mult(diag);

                for(int k=i+1; k<n; k++)
                    sum = sum.add(U.data[uIndex + k].mult(xCol[k]));

                Complex128 value = L.data[xIndex].sub(sum).div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }
}
