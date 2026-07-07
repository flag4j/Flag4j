/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * This solver solves a complex linear system of equations where the coefficient matrix is lower triangular.
 * That is, solves the systems <span class="latex-inline">Lx = b</span> or <span class="latex-inline">LX = B</span>
 * where <span class="latex-inline">L</span> is a lower triangular
 * matrix. This is accomplished using a simple forward substitution.
 *
 * @see RealForwardSolver
 */
public class ComplexForwardSolver extends ForwardSolver<CMatrix, CVector, Complex128[]> {


    /**
     * For computing determinant of lower triangular matrix during solve.
     */
    private Complex128 det;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public ComplexForwardSolver() {
        super(false, true);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not.
     * <ul>
     *     <li>If {@code true}, the coefficient matrix is expected to be unit lower triangular.</li>
     *     <li>If {@code false}, the coefficient matrix is expected to be lower triangular.</li>
     * </ul>
     */
    public ComplexForwardSolver(boolean isUnit) {
        super(isUnit, true);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     * <ul>
     *     <li>If {@code true}, the coefficient matrix is expected to be unit lower triangular.</li>
     *     <li>If {@code false}, the coefficient matrix is expected to be lower triangular.</li>
     * </ul>
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    public ComplexForwardSolver(boolean isUnit, boolean enforceLower) {
        super(isUnit, enforceLower);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public Complex128 getDet() {
        return det;
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix <span class="latex-inline">L</span> and a vector <span class="latex-inline">b</span>.
     * That is, solves the linear system <span class="latex-inline">Lx = b</span> for <span class="latex-inline">x</span> where <span class="latex-inline">L</span> is lower triangular.
     * @param L Lower triangular coefficient matrix <span class="latex-inline">L</span>. {@code L} is assumed to be lower triangular and only entries
     * at and below the principle diagonal will be accessed.
     * @param b Constant vector <span class="latex-inline">b</span>.
     * @return The result of solving the linear system <span class="latex-inline">Lx = b</span> where <span class="latex-inline">L</span> is a lower triangular.
     * @throws SingularMatrixException If {@code L} is singular (i.e., has at least one zero on the principle diagonal).
     */
    @Override
    public CVector solve(CMatrix L, CVector b) {
        return isUnit ? solveUnitLower(L, b) : solveLower(L, b);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix <span class="latex-inline">L</span> and a matrix <span class="latex-inline">B</span>.
     * That is, solves the linear system <span class="latex-inline">LX = B</span> for <span class="latex-inline">X</span> where <span class="latex-inline">L</span> is lower triangular.
     * @param L Lower triangular coefficient matrix <span class="latex-inline">L</span>. {@code L} is assumed to be lower triangular and only entries
     * at and below the principle diagonal will be accessed.
     * @param b Constant matrix <span class="latex-inline">B</span>.
     * @return The result of solving the linear system <span class="latex-inline">LX = B</span> where <span class="latex-inline">L</span> is a lower triangular.
     * @throws SingularMatrixException If {@code L} is singular (i.e., has at least one zero on the principle diagonal).
     */
    @Override
    public CMatrix solve(CMatrix L, CMatrix B) {
        return isUnit ? solveUnitLower(L, B) : solveLower(L, B);
    }


    /**
     * Solves a linear system <span class="latex-inline">LX = P</span> for <span class="latex-inline">X</span> where <span class="latex-inline">L</span> is a lower triangular matrix and
     * <span class="latex-inline">P</span> is a permutation matrix.
     * @param L Lower triangular coefficient matrix <span class="latex-inline">L</span>.
     * @param P Constant permutation matrix <span class="latex-inline">P</span>.
     * @return The solution to <span class="latex-inline">X</span> in the linear system <span class="latex-inline">LX = P</span>.
     */
    @Override
    public CMatrix solve(CMatrix L, PermutationMatrix P) {
        return isUnit ? solveUnitPerm(L, P) : solvePerm(L, P);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix <span class="latex-inline">L</span> and the identity matrix.
     * That is, solves the linear system <span class="latex-inline">LX = I</span> where <span class="latex-inline">L</span> is a lower triangular matrix and
     * <span class="latex-inline">I</span> is the appropriately sized identity matrix.
     * @param L Lower triangular coefficient matrix, <span class="latex-inline">L</span>. If {@code L} is not lower triangular, it will be treated
     *          as if it were and only items in the lower triangular portion will be accessed.
     * @return The result of solving the linear system <span class="latex-inline">LX = B</span> where <span class="latex-inline">L</span> is a lower triangular matrix.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular
     * (i.e., has at least one zero on the principle diagonal).
     */
    public CMatrix solveIdentity(CMatrix L) {
        return isUnit ? solveUnitLowerIdentity(L) : solveLowerIdentity(L);
    }


    /**
     * Solves the linear system <span class="latex-inline">Lx = b</span> where <span class="latex-inline">L</span> is <em>unit</em> lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b Vector of constants in the linear system.
     * @return The solution of x for the linear system <span class="latex-inline">Lx = b</span>.
     */
    private CVector solveUnitLower(CMatrix L, CVector b) {
        checkParams(L, b.size);

        Complex128 sum;
        int lIndexStart;
        CVector x = new CVector(L.numRows);
        det = new Complex128(L.numRows); // Since it is unit lower, matrix has full rank.
        x.data[0] = b.data[0];

        for(int i=1; i<L.numRows; i++) {
            sum = Complex128.ZERO;
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--)
                sum = sum.add(L.data[lIndexStart + j].mult(x.data[j]));

            x.data[i] = b.data[i].sub(sum);
        }

        return x; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular and the constant matrix
     * is the identity matrix.
     * @param L Unit lower triangular matrix.
     * @return The solution of <span class="latex-inline">X</span> for the linear system <span class="latex-inline">LX = I</span>.
     */
    private CMatrix solveUnitLowerIdentity(CMatrix L) {
        checkParams(L, L.numRows);

        Complex128 sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(L.shape);
        det = new Complex128(L.numRows); // Since it is unit lower, matrix has full rank.
        xCol = new Complex128[L.numRows];

        X.data[0] = Complex128.ONE;

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? Complex128.ONE : Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;

                for(int k=0; k<i; k++)
                    sum = sum.sub(L.data[lIndexStart++].mult(xCol[k]));

                X.data[xIndex] = xCol[i] = sum;
            }
        }

        return X; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system <span class="latex-inline">LX = I</span> where the coefficient matrix L is lower triangular and the
     * constant matrix I is the appropriately sized identity matrix.
     * @param L Unit lower triangular matrix (Note, this is not checked).
     *          If {@code L} is not lower triangular, it will be treated as if it were. No error will be thrown.
     * @return The solution of X for the linear system <span class="latex-inline">LX = I</span>.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e., has a zero on the
     * principle diagonal).
     */
    private CMatrix solveLowerIdentity(CMatrix L) {
        checkParams(L, L.numRows);

        Complex128 sum, diag;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(L.shape);
        det = L.data[0];
        xCol = new Complex128[L.numRows];

        X.data[0] = L.data[0].invert();

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                if(j==0) det = det.mult(L.data[i*L.numCols + i]);

                sum = (i==j) ? Complex128.ONE : Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;
                diag = L.data[i*(L.numCols + 1)];

                for(int k=0; k<i; k++)
                    sum = sum.sub(L.data[lIndexStart++].mult(xCol[k]));

                Complex128 value = sum.div(diag);
                X.data[xIndex] = xCol[i] = value;
            }
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b Vector of constants in the linear system.
     * @return The solution of x for the linear system <span class="latex-inline">Lx = b</span>.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e., has a zero on the
     * principle diagonal).
     */
    private CVector solveLower(CMatrix L, CVector b) {
        checkParams(L, b.size);

        Complex128 sum, diag;
        int lIndexStart;
        CVector x = new CVector(L.numRows);
        det = L.data[0];
        x.data[0] = b.data[0].div(L.data[0]);

        for(int i=1; i<L.numRows; i++) {
            sum = Complex128.ZERO;
            lIndexStart = i*L.numCols;
            diag = L.data[i*(L.numCols + 1)];
            det = det.mult(diag);

            for(int j=i-1; j>-1; j--)
                sum = sum.add(L.data[lIndexStart + j].mult(x.data[j]));

            x.data[i] = b.data[i].sub(sum).div(diag);
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution to <span class="latex-inline">X</span> in the linear system <span class="latex-inline">LX = B</span>.
     */
    private CMatrix solveUnitLower(CMatrix L, CMatrix B) {
        checkParams(L, B.numRows);

        Complex128 sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);
        det = new Complex128(L.numRows); // Since it is unit lower, matrix has full rank.
        xCol = new Complex128[L.numRows];

        for(int j=0; j<B.numCols; j++) {
            X.data[j] = B.data[j];

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;

                for(int k=0; k<i; k++)
                    sum = sum.add(L.data[lIndexStart++].mult(xCol[k]));

                Complex128 value = B.data[xIndex].sub(sum);
                X.data[xIndex] = xCol[i] = value;
            }
        }

        return X; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of X for the linear system <span class="latex-inline">LX = B</span>.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e., has a zero on the
     * principle diagonal).
     */
    private CMatrix solveLower(CMatrix L, CMatrix B) {
        checkParams(L, B.numRows);

        Complex128 sum;
        Complex128 diag;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);
        det = L.data[0];
        xCol = new Complex128[L.numRows];

        for(int j=0; j<B.numCols; j++) {
            X.data[j] = B.data[j].div(L.data[0]);

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;
                diag = L.data[i*(L.numCols + 1)];

                if(j==0) det = det.mult(diag);

                for(int k=0; k<i; k++)
                    sum = sum.add(L.data[lIndexStart++].mult(xCol[k]));

                Complex128 value = B.data[xIndex].sub(sum).div(diag);
                X.data[xIndex] = xCol[i] = value;
            }
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }


    /**
     * Solves a linear system <span class="latex-inline">LX = P</span> where the coefficient matrix <span
     * class="latex-inline">L</span> is lower triangular and the
     * constant matrix <span class="latex-inline">P</span> is a permutation matrix.
     * @param L Lower triangular coefficient matrix <span class="latex-inline">L</span>.
     * @return The solution of <span class="latex-inline">X</span> to the linear system <span class="latex-inline">LX = P</span>.
     * @throws SingularMatrixException If {@code L} is singular (i.e., has a zero on the principle diagonal).
     */
    private CMatrix solvePerm(CMatrix L, PermutationMatrix P) {
        checkParams(L, P.size);

        final int n = P.size;
        int[] perm = P.getPermutation();
        X = new CMatrix(P.size, P.size);
        det = L.data[0];  // Since it is unit lower, matrix has full rank.
        xCol = new Complex128[n];

        for (int j = 0; j < n; j++) {
            Complex128 bVal0 = (perm[0] == j) ? Complex128.ONE : Complex128.ZERO;
            X.data[j] = bVal0.div(L.data[0]);

            // Temporarily store column for better cache performance on innermost loop.
            for (int k = 0; k < n; k++)
                xCol[k] = X.data[k*n + j];

            for (int i = 1; i < n; i++) {
                Complex128 sum = Complex128.ZERO;
                int lIndexStart = i*n;

                // Accumulate the dot product.
                for (int k = 0; k < i; k++)
                    sum = sum.add(L.data[lIndexStart + k].mult(xCol[k]));

                Complex128 diag = L.data[i*(n + 1)];

                if (j == 0) det = det.mult(diag);

                Complex128 bVal = (perm[i] == j) ? Complex128.ONE : Complex128.ZERO;
                Complex128 value = bVal.sub(sum).div(diag);

                X.data[lIndexStart + j] = xCol[i] = value;
            }
        }

        // If you want to check for singularity:
        checkSingular(det.mag(), n, n);

        return X;
    }


    /**
     * Solves a linear system <span class="latex-inline">LX = P</span> where the coefficient matrix <span
     * class="latex-inline">L</span> is unit-lower triangular and the
     * constant matrix <span class="latex-inline">P</span> is a permutation matrix.
     * @param L Unit lower triangular coefficient matrix <span class="latex-inline">L</span>.
     * @return The solution of <span class="latex-inline">X</span> to the linear system <span class="latex-inline">LX = P</span>.
     */
    private CMatrix solveUnitPerm(CMatrix L, PermutationMatrix P) {
        checkParams(L, P.size);

        final int n = P.size;
        int[] perm = P.getPermutation();
        X = new CMatrix(P.size, P.size);
        det = new Complex128(L.numRows);  // Since it is unit lower, matrix has full rank.
        xCol = new Complex128[n];

        for (int j = 0; j < n; j++) {
            Complex128 bVal0 = (perm[0] == j) ? Complex128.ONE : Complex128.ZERO;
            X.data[j] = bVal0;

            // Temporarily store column for better cache performance on innermost loop.
            for (int k = 0; k < n; k++)
                xCol[k] = X.data[k*n + j];

            for (int i = 1; i < n; i++) {
                Complex128 sum = Complex128.ZERO;
                int lIndexStart = i*n;   // Start of row i in L

                // Accumulate the dot product
                for (int k = 0; k < i; k++)
                    sum = sum.add(L.data[lIndexStart + k].mult(xCol[k]));

                Complex128 bVal = (perm[i] == j) ? Complex128.ONE : Complex128.ZERO;
                Complex128 value = bVal.sub(sum);

                X.data[lIndexStart + j] = xCol[i] = value;
            }
        }

        // No need to check if matrix is singular since it has full rank.
        return X;
    }
}
