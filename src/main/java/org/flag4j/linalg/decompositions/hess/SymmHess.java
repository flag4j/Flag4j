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

package org.flag4j.linalg.decompositions.hess;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>Computes the Hessenberg decomposition of a real dense symmetric matrix.
 * <p>The Hessenberg decomposition decomposes a given symmetric matrix <span class="latex-inline">A</span> into the product:
 * <span class="latex-display"><pre>
 *     A = QHQ<sup>T</sup></pre></span>
 * where <span class="latex-inline">Q</span> is an orthogonal matrix and <span class="latex-inline">H</span>
 * is a symmetric tri-diagonal matrix (special case of Hessenburg form)
 * which is similar to <span class="latex-inline">A</span> (i.e., has the same eigenvalues)
 *
 * <p>A matrix <span class="latex-inline">H</span> is in tri-diagonal form if it has all zeros below the first sub-diagonal
 * and above the first super-diagonal.
 *
 * <p>For example, the following matrix is in symmetric tri-diagonal form where each '<span class="latex-inline">&times;</span>'
 * may hold a different value (provided the matrix is symmetric):
 * <span class="latex-replace"><pre>
 *     [[ &times; &times; 0 0 0 ]
 *      [ &times; &times; &times; 0 0 ]
 *      [ 0 &times; &times; &times; 0 ]
 *      [ 0 0 &times; &times; &times; ]
 *      [ 0 0 0 &times; &times; ]]</pre></span>
 *
 * <!-- LATEX: \[ \begin{bmatrix}
 * \times & \times & 0 & 0 & 0 \\
 * \times & \times & \times & 0 & 0 \\
 * 0 & \times & \times & \times & 0 \\
 * 0 & 0 & \times & \times & \times \\
 * 0 & 0 & 0 & \times & \times \\
 * \end{bmatrix} \] -->
 *
 * <h2>Efficiency Considerations:</h2>
 * <ul>
 *     <li>If the orthogonal matrix <span class="latex-inline">Q</span> is not required, setting {@code computeQ = false} in the constructor
 *     <em>may</em> improve performance.</li>
 *     <li>Support for in-place decomposition to reduce memory usage.</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code SymmHess}.</li>
 *     <li>Call {@link #decompose(Matrix)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getH()} and {@link #getQ()}.</li>
 * </ol>
 *
 * @see HermHess
 * @see #getH()
 * @see #getQ()
 */
public class SymmHess extends RealHess {

    /**
     * Flag indicating if an explicit check should be made that the matrix to be decomposed is symmetric.
     */
    protected boolean enforceSymmetric;


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices. By default, the Householder vectors used in the decomposition will be
     * stored so that the full orthogonal <span class="latex-inline">Q</span> matrix can be formed by calling {@link #getQ()}.
     */
    public SymmHess() {
        super();
    }


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices.
     *
     * @param computeQ Flag indicating if the orthogonal <span class="latex-inline">Q</span> matrix from the Hessenberg decomposition should be explicitly computed.
     * if {@code true}, then the <span class="latex-inline">Q</span> matrix will be computed explicitly.
     * If <span class="latex-inline">Q</span> is not needed, setting this to {@code false} <em>may</em> yield an increase in
     * performance.
     */
    public SymmHess(boolean computeQ) {
        super(computeQ);
    }


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices.
     * @param computeQ Flag indicating if the orthogonal <span class="latex-inline">Q</span> matrix from the
     * Hessenberg decomposition should be explicitly computed.
     * if {@code true}, then the <span class="latex-inline">Q</span> matrix will be computed explicitly.
     * If <span class="latex-inline">Q</span> is not
     * needed, setting this to {@code false} <em>may</em> yield an increase in performance.
     * @param enforceSymmetric Flag indicating if an explicit check should be made to ensure any matrix passed to
     * {@link #decompose(Matrix)} is truly symmetric. If {@code true}, an exception will be thrown if the matrix is not symmetric. If
     * {@code false}, the decomposition will proceed under the assumption that the matrix is symmetric whether it actually is or not.
     * If the matrix is not symmetric, then the values in the upper triangular portion of the matrix are taken to be the values.
     */
    public SymmHess(boolean computeQ, boolean enforceSymmetric) {
        super(computeQ);
        this.enforceSymmetric = enforceSymmetric;
    }


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices.
     * @param computeQ Flag indicating if the orthogonal <span class="latex-inline">Q</span> matrix from the Hessenberg decomposition should be explicitly computed.
     * if {@code true}, then the <span class="latex-inline">Q</span> matrix will be computed explicitly. If <span class="latex-inline">Q</span> is not
     * needed, setting this to {@code false} <em>may</em> yield an increase in performance.
     * @param enforceSymmetric Flag indicating if an explicit check should be made to ensure any matrix passed to
     * {@link #decompose(Matrix)} is truly symmetric. If {@code true}, an exception will be thrown if the matrix is not symmetric. If
     * {@code false}, the decomposition will proceed under the assumption that the matrix is symmetric whether it actually is or not.
     * If the matrix is not symmetric, then the values in the upper triangular portion of the matrix are taken to be the values.
     * @param inPlace Flag indicating if the decomposition should be done in-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in-place.</li>
     *     <li>If {@code false}, then the decomposition will be done out-of-place.</li>
     * </ul>
     */
    public SymmHess(boolean computeQ, boolean enforceSymmetric, boolean inPlace) {
        super(computeQ, inPlace);
        this.enforceSymmetric = enforceSymmetric;
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. (Assumed to be a symmetric matrix).
     * @return A reference to this decomposer.
     */
    @Override
    public SymmHess decompose(Matrix src) {
        super.decompose(src);
        return this;
    }


    /**
     * Gets the Hessenberg matrix, <span class="latex-inline">H</span>, from the decomposition. The matrix will be symmetric tri-diagonal.
     * @return The symmetric tri-diagonal (Hessenberg) matrix, <span class="latex-inline">H</span>, from this decomposition.
     */
    @Override
    public Matrix getH() {
        ensureHasDecomposed();
        Matrix H = new Matrix(numRows);
        H.data[0] = transformData[0];

        int idx1;
        int idx0;
        int rowOffset = numRows;

        for(int i=1; i<numRows; i++) {
            idx1 = rowOffset + i;
            idx0 = idx1 - numRows;

            H.data[idx1] = transformData[idx1]; // extract diagonal value.

            // extract off-diagonal values.
            double a = transformData[idx0];
            H.data[idx0] = a;
            H.data[idx1 - 1] = a;

            // Update row index.
            rowOffset += numRows;
        }

        if(numRows > 1) {
            int rowColBase = numRows*numRows - 1;
            H.data[rowColBase] = transformData[rowColBase];
            H.data[rowColBase - 1] = transformData[rowColBase - numRows];
        }

        return H;
    }


    /**
     * Finds the maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} data of the storage array {@link #householderVector} to the data of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     */
    @Override
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;

        // Compute max-abs value in row. (Equivalent to max value in column since matrix is symmetric.)
        int rowU = (j-1)*numRows;
        for(int i=j; i<numRows; i++) {
            double d = householderVector[i] = transformData[rowU + i];
            maxAbs = Math.max(Math.abs(d), maxAbs);
        }

        return maxAbs;
    }


    /**
     * Performs basic setup for the decomposition.
     * @param src The matrix to be decomposed.
     * @throws LinearAlgebraException If the matrix is not symmetric and {@link #enforceSymmetric} is true or if the matrix is not
     * square regardless of the value of {@link #enforceSymmetric}.
     */
    @Override
    protected void setUp(Matrix src) {
        if(enforceSymmetric && !src.isSymmetric()) // If requested, check the matrix is symmetric.
            throw new LinearAlgebraException(this.getClass().getSimpleName() + " only supports symmetric matrices.");
        else
            ValidateParameters.ensureSquareMatrix(src.shape); // Otherwise, Just ensure the matrix is square.

        numRows = numCols = minAxisSize = src.numRows;
        copyUpperTri(src);  // Initializes transform matrix.
        initWorkArrays(numRows);
    }


    /**
     * Copies the upper triangular portion of a matrix to the working matrix {@link #transformMatrix}.
     * @param src The source matrix to decompose of.
     */
    private void copyUpperTri(Matrix src) {
        transformMatrix = new Matrix(numRows);

        // Copy upper triangular portion.
        for(int i=0; i<numRows; i++) {
            int pos = i*numRows + i;
            System.arraycopy(src.data, pos, transformMatrix.data, pos, numRows - i);
        }
    }


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of submatrix for which the Householder reflector was computed for.
     */
    @Override
    protected void updateData(int j) {
        Householder.symmLeftRightMultReflector(transformMatrix, householderVector, currentFactor, j, workArray);

        if(j < numRows) transformMatrix.data[(j-1)*numRows + j] = -norm;

        if(storeReflectors) {
            // Store the Q matrix in the lower portion of the transformation data matrix.
            int col = j-1;

            for(int i=j+1; i<numRows; i++)
                transformMatrix.data[i*numRows + col] = householderVector[i];
        }
    }
}
