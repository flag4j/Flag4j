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

package org.flag4j.linalg.decompositions.svd;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.util.Flag4jConstants;

import java.util.Arrays;

/**
 * <p>This abstract class specifies methods for computing the singular value decomposition (SVD) of a matrix.
 *
 * <p>That is, decomposes a rectangular matrix <span class="latex-inline">M</span> into
 * <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span>
 * where <span class="latex-inline">U</span> and <span class="latex-inline">V</span> are
 * unitary matrices whose columns are the left and right singular vectors of <span class="latex-inline">M</span> and 
 * <span class="latex-inline">&Sigma;</span> is a rectangular
 * diagonal matrix containing the singular values of <span class="latex-inline">M</span>.
 *
 * <p>The SVD may also be used to compute the (numerical) rank of the matrix using {@link #getRank()}.
 *
 * <p>The SVD proceeds by an iterative algorithm with possible random behavior. For reproducibility, constructors
 * support specifying a seed for the pseudo-random number generator.
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate a concrete instance of {@code SVD}.</li>
 *     <li>Call {@link #decompose(MatrixMixin)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getU()} and {@link #getS()}.</li>
 * </ol>
 *
 * <h2>Efficiency Considerations:</h2>
 * If singular vectors are not required, setting {@code computeUV = false} <em>may</em> improve performance.
 *
 * @see #getU()
 * @see #getS()
 * @see #getV()
 * @see #getRank()
 *
 * @param <T> The type of the matrix to compute the singular value decomposition of.
 */
public abstract class SVD<T extends MatrixMixin<T, ?, ?, ?>> extends Decomposition<T> {

    /**
     * Seed to use in pseudo-random number generator.
     */
    protected long seed;
    /**
     * Flag indicating if seeding should be used in pseudo-random operations.
     * <ul>
     *     <li>If {@code true}, then seeding <em>will</em> be used.</li>
     *     <li>If {@code false}, then seeding <em>will not</em> be used.</li>
     * </ul>
     */
    protected boolean useSeed;
    /**
     * Flag which indicates if the singular vectors should be computed in addition to the singular values.
     */
    protected boolean computeUV;
    /**
     * Flag which indicates if the reduced (or full) SVD should be computed.
     */
    protected boolean reduced;
    /**
     * The unitary matrix <span class="latex-inline">U</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     */
    protected T U;
    /**
     * The rectangular diagonal <span class="latex-inline">&Sigma;</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     */
    protected Matrix S;
    /**
     * The unitary matrix <span class="latex-inline">V</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     */
    protected T V;
    /**
     * The rank of the matrix being decomposed. This is calculated as a byproduct of the decomposition.
     */
    protected int rank;


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeUV A flag which indicates if the unitary matrices <span class="latex-inline">U</span> and <span class="latex-inline">V</span> should be computed
     * (i.e., the singular vectors).
     * <ul>
     *     <li>If {@code true}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span> matrices will be computed.</li>
     *     <li>If {@code false}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span> matrices  will <em>not</em> be computed. If it is not
     *     needed, this <em>may</em> provide a performance improvement.</li>
     * </ul>
     * @param reduced Flag which indicates if the reduced (or full) SVD should be computed.
     * <ul>
     *     <li>If {@code true}, reduced SVD is computed.</li>
     *     <li>If {@code false}, the full SVD is computed.</li>
     * </ul>
     */
    protected SVD(boolean computeUV, boolean reduced) {
        this.computeUV = computeUV;
        this.reduced = reduced;
        this.useSeed = false;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeUV A flag which indicates if the unitary matrices <span class="latex-inline">U</span> and <span class="latex-inline">V</span> should be computed
     * (i.e., the singular vectors).
     * <ul>
     *     <li>If {@code true}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span> matrices will be computed.</li>
     *     <li>If {@code false}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span> matrices  will <em>not</em> be computed. If it is not
     *     needed, this <em>may</em> provide a performance improvement.</li>
     * </ul>
     * @param reduced Flag which indicates if the reduced (or full) SVD should be computed.
     * <ul>
     *     <li>If {@code true}, reduced SVD is computed.</li>
     *     <li>If {@code false}, the full SVD is computed.</li>
     * </ul>
     * @param seed Seed to use in pseudo-random number generators. Setting this will allow for reproducibility
     * between multiple calls with the same inputs.
     */
    protected SVD(boolean computeUV, boolean reduced, long seed) {
        this.computeUV = computeUV;
        this.reduced = reduced;
        this.seed = seed;
        this.useSeed = true;
    }


    /**
     * Gets the unitary matrix <span class="latex-inline">U</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     * @return <span class="latex-inline">U</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     */
    public T getU() {
        ensureHasDecomposed();
        return U;
    }


    /**
     * Gets the diagonal matrix <span class="latex-inline">&Sigma;</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     * @return <span class="latex-inline">&Sigma;</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     */
    public Matrix getS() {
        ensureHasDecomposed();
        return S;
    }


    /**
     * Gets the singular values of the last matrix decomposed.
     *
     * @return The singular values of the last matrix decomposed.
     */
    public Vector getSingularValues() {
        ensureHasDecomposed();
        return S.getDiag();
    }


    /**
     * Gets the unitary matrix <span class="latex-inline">V</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD.
     * @return <span class="latex-inline">V</span> corresponding to <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> in the SVD. Note that the Hermitian transpose has
     * <em>not</em> been computed.
     */
    public T getV() {
        ensureHasDecomposed();
        return V;
    }


    /**
     * Gets the rank of the last matrix decomposed. This is computed as a byproduct of the decomposition.
     * @return The rank of the last matrix decomposed.
     */
    public int getRank() {
        ensureHasDecomposed();
        return rank;
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public SVD<T> decompose(T src) {
        T B = invDirectSum(src); // Convert the problem to an eigenvalue problem.

        double[] singularVals = new double[B.numRows()];
        int stopIdx;
        T singularVecs = null;

        if(computeUV) {
            // Compute both the eigenvalues and eigenvectors of B.
            singularVecs = makeEigenPairs(B, singularVals);
        } else {
            // Only compute the eigenvalues of B.
            makeEigenVals(B, singularVals);
        }

        computeRank(src.numRows(), src.numCols(), singularVals);
        stopIdx = reduced ? rank : Math.min(src.numRows(), src.numCols());

        if(computeUV) initUV(src.getShape(), stopIdx); // Initialize the U and V matrices.
        S = new Matrix(stopIdx); // initialize the S matrix.

        int idx = 0;
        int j = 0;

        while(j < singularVals.length && idx < stopIdx) {
            double sigma = singularVals[j];

            if(sigma > 0) {
                S.set(sigma, idx, idx);

                if(computeUV && singularVecs != null) {
                    // Extract left and right singular vectors and normalize.
                    extractNormalizedCols(singularVecs, idx);
                }

                idx++;
            }

            j++;
        }

        super.hasDecomposed = true;
        return this;
    }


    /**
     * Computes the inverse direct sum of a matrix and its Hermitian transpose.
     * @param src Matrix to inverse direct add with its Hermitian transpose.
     * @return The inverse direct sum of the {@code src} matrix with its Hermitian transpose.
     */
    protected abstract T invDirectSum(T src);


    /**
     * Computes the rank of the matrix being decomposed using the singular values of the matrix.
     * @param rows The number of rows in the original source matrix.
     * @param cols The number of columns in the original source matrix.
     * @param singularValues The singular values of the original source matrix.
     */
    protected void computeRank(int rows, int cols, double[] singularValues) {
        rank = 0; // Ensure the rank is reset.

        double[] sorted = new double[singularValues.length];
        System.arraycopy(singularValues, 0, sorted, 0, singularValues.length);
        Arrays.sort(sorted);

        // Tolerance for considering a singular value zero.
        // TODO: Make tolerance configurable and use this as the default tolerance.
        double tol = 2.0*Math.max(rows, cols)*Flag4jConstants.EPS_F64*sorted[sorted.length - 1];

        for(double val : singularValues)
            if(val > tol) rank++;
    }


    /**
     * Gets the eigen values and vectors of symmetric block matrix which corresponds
     * to the singular values and vectors of the matrix being decomposed.
     * @param B Symmetric block matrix to compute the eigenvalues of.
     * @param eigVals Storage for eigenvalues.
     * @return The eigenvalues and eigenvectors of the symmetric block matrix which corresponds
     * to the singular values and vectors of the matrix being decomposed.
     */
    protected abstract T makeEigenPairs(T B, double[] eigVals);


    /**
     * Gets the eigen values of the symmetric block matrix which corresponds
     * to the singular values of the matrix being decomposed.
     * @param B Symmetric block matrix to compute the eigenvalues of.
     * @param eigVals Storage for eigenvalues.
     */
    protected abstract void makeEigenVals(T B, double[] eigVals);


    /**
     * Initializes the unitary <span class="latex-inline">U</span> and <span class="latex-inline">V</span> matrices for the SVD.
     * @param src Shape of the source matrix being decomposed.
     * @param cols The number of columns for <span class="latex-inline">U</span> and <span class="latex-inline">V</span>.
     */
    protected abstract void initUV(Shape src, int cols);


    /**
     * Extracts the singular vectors, normalizes them and sets the columns of <span class="latex-inline">U</span>
     * and <span class="latex-inline">V</span> to be the left/right singular vectors.
     * @param singularVecs Computed left and right singular vectors.
     * @param j Index of the column in <span class="latex-inline">U</span> and <span class="latex-inline">V</span> to set.
     */
    protected abstract void extractNormalizedCols(T singularVecs, int j);
}
