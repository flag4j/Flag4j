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
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.DirectSum;
import org.flag4j.linalg.Eigen;


/**
 * <p>Instances of this class can be used to compute the singular value decomposition (SVD) of a
 * {@link CMatrix complex dense matrix}.
 *
 *
 * <p>That is, decomposes a rectangular matrix <span class="latex-inline">M</span> into <span class="latex-inline">M=U&Sigma;V<sup>H</sup></span> 
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
 *     <li>Instantiate a concrete instance of {@code ComplexSVD}.</li>
 *     <li>Call {@link SVD#decompose(MatrixMixin)} to perform the factorization.</li>
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
 */
public class ComplexSVD extends SVD<CMatrix> {


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix. The left and right singular
     * vectors will be computed.
     */
    public ComplexSVD() {
        super(true, false);
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     *
     * @param computeUV A flag which indicates if the unitary matrices <span class="latex-inline">U</span> and
     * <span class="latex-inline">V</span> should be computed
     * (i.e., the singular vectors).
     * <ul>
     *     <li>If {@code true}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     *     matrices will be computed.</li>
     *     <li>If {@code false}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     *     matrices  will <em>not</em> be computed. If it is not
     *     needed, this <em>may</em> provide a performance improvement.</li>
     * </ul>
     */
    public ComplexSVD(boolean computeUV) {
        super(computeUV, false);
    }


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix.
     * @param computeUV A flag which indicates if the unitary matrices <span class="latex-inline">U</span>
     * and <span class="latex-inline">V</span> should be computed
     * (i.e., the singular vectors).
     * <ul>
     *     <li>If {@code true}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     *     matrices will be computed.</li>
     *     <li>If {@code false}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     *     matrices  will <em>not</em> be computed. If it is not
     *     needed, this <em>may</em> provide a performance improvement.</li>
     * </ul>
     * @param reduced Flag which indicates if the reduced (or full) SVD should be computed.
     * <ul>
     *     <li>If {@code true}, reduced SVD is computed.</li>
     *     <li>If {@code false}, the full SVD is computed.</li>
     * </ul>
     */
    public ComplexSVD(boolean computeUV, boolean reduced) {
        super(computeUV, reduced);
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeUV A flag which indicates if the unitary matrices <span class="latex-inline">U</span> and <span class="latex-inline">V</span> should be computed
     * (i.e., the singular vectors).
     * <ul>
     *     <li>If {@code true}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     *     matrices will be computed.</li>
     *     <li>If {@code false}, the <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     *     matrices  will <em>not</em> be computed. If it is not
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
    public ComplexSVD(boolean computeUV, boolean reduced, long seed) {
        super(computeUV, reduced, seed);
    }


    /**
     * Computes the inverse direct sum of a matrix and its Hermitian transpose.
     *
     * @param src Matrix to inverse direct add with its Hermitian transpose.
     *
     * @return The inverse direct sum of the {@code src} matrix with its Hermitian transpose.
     */
    @Override
    protected CMatrix invDirectSum(CMatrix src) {
        return DirectSum.directSum(src, src.H());
    }


    /**
     * Gets the eigen values and vectors of symmetric the block matrix which corresponds
     * to the singular values and vectors of the matrix being decomposed.
     *
     * @param B       Symmetric block matrix to compute the eigenvalues of.
     * @param eigVals Storage for eigenvalues.
     * @return The eigenvalues and eigenvectors of the symmetric block matrix which corresponds
     * to the singular values and vectors of the matrix being decomposed.
     */
    @Override
    protected CMatrix makeEigenPairs(CMatrix B, double[] eigVals) {
        CMatrix[] pairs = useSeed ? Eigen.getEigenPairs(B, seed) : Eigen.getEigenPairs(B);
        double[] vals = pairs[0].toReal().data;
        System.arraycopy(vals, 0, eigVals, 0, eigVals.length);

        return pairs[1];
    }


    /**
     * Gets the eigen values of the symmetric block matrix which corresponds
     * to the singular values of the matrix being decomposed.
     *
     * @param B       Symmetric block matrix to compute the eigenvalues of.
     * @param eigVals Storage for eigenvalues.
     */
    @Override
    protected void makeEigenVals(CMatrix B, double[] eigVals) {
        CVector valsTest = useSeed ? Eigen.getEigenValues(B, seed) : Eigen.getEigenValues(B);
        double[] vals = valsTest.toReal().data;
        System.arraycopy(vals, 0, eigVals, 0, eigVals.length);
    }


    /**
     * Initializes the unitary <span class="latex-inline">U</span> and <span class="latex-inline">V</span>
     * matrices for the SVD.
     *
     * @param src  Shape of the source matrix being decomposed.
     * @param cols The number of columns for <span class="latex-inline">U</span> and <span class="latex-inline">V</span>.
     */
    @Override
    protected void initUV(Shape src, int cols) {
        U = new CMatrix(src.getSize(0), cols);
        V = new CMatrix(src.getSize(1), cols);
    }


    /**
     * Extracts the singular vectors, normalizes them and sets the columns of <span class="latex-inline">U</span>
     * and <span class="latex-inline">V</span> to be the left/right singular vectors.
     *
     * @param singularVecs Computed left and right singular vectors.
     * @param j            Index of the column in <span class="latex-inline">U</span> and
     * <span class="latex-inline">V</span> to set.
     */
    @Override
    protected void extractNormalizedCols(CMatrix singularVecs, int j) {
        // Extract left and right singular vectors and normalize.
        V.setCol(singularVecs.getCol(2*j, 0, V.numRows).normalize(), j);
        U.setCol(singularVecs.getCol(2*j, V.numRows, singularVecs.numRows()).normalize(), j);
    }
}
