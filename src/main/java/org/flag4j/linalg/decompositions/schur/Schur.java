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

package org.flag4j.linalg.decompositions.schur;


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.linalg.decompositions.balance.Balancer;
import org.flag4j.linalg.decompositions.unitary.UnitaryDecomposition;
import org.flag4j.rng.RandomComplex;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>An abstract base class for computing the Schur decomposition of a square matrix.
 *
 * <p>The Schur decomposition decomposes a given square matrix <span class="latex-inline">A</span> into:
 * <span class="latex-display"><pre>
 *     A = UTU<sup>H</sup></pre></span>
 * where <span class="latex-inline">U</span> is a unitary (or orthogonal for real matrices) matrix <span class="latex-inline">T</span>
 * is a quasi-upper triangular matrix known as the <em>Schur form</em> of <span class="latex-inline">A</span>.
 * This means <span class="latex-inline">T</span> is upper triangular except
 * for possibly <span class="latex-inline">2&times;2</span> blocks along its diagonal, which correspond to complex conjugate
 * pairs of eigenvalues.
 *
 * <p>The Schur decomposition proceeds by an iterative algorithm with possible random behavior. For reproducibility, constructors
 * support specifying a seed for the pseudo-random number generator.
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate a con concrete instance of {@code Schur}.</li>
 *     <li>Call {@link #decompose(MatrixMixin)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getU()} and {@link #getT()}.</li>
 * </ol>
 *
 * <h2>Efficiency Considerations:</h2>
 * If eigenvectors are not required, setting {@code computeU = false} <em>may</em> improve performance.
 *
 * <p>This class was inspired by code from the <a href="http://ejml.org/wiki/index.php?title=Main_Page">EJML</a>
 * library and the description of the Francis implicit double-shifted QR algorithm from
 * <a href="https://www.math.wsu.edu/faculty/watkins/books.html">Fundamentals of Matrix
 * Computations 3rd Edition by David S. Watkins</a>.
 *
 * @implNote This decomposition is performed using the <strong>implicit double-shift QR algorithm</strong>, which iteratively
 * reduces the matrix to Schur form using orthogonal transformations. In addition to this, random shifting is used in cases where
 * normal convergence fails.
 *
 * <p>As a preprocessing step to improve conditioning and stability, the matrix is first {@link Balancer balanced} then reduced to
 * <strong>Hessenberg form</strong> via a {@link UnitaryDecomposition}.
 *
 * @param <T> The type of matrix to be decomposed.
 * @param <U> The type for the internal storage data structure of the matrix to be decomposed.
 *
 * @see Balancer
 * @see org.flag4j.linalg.decompositions.hess.RealHess
 * @see org.flag4j.linalg.decompositions.hess.ComplexHess
 * @see #getT()
 * @see #getU()
 * @see #setMaxIterationFactor(int)
 * @see #setExceptionalThreshold(int)
 */
public abstract class Schur<T extends MatrixMixin<T, ?, ?, ?>, U> extends Decomposition<T> {

    /**
     * Random number generator to be used when computing a random exceptional shift.
     */
    protected final RandomComplex rng;
    /**
     * Default number of iterations to apply before doing an exceptional shift.
     */
    protected final int DEFAULT_EXCEPTIONAL_ITERS = 20;
    /**
     * Default factor for computing the maximum number of iterations to perform.
     */
    protected final int DEFAULT_MAX_ITERS_FACTOR = 50;
    /**
     *For storing the (possibly block) upper triangular matrix <span class="latex-inline">T</span> in the Schur decomposition.
     */
    protected T T;
    /**
     *For storing the unitary <span class="latex-inline">U</span> matrix in the Schur decomposition.
     */
    protected T U;
    /**
     *Decomposer to compute the Hessenburg decomposition as a setup step for the implicit double step QR algorithm.
     */
    protected UnitaryDecomposition<T, U> hess;
    /**
     * <p>Balancer to apply a similarity transform to the matrix before the QR-algorithm is executed. This similarity transform
     * consists of permuting rows/columns to isolate decoupled eigenvalues then scaling the rows and columns of the matrix
     *
     * <p>This is done to attempt to improve the conditioning of the eigen-problem.
     */
    protected Balancer<T> balancer;
    /**
     * The lower bound (inclusive) of the row/column indices of the block to reduce to Schur form.
     */
    protected int iLow;
    /**
     * The upper bound (exclusive) of the row/column indices of the block to reduce to Schur form.
     */
    protected int iHigh;
    /**
     * Stores the number of rows in the matrix being decomposed (<em>after</em> balancing).
     */
    protected int numRows;
    /**
     * Stores the vector <span class="latex-inline">v</span> in the Householder reflector
     * <span class="latex-inline">P = I - &alpha;vv<sup>T</sup></span>.
     */
    protected U householderVector;
    /**
     * Stores the non-zero data of the first column in the shifted matrix
     * <span class="latex-inline">(A- &rho;<sub>1</sub>I)(A-&rho;<sub>2</sub> I)</span>
     * where <span class="latex-inline">&rho;<sub>1</sub></span> and <span class="latex-inline">&rho;<sub>2</sub></span>
     * are the two shifts.
     */
    protected U shiftCol;
    /**
     * An array for storing temporary values along the colum of a matrix when applying Householder reflectors.
     * This can help improve cache performance when applying the reflector.
     */
    protected U workArray;
    /**
     * Array for holding temporary values when computing the shifts.
     */
    protected U temp;
    /**
     * Factor for computing the maximum number of iterations to run the QR algorithm for.
     */
    protected int maxIterationsFactor;
    /**
     * Maximum number of iterations to run QR algorithm for.
     */
    protected int maxIterations;
    /**
     * Number of iterations to run without deflation before an exceptional shift is done.
     */
    protected int exceptionalThreshold;
    /**
     * The number of iterations runs in the QR algorithm without deflating or performing an exceptional shift.
     */
    protected int sinceLastExceptional;
    /**
     * The total number of exceptional shifts that have been used during the decomposition.
     */
    protected int numExceptional;
    /**
     * Flag indicating if a check should be made during the decomposition that the working matrix contains only finite values.
     * If true, an explicit check will be made and an exception will be thrown if {@link Double#isFinite(double) non-finite} values are
     * found. If false, no check will be made and the floating point arithmetic will carry on with {@link Double#POSITIVE_INFINITY
     * infinities},  {@link Double#NEGATIVE_INFINITY negative-infinities}, and {@link Double#NaN NaNs} present.
     */
    protected boolean checkFinite = false;
    /**
     * Flag indicating if the orthogonal matrix <span class="latex-inline">U</span> in the Schur decomposition should be computed.
     * <ul>
     *     <li>If {@code true}, <span class="latex-inline">U</span> will be computed.</li>
     *     <li>If {@code false}, <span class="latex-inline">U</span> will <em>not</em> be computed. This <em>may</em> improve performance if <span class="latex-inline">U</span>
     *     is not required.</li>
     * </ul>
     */
    protected final boolean computeU;


    /**
     * <p>Creates a decomposer to compute the Schur decomposition for a real dense matrix.
     *
     * <p>If the <span class="latex-inline">U</span> matrix is not needed, passing {@code computeU = false} may provide a performance improvement.
     *
     * @param computeU Flag indicating if the orthogonal matrix <span class="latex-inline">U</span> in the Schur decomposition should be computed.
     * <ul>
     *     <li>If {@code true}, <span class="latex-inline">U</span> will be computed.</li>
     *     <li>If {@code false}, <span class="latex-inline">U</span> will <em>not</em> be computed. This <em>may</em>
     *     improve performance if <span class="latex-inline">U</span>
     *     is not required.</li>
     * </ul>
     * @param rng Random number generator to use when performing random exceptional shifts.
     * @param hess Decomposer to compute the Hessenburg decomposition as a setup step for the QR algorithm.
     * @param balancer Balancer which balances the matrix as a preprocessing step to improve the conditioning of the eigenvalue
     * problem.
     */
    protected Schur(boolean computeU, RandomComplex rng, UnitaryDecomposition<T, U> hess, Balancer<T> balancer) {
        maxIterationsFactor = DEFAULT_MAX_ITERS_FACTOR;
        exceptionalThreshold = DEFAULT_EXCEPTIONAL_ITERS;
        this.rng = rng;
        this.computeU = computeU;
        this.hess = hess;
        this.balancer = balancer;
    }


    /**
     * <p>Sets the number of iterations in the QR algorithm to perform without deflation before performing a random shift.
     *
     * <p>That is, if {@code exceptionalThreshold = 10}, then at most 10 iterations QR algorithm iterations will be performed.
     * If, by the 10th iteration, no convergence has been detected which allows for deflation, then a QR algorithm iteration
     * will be performed with a random (i.e., exceptional) shift.
     *
     * <p>By default, the threshold is set to {@link #DEFAULT_EXCEPTIONAL_ITERS}
     *
     * @param exceptionalThreshold The new exceptional shift threshold. I.e., the number of iterations to perform without
     * deflation before performing an iteration with random shifts.
     *
     * @return A reference to this Schur decomposer.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code exceptionalThreshold} is not positive.
     */
    public Schur<T, U> setExceptionalThreshold(int exceptionalThreshold) {
        ValidateParameters.ensurePositive(exceptionalThreshold);
        this.exceptionalThreshold = exceptionalThreshold;
        return this;
    }


    /**
     * <p>Specify a maximum iteration factor for computing the total number of iterations to run the QR algorithm
     * for when computing the decomposition. The maximum number of iterations is computed as
     * {@code maxIteration = maxIterationFactor * src.numRows;} If the algorithm does not converge within this limit, an
     * exception will be thrown.
     *
     * <p>By default, this is computed as {@code maxIterations = DEFAULT_MAX_ITERS_FACTOR * src.numRows;}
     * where {@code src} is the matrix being decomposed (see {@link #DEFAULT_MAX_ITERS_FACTOR}).
     *
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * QR algorithm for.
     *
     * @return A reference to this Schur decomposer.
     *
     * @throws IllegalArgumentException If {@code maxIterationFactor} is not positive.
     */
    public Schur<T, U> setMaxIterationFactor(int maxIterationFactor) {
        ValidateParameters.ensurePositive(maxIterationFactor);
        this.maxIterationsFactor = maxIterationFactor;
        return this;
    }


    /**
     * <p>Sets a flag indicating if a check should be made to ensure the matrix being decomposed only contains finite values.
     * <p>By default, this will be {@code false}.
     * @param enforceFinite Flag indicating if a check should be made to ensure matrices decomposed by this instance only contain
     * finite values.
     * <ul>
     *     <li>If {@code true}, an explicit check will be made.</li>
     *     <li>If {@code false}, an explicit check will <em>not</em> be made.</li>
     * </ul>
     * @return A reference to this Schur decomposer.
     */
    public Schur<T, U> enforceFinite(boolean enforceFinite) {
        this.checkFinite = enforceFinite;
        return this;
    }


    /**
     * Gets the upper, or possibly block-upper, triangular Schur matrix <span class="latex-inline">T</span>
     * from the Schur decomposition
     * @return The <span class="latex-inline">T</span> matrix from the Schur decomposition
     * <span class="latex-inline">A=UTU<sup>H</sup></span>.
     */
    public T getT() {
        ensureHasDecomposed();
        return T;
    }


    /**
     * Gets the unitary matrix <span class="latex-inline">U</span> from the Schur decomposition containing
     * the Schur vectors as its columns.
     * @return The <span class="latex-inline">U</span> matrix from the Schur decomposition
     * <span class="latex-inline">A=UTU<sup>H</sup></span>.
     */
    public T getU() {
        ensureHasDecomposed();
        return U;
    }


    /**
     * <p>Computes the Schur decomposition of the input matrix.
     *
     * @implNote The Schur decomposition is computed using Francis implicit double-shifted QR algorithm.
     * There are known cases where this variant of the QR algorithm <em>may</em> fail to converge. Random shifting is used when the
     * matrix is not converging, which greatly minimizes this issue. It is unlikely that a general matrix will fail to converge with
     * these random shifts; however, no guarantees of convergence can be made.
     * @param src The source matrix to decompose.
     * @throws LinearAlgebraException If the decomposition does not converge within the specified number of max iterations.
     */
    protected void decomposeBase(T src) {
        setUp(src);

        int workingSize = iHigh - iLow - 1;
        int workEnd = iHigh - 1;  // Equivalent to iLow + workingSize.
        int iters = 0;

        // Each iteration of this loop is a complete implicit QR algorithm iteration.
        while(workingSize >= 2 && iters < maxIterations) {
            if(sinceLastExceptional >= exceptionalThreshold) {
                // Perform an exceptional shift iteration.
                sinceLastExceptional = 0; // Reset number of iterations completed.
                numExceptional++;
                performExceptionalShift(workEnd);
            } else {
                // Perform a normal double shift iteration.
                sinceLastExceptional++; // Increase the number of iterations performed without an exceptional shift.
                performDoubleShift(workEnd);
            }

            // Check for convergence and deflate as needed.
            int deflate = checkConvergence(workEnd);
            if(deflate > 0) {
                sinceLastExceptional = 0; // Reset the number of iterations since the last exceptional shift.
                workingSize -= deflate; // Reduce working size.
                workEnd -= deflate;
            }

            iters++;
        }

        if(iters == maxIterations) {
            throw new LinearAlgebraException("Schur decomposition failed to converge in " + maxIterations + " iterations. " +
                    "Increasing maxIterationsFactor may allow for the decomposition to converge.");
        }

        // Undo any transformations applied during balancing and reconstitute full matrix.
        unbalance();
        this.hasDecomposed = true;
    }


    /**
     * Performs basic setup and initializes data structures to be used in the decomposition.
     * @param src The matrix to be decomposed.
     */
    protected void setUp(T src) {
        ValidateParameters.ensureSquare(src.getShape());

        sinceLastExceptional = 0;
        numExceptional = 0;
        numRows = src.numRows();
        maxIterations = numRows*maxIterationsFactor;

        setUpArrays();

        // Balance matrix.
        balancer.decompose(src);
        iLow = balancer.getILow();
        iHigh = balancer.getIHigh();
        T = balancer.getB();

        // Reduce to upper Hessenburg form.
        hess.decompose(T, iLow, iHigh);
        T = hess.getUpper();
        // Initialize U as the product of transformations used in Hessenburg decomposition if requested.
        U = computeU ? hess.getQ() : null; // Hessenburg decomposition computes U lazily only when getQ() is called.
    }


    /**
     * <p>Reverts the scaling and permutations applied during the balancing step to get the correct form.
     *
     * <p>Specifically, this method computes
     * <span class="latex-eq-align"><pre>
     *     U := PDU
     *        = TU</pre></span>
     * where <span class="latex-inline">P</span> and <span class="latex-inline">D</span> are the permutation and
     * scaling matrices respectively from balancing.
     */
    protected abstract void unbalance();


    /**
     * Initializes temporary work arrays to be used in the decomposition.
     */
    protected abstract void setUpArrays();


    /**
     * Performs a full iteration of the single-shifted QR algorithm (this includes the bulge chase) where the shift is
     * chosen to be a random value with the same magnitude as the lower right element of the working matrix. This can help the
     * QR converge for certain pathological cases where the double shift algorithm oscillates or fails to converge for
     * repeated eigenvalues.
     * @param workEnd The ending row (inclusive) of the current active working block.
     */
    protected abstract void performExceptionalShift(int workEnd);


    /**
     * Performs a full iteration of the Francis implicit double-shifted QR algorithm (this includes the bulge chase).
     * @param workEnd The ending row (inclusive) of the current active working block.
     */
    protected abstract void performDoubleShift(int workEnd);


    /**
     * Checks for convergence of lower <span class="latex-inline">2&times;2</span> submatrix within working matrix to an upper
     * triangular or block upper triangular form.
     * If convergence is found, this will also zero out the values which have converged to near zero.
     * @param workEnd The ending row (inclusive) of the current active working block.
     * @return Returns the amount the working matrix size should be deflated. Will be zero if no convergence is detected, one if
     * convergence to an upper triangular form is detected and two if convergence to block upper triangular form is detected.
     */
    protected abstract int checkConvergence(int workEnd);


    /**
     * Ensures that {@code src} only contains finite values.
     * @param src Matrix of interest.
     * @throws IllegalArgumentException If {@code src} does <em>not</em> contain only finite values.
     */
    protected abstract void checkFinite(T src);
}
