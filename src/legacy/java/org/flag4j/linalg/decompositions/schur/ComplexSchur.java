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

package org.flag4j.linalg.decompositions.schur;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.Eigen;
import org.flag4j.linalg.decompositions.balance.ComplexBalancer;
import org.flag4j.linalg.decompositions.hess.ComplexHess;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.linalg.transformations.Givens;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.numbers.Complex128;
import org.flag4j.rng.RandomComplex;
import org.flag4j.rng.RandomState;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * <p>Instanced of this class can be used for computing the Schur decomposition of a real dense square matrix.
 *
 * <p>The Schur decomposition decomposes a given square matrix <span class="latex-inline">A</span> into:
 * <span class="latex-display"><pre>
 *     A = UTU<sup>T</sup></pre></span>
 * where <span class="latex-inline">U</span> is an orthogonal matrix <span class="latex-inline">T</span> is a
 * quasi-upper triangular matrix known as the <em>Schur form</em> of <span class="latex-inline">A</span>.
 * This means <span class="latex-inline">T</span> is upper triangular except
 * for possibly <span class="latex-inline">2&times;2</span> blocks along its diagonal, which correspond to complex conjugate pairs
 * of eigenvalues.
 *
 * <p>The Schur decomposition proceeds by an iterative algorithm with possible random behavior. For reproducibility, constructors
 * support specifying a seed for the pseudo-random number generator.
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code ComplexSchur}.</li>
 *     <li>Call {@link #decompose(CMatrix)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getU()} and {@link #getT()}.</li>
 * </ol>
 *
 * <h2>Efficiency Considerations:</h2>
 * If eigenvectors are not required, setting {@code computeU = false} <em>may</em> improve performance.
 *
 * <p>This class was inspired by code from the <a href="http://ejml.org/wiki/index.php?title=Main_Page">EJML</a>
 * library and the description of the Francis implicit double shifted QR algorithm from
 * <a href="https://www.math.wsu.edu/faculty/watkins/books.html">Fundamentals of Matrix
 * Computations 3rd Edition by David S. Watkins</a>.
 *
 * @implNote This decomposition is performed using the <b>implicit double-shift QR algorithm</b>, which iteratively
 * reduces the matrix to Schur form using orthogonal transformations. In addition to this, random shifting is used in cases where
 * normal convergence fails.
 *
 * <p>As a preprocessing step to improve conditioning and stability, the matrix is first {@link ComplexBalancer balanced} then
 * reduced to {@link ComplexHess Hessenberg form}.
 *
 * @param <T> The type of matrix to be decomposed.
 * @param <U> The type for the internal storage items structure of the matrix to be decomposed.
 *
 * @see ComplexBalancer
 * @see org.flag4j.linalg.decompositions.hess.ComplexHess
 * @see #getT()
 * @see #getU()
 * @see #setMaxIterationFactor(int)
 * @see #setExceptionalThreshold(int)
 */
public class ComplexSchur extends Schur<CMatrix, Complex128[]> {

    /**
     * The complex number equal to zero.
     */
    private final static Complex128 ZERO = Complex128.ZERO;

    /**
     * For computing the norm of a column for use when computing Householder reflectors.
     */
    protected Complex128 norm;
    /**
     * Stores the scalar factor <span class="latex-inline">&alpha;</span> for use in computation of the Householder reflector
     * <span class="latex-inline">P = I - &alpha;vv<sup>T</sup></span>.
     */
    protected Complex128 currentFactor;


    /**
     * <p>Creates a decomposer to compute the Schur decomposition for a complex dense matrix.
     *
     * <p>Note: This decomposer <em>may</em> use random numbers during the decomposition. If reproducible results are needed,
     * set the seed for the pseudo-random number generator using {@link #ComplexSchur(long)}
     */
    public ComplexSchur() {
        super(true, RandomState.getDefaultRng(), new ComplexHess(true, true), new ComplexBalancer());
    }


    /**
     * <p>Creates a decomposer to compute the Schur decomposition for a real dense matrix where the <span class="latex-inline">U</span> matrix may or may not
     * be computed.
     *
     * <p>If the <span class="latex-inline">U</span> matrix is not needed, passing {@code computeU = false} may provide a performance improvement.
     *
     * <p>By default, if a constructor with no {@code computeU} parameter is called, <span class="latex-inline">U</span> <em>will</em> be computed.
     *
     * <p>Note: This decomposer <em>may</em> use random numbers during the decomposition. If reproducible results are desired,
     * set the seed for the pseudo-random number generator using {@link #ComplexSchur(boolean, long)}
     *
     * @param computeU Flag indicating if the unitary U matrix should be computed for the Schur decomposition. If true,
     * U will be computed. If false, <span class="latex-inline">U</span> will not be computed.
     */
    public ComplexSchur(boolean computeU) {
        super(computeU, RandomState.getDefaultRng(), new ComplexHess(computeU, true), new ComplexBalancer());
    }


    /**
     * Creates a decomposer to compute the Schur decomposition for a real dense matrix.
     * @param seed Seed to use for pseudo-random number generator when computing exceptional shifts during the QR algorithm.
     */
    public ComplexSchur(long seed) {
        super(true, new RandomComplex(seed), new ComplexHess(true, true), new ComplexBalancer());
    }


    /**
     * Creates a decomposer to compute the Schur decomposition for a real dense matrix.
     * @param seed Seed to use for pseudo-random number generator when computing exceptional shifts during the QR algorithm.
     */
    public ComplexSchur(boolean computeU, long seed) {
        super(computeU, new RandomComplex(seed), new ComplexHess(computeU, true), new ComplexBalancer());
    }


    @Override
    public ComplexSchur setExceptionalThreshold(int exceptionalThreshold) {
        // Provided so calls like the following can be made:
        //    RealSchur schur = new RealSchur().setExceptionalThreshold(x).decompose(src)
        return (ComplexSchur) super.setExceptionalThreshold(exceptionalThreshold);
    }


    @Override
    public ComplexSchur setMaxIterationFactor(int maxIterationFactor) {
        // Provided so calls like the following can be made:
        //    RealSchur schur = new RealSchur().setMaxIterationFactor(x).decompose()
        return (ComplexSchur) super.setMaxIterationFactor(maxIterationFactor);
    }

    @Override
    public ComplexSchur enforceFinite(boolean enforceFinite) {
        // Provided so that calls like the following can be made:
        //    RealSchur schur = new RealSchur().enforceFinite(true).decompose()
        super.enforceFinite(enforceFinite);
        return this;
    }


    /**
     * <p>Reverts the scaling and permutations applied during the balancing step to obtain the correct form.
     * <p>Specifically, this method computes
     * <span class="latex-eq-align"><pre>
     *     U := PDU
     *        = TU</pre>
     * where <span class="latex-inline">P</span> and <span class="latex-inline">D</span> are the permutation and scaling matrices respectively from balancing.
     */
    @Override
    protected void unbalance() {
        // Unbalancing can be skipped entirely if the Schur basis Q is not being computed.
        if (computeU) U = balancer.applyLeftTransform(U);
    }


    /**
     * <p>Computes the Schur decomposition of the input matrix.
     *
     * @implNote The Schur decomposition is computed using the Francis implicit double shifted QR algorithm.
     * There are known cases where this variant of the QR algorithm fail to converge. Random shifting is employed when the
     * matrix is not converging which greatly minimizes this issue. It is unlikely that a general matrix will fail to converge with
     * these random shifts however, no guarantees of convergence can be made.
     * @param src The source matrix to decompose.
     */
    @Override
    public ComplexSchur decompose(CMatrix src) {
        decomposeBase(src);
        return this;
    }


    /**
     * Initializes temporary work ndarrays to be used in the decomposition.
     */
    @Override
    protected void setUpArrays() {
        householderVector = new Complex128[numRows];
        workArray = new Complex128[numRows]; // TODO: If givens are used in the single step, the work array will need length 2*numRows.
        shiftCol = new Complex128[3]; // For storing non-zero items (the first 2 or 3) of the first column of the double/single shift.
        temp = new Complex128[9]; // For storing temporary values when computing shifts.
    }


    /**
     * Performs a full iteration of the single shifted QR algorithm (this includes the bulge chase) where the shift is
     * chosen to be a random value with the same magnitude as the lower right element of the working matrix. This can help the
     * QR converge for certain pathological cases where the double shift algorithm oscillates or fails to converge for
     * repeated eigenvalues.
     *
     * @param workEnd The ending row (inclusive) of the current active working block.
     */
    @Override
    protected void performExceptionalShift(int workEnd) {
        performSingleShift(workEnd, computeExceptionalShift(workEnd));
    }


    /**
     * Computes a random shift to help the QR algorithm converge if it gets stuck.
     * @param k The current size of the working matrix. Specifically, the index of the lower right value in the working matrix is
     *          {@code (k, k)}.
     * @return A shift in a random direction which has the same magnitude as the elements in the matrix.
     */
    protected Complex128 computeExceptionalShift(int k) {
        double mag = T.data[k*numRows + k].mag();
        mag = (mag == 0.0) ? 1.0 : Math.abs(mag); // Ensure shift is not zero.

        double p = 1.0 - Math.pow(0.1, numExceptional);
        mag *= p + 2.0*(1.0 - p)*(rng.nextDouble() - 0.5);

        return rng.randomComplex128(mag);  // Choose complex number with specified magnitude in a random direction.
    }


    /**
     * Computes the non-zero items of the first column for the single shifted QR algorithm.
     * @param k Size of current working matrix.
     * @param shift The shift to use.
     */
    protected void computeImplicitSingleShift(int k, Complex128 shift) {
        int leftIdx = k-1;
        shiftCol[0] = T.data[leftIdx*numRows + leftIdx].sub(shift);
        shiftCol[1] = T.data[k*numRows + leftIdx];
    }


    /**
     * Performs a full iteration of the implicit single shifted QR algorithm (this includes the bulge chase).
     * @param workEnd The ending row (inclusive) of the current active working block.
     * @param shift The shift to use in the implicit single shifted QR algorithm.
     */
    protected void performSingleShift(int workEnd, Complex128 shift) {
        // Compute the non-zero items of first column for shifted matrix.
        computeImplicitSingleShift(workEnd, shift);

        // Extract non-zero values from first column in shifted matrix.
        Complex128 p1 = shiftCol[0];
        Complex128 p2 = shiftCol[1];

        for(int i = iLow; i <= workEnd - 1; i++) {
            if(makeReflector(i, p1, p2)) // Construct reflector.
                applySingleShiftReflector(i, i > iLow); // Apply the reflector if needed.

            // Set values to be used in computing the next bulge chasing reflector.
            p1 = T.data[(i + 1)*numRows + i];
            if(i < workEnd-1) p2 = T.data[(i + 2)*numRows + i];
        }
    }


    /**
     * Applies reflector for the double shift. This method can be used to apply either be the reflector constructed for the first
     * column of the shifted matrix, or a reflector being used in the bulge chase of size 2 which arises from the first case.
     * @param i The starting row the reflector is being applied to.
     */
    protected void applySingleShiftReflector(int i, boolean set) {
        applyReflector(i, 1);  // Apply reflector for shift size of 1.

        // TODO: Since we only need to zero a single element, consider using a givens rotator here instead.
        //  There should be negligible stability difference between the two for a 2x2 rotator, but the givens rotator is more
        //  simple to calculate. However, this seems to be incorrect. Figure out what this should be or if the left/right
        //  multiplication methods are incorrect.
//        Matrix G = Givens.get2x2Rotator(T.items[i*numRows + i - 1], T.items[(i+1)*numRows + i - 1]);
//        Givens.leftMult2x2Rotator(T, G, i+1, givensWorkArray);
//        Givens.rightMult2x2Rotator(T, G, i+1, givensWorkArray);

        if(set) {
            // Explicitly zeros out values which should be zeroed by the reflector.
            T.data[i*numRows + i - 1] = norm.negate();
            T.data[(i+1)*numRows + i - 1] = Complex128.ZERO;
        }
    }


    /**
     * Performs a full iteration of the Francis implicit double shifted QR algorithm (this includes the bulge chase).
     * @param workEnd The ending row (inclusive) of the current active working block.
     */
    protected void performDoubleShift(int workEnd) {
        // Compute the non-zero items (first three) of the first column of the double shifted matrix.
        computeImplicitDoubleShift(workEnd);

        // Extract non-zero values in first column of the double shifted matrix.
        Complex128 p1 = shiftCol[0];
        Complex128 p2 = shiftCol[1];
        Complex128 p3 = shiftCol[2];

        // Apply shift and chase bulge.
        for(int i = iLow; i <= workEnd - 2; i++) {
            if(makeReflector(i, p1, p2, p3)) // Construct Householder reflector.
                applyDoubleShiftReflector(i, i > iLow); // Apply the reflector if needed.

            // Set values to be used in computing the next bulge chasing reflector.
            p1 = T.data[(i + 1)*numRows + i];
            p2 = T.data[(i + 2)*numRows + i];
            if(i < workEnd - 2) p3 = T.data[(i + 3)*numRows + i];
        }

        // The last reflector in the bulge chase only acts on last two rows of the working matrix.
        if(makeReflector(workEnd - 1, p1, p2)) // Construct Householder reflector.
            applySingleShiftReflector(workEnd - 1, true); // Apply the reflector if needed.
    }


    /**
     * Computes the shifts for a Francis double shift iteration. Specifically, the shifts are the generalized Rayleigh quotients of
     * degree two.
     * @param workEnd The ending row (inclusive) of the current active working block.
     */
    protected void computeImplicitDoubleShift(int workEnd) {
        // The shift computed here, p, represent the double shift
        //  p = (T - rho1*I)(T - rho2*I)*e1 where I is the identity matrix, e1 is the first column of I, and (rho1, rho2)
        //  are taken to be the eigenvalues of the lower 2x2 submatrix within the working matrix.


        // Extract values from lower right 2x2 submatrix within the working size.
        int topIdx = (workEnd - 1)*numRows + workEnd;
        int bottomIdx = workEnd*numRows + workEnd;
        Complex128 x11 = T.data[topIdx - 1];
        Complex128 x12 = T.data[topIdx];
        Complex128 x21 = T.data[bottomIdx - 1];
        Complex128 x22 = T.data[bottomIdx];

        // Extract top right items of T for use in computing the shift p.
        topIdx = iLow*numRows + iLow;
        bottomIdx = (iLow + 1)*numRows + iLow;
        Complex128 a11 = T.data[topIdx];
        Complex128 a12 = T.data[topIdx + 1];
        Complex128 a21 = T.data[bottomIdx];
        Complex128 a22 = T.data[bottomIdx + 1];
        Complex128 a32 = T.data[(iLow + 2)*numRows + iLow + 1];

        // Scale values to improve stability and help avoid possible over(under)flow issues.
        temp[0] = a11; temp[1] = a21; temp[2] = a12; temp[3] = a22; temp[4] = a32;
        temp[5] = x11; temp[6] = x22; temp[7] = x12; temp[8] = x21;
        double maxAbsInv = 1.0 / CompareRing.maxAbs(temp);

        a11 = a11.mult(maxAbsInv); a12 = a12.mult(maxAbsInv); a21 = a21.mult(maxAbsInv);
        a22 = a22.mult(maxAbsInv); a32 = a32.mult(maxAbsInv);
        x11 = x11.mult(maxAbsInv); x12 = x12.mult(maxAbsInv); x21 = x21.mult(maxAbsInv); x22 = x22.mult(maxAbsInv);

        // Compute shifts to be eigenvalues of trailing 2x2 submatrix.
        Complex128[] rho = Eigen.get2x2EigenValues(x11, x12, x21, x22);

        // Compute first three non-zero items of the shift p.
        shiftCol[0] = a11.sub(rho[0]).mult( a11.sub(rho[1]) ).add( a12.mult(a21) );
        shiftCol[1] = a21.mult( a11.add(a21).sub( rho[0].add(rho[1])) );
        shiftCol[2] = a32.mult(a21);
    }


    /**
     * Applies reflector for the double shift. This method can be used to apply either be the reflector constructed for the first
     * column of the shifted matrix, or a reflector being used in the bulge chase of size 2 which arises from the first case.
     * @param i The starting row the reflector is being applied to.
     */
    protected void applyDoubleShiftReflector(int i, boolean set) {
        applyReflector(i, 2); // Apply reflector for shift size of 2.

        if(set) {
            // Explicitly zeros out values which should be zeroed by the reflector.
            T.data[i*numRows + i - 1] = norm.negate();
            T.data[(i+1)*numRows + i - 1] = Complex128.ZERO;
            T.data[(i+2)*numRows + i - 1] = Complex128.ZERO;
        }
    }


    /**
     * Applies the constructed Householder reflector which has been constructed for the given shift size.
     * @param i The stating row the reflector is being applied to.
     * @param shiftSize The size of the shift the reflector was constructed for.
     */
    protected void applyReflector(int i, int shiftSize) {
        int endRow = i + shiftSize + 1;

        // Apply reflector to left (Assumes T is upper hessenburg except for possibly a bulge of size shiftSize).
        Householder.leftMultReflector(T, householderVector, currentFactor, i, i, endRow, workArray);
        // Apply reflector to right (Assumes T is upper hessenburg except for possibly a bulge of size shiftSize).
        Householder.rightMultReflector(T, householderVector, currentFactor, iLow, i, endRow);

        if(computeU) {
            // Accumulate the reflector in U if it is being computed.
            Householder.rightMultReflector(U, householderVector, currentFactor, iLow, i, endRow);
        }
    }


    /**
     * Constructs a householder reflector given specified values for a column to apply the reflector to. This reflector is stored in
     * indices {@code i}, {@code i+1}, and {@code i+2} of {@link #householderVector}.
     * @param i Row of working matrix to construct reflector for.
     * @param p1 First entry to in column to apply reflector to.
     * @param p2 Second entry in column to apply reflector to.
     * @param p3 Third entry in column to apply reflector to.
     * @return True if a reflector needs to be constructed to return matrix to upper Hessenburg form. False if column is
     * already in the correct form.
     */
    protected boolean makeReflector(int i, Complex128 p1, Complex128 p2, Complex128 p3) {
        // Scale components for stability and overflow purposes.
        double maxAbs = Math.max(p1.mag(), Math.max(p2.mag(), p3.mag()));

        if(maxAbs <= EPS_F64*T.data[i*numRows + i].mag())
            return false; // No reflector needs to be constructed or applied.

        double maxAbsInv = 1.0/maxAbs;  // Reciprocal to save a couple division operations.
        p1 = p1.mult(maxAbsInv);
        p2 = p2.mult(maxAbsInv);
        p3 = p3.mult(maxAbsInv);

        double m1 = p1.mag();
        double m2 = p2.mag();
        double m3 = p3.mag();

        double normRe = Math.sqrt(m1*m1 + m2*m2 + m3*m3); // Compute scaled 2-norm.

        // Change phase of the norm depending on first entry in column for stability purposes in Householder vector.
        norm = p1.equals(ZERO) ? new Complex128(normRe) : Complex128.sgn(p1).mult(normRe);

        Complex128 div = p1.add(norm);
        currentFactor = div.div(norm);
        norm = norm.mult(maxAbs); // Rescale norm to be proper magnitude.

        householderVector[i] = Complex128.ONE;
        householderVector[i + 1] = p2.div(div);
        householderVector[i + 2] = p3.div(div);

        return true;
    }


    /**
     * Constructs a householder reflector given specified values for a column to apply the reflector to. This reflector is stored in
     * indices {@code i} and {@code i+1} of {@link #householderVector}.
     * @param i Row of working matrix to construct reflector for.
     * @param p1 First entry to in column to apply reflector to.
     * @param p2 Second entry in column to apply reflector to.
     * @return True if a reflector needs to be constructed to return matrix to upper Hessenburg form. False if column is
     * already in the correct form.
     */
    protected boolean makeReflector(int i, Complex128 p1, Complex128 p2) {
        double maxAbs = Math.max(p1.mag(), p2.mag());
        if(maxAbs <= EPS_F64*T.data[i*numRows + i].mag())
            return false; // No reflector needs to be constructed or applied.

        // Scale components for stability and over(under)flow purposes.
        p1 = p1.div(maxAbs);
        p2 = p2.div(maxAbs);

        double m1 = p1.mag();
        double m2 = p2.mag();

        double normRe = Math.sqrt(m1*m1 + m2*m2); // Compute scaled norm.

        // Change phase of the norm depending on first entry in column for stability purposes in Householder vector.
        norm = p1.equals(ZERO) ? new Complex128(normRe) : Complex128.sgn(p1).mult(normRe);

        Complex128 divisor = p1.add(norm);
        currentFactor = divisor.div(norm);
        norm = norm.mult(maxAbs); // Rescale norm to be proper magnitude.

        householderVector[i] = Complex128.ONE; // Ensure first value of reflector is 1.
        householderVector[i + 1] = p2.div(divisor);

        return true; // Reflector has been constructed and must be applied.
    }


    /**
     * Checks for convergence of lower <span class="latex-inline">2&times;2</span> submatrix within working matrix to upper
     * triangular or block upper triangular form. If
     * convergence is found, this will also zero out the values which have converged to near zero.
     * @param workEnd The ending row (inclusive) of the current active working block.
     * @return Returns the amount the working matrix size should be deflated. Will be zero if no convergence is detected, one if
     * convergence to upper triangular form is detected and two if convergence to block upper triangular form is detected.
     */
    protected int checkConvergence(int workEnd) {
        int leftRow = (workEnd-1)*numRows + workEnd;
        int rightRow = workEnd*numRows + workEnd;

        Complex128 a11 = T.data[(workEnd-2)*numRows + workEnd - 2];
        Complex128 a21 = T.data[leftRow - 2];
        Complex128 a22 = T.data[leftRow - 1];
        Complex128 a23 = T.data[leftRow];
        Complex128 a32 = T.data[rightRow - 1];
        Complex128 a33 = T.data[rightRow];

        // Uses deflation criteria proposed by Wilkinson: |A[k, k-1]| < eps*(|A[k, k]| + |A[k-1, k-1]|)
        // AND the deflation criteria proposed by Ahues and Tisseur:
        //     |A[k, k-1]| *|A[k-1, k]| <= eps * |A[k, k]| * |A[k, k] - A[k-1, k-1]|
        if(a32.mag() < EPS_F64*(a33.mag() + a22.mag())
                && a32.mag()*a23.mag() <= EPS_F64*a33.mag() * (a33.sub(a22)).mag()) {
            T.data[rightRow - 1] = Complex128.ZERO; // Zero out converged value.
            return 1; // Deflate by 1.
        } else if(a21.mag() < EPS_F64*(a11.mag() + a22.mag())) {
            T.data[leftRow - 2] = Complex128.ZERO; // Zero out converged value.
            return 2; // Deflate by 2.
        }

        return 0; // No convergence detected. Do not deflate.
    }


    /**
     * Ensures that {@code src} only contains finite values.
     *
     * @param src Matrix of interest.
     *
     * @throws IllegalArgumentException If {@code src} does <em>not</em> contain only finite values.
     */
    @Override
    protected void checkFinite(CMatrix src) {
        if(!src.isAllFinite())
            throw new IllegalArgumentException("Matrix is not finite.");
    }


    public CMatrix[] real2ComplexSchur() {
        // Convert matrices to complex matrices.
        CMatrix tComplex = T.copy();
        CMatrix uComplex = computeU ? U.copy() : null;
        Complex128[] givensWorkComplex = new Complex128[2*numRows];

        for(int m=numRows-1; m>0; m--) {
            Complex128 a11 = tComplex.data[(m - 1)*numRows + m - 1];
            Complex128 a12 = tComplex.data[(m - 1)*numRows + m];
            Complex128 a21 = tComplex.data[m*numRows + m - 1];
            Complex128 a22 = tComplex.data[m*numRows + m];

            if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                // non-converged 2x2 block found.
                Complex128[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                mu[0] = mu[0].sub(a22); // Shift eigenvalue.

                // Construct a givens rotator to bring matrix into properly upper triangular form.
                CMatrix G = Givens.get2x2Rotator(new CVector(mu[0], a21));
                // Apply rotation to T matrix to bring it into upper triangular form.
                Givens.leftMult2x2Rotator(tComplex, G, m, givensWorkComplex);
                // Apply Hermitian transpose to keep transformation similar.
                Givens.rightMult2x2Rotator(tComplex, G, m, givensWorkComplex);

                if(uComplex != null) {
                    // Accumulate similarity transforms in the U matrix.
                    Givens.rightMult2x2Rotator(uComplex, G, m, givensWorkComplex);
                }

                tComplex.set(Complex128.ZERO, m, m-1);
            }
        }

        return new CMatrix[]{tComplex, uComplex};
    }
}
