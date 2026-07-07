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

package org.flag4j.linalg;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.schur.ComplexSchur;
import org.flag4j.linalg.decompositions.schur.RealSchur;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * This class provides several methods useful for computing eigenvalues and eigenvectors.
 */
public final class Eigen {


    private Eigen() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the eigenvalues of a 2&times;2 matrix explicitly.
     * @param src The 2&times;2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2&times;2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(Matrix src) {
        ValidateParameters.ensureAllEqual(2, src.numRows, src.numCols);
        return new CVector(get2x2EigenValues(src.data[0], src.data[1], src.data[2], src.data[3]));
    }


    /**
     * Computes the eigenvalues of a 2&times;2 matrix explicitly.
     * @param a11 First entry in matrix (at index (0, 0)).
     * @param a12 Second entry in matrix (at index (0, 1)).
     * @param a21 Third entry in matrix (at index (1, 0)).
     * @param a22 Fourth entry in matrix (at index (1, 1)).
     * @return A complex array containing the eigenvalues of the 2&times;2 {@code src} matrix.
     */
    public static Complex128[] get2x2EigenValues(double a11, double a12, double a21, double a22) {
        // This method computes eigenvalues in a stable way which is more resilient to overflow errors than
        // standard methods.
        Complex128[] lambda = new Complex128[2];
        double maxAbs = RealProperties.maxAbs(a11, a12, a21, a22);

        if(maxAbs == 0) {
            Arrays.fill(lambda, Complex128.ZERO);
            return lambda;
        } else {
            a11 /= maxAbs;
            a12 /= maxAbs;
            a21 /= maxAbs;
            a22 /= maxAbs;

            double c;
            double s;

            if (a12 + a21 == 0) {
                c = s = 1.0 / Math.sqrt(2);
            } else {
                double aa = (a11 - a22);
                double bb = (a12 + a21);

                double t_hat = aa/bb;
                double t = t_hat/(1.0 + Math.sqrt(1.0 + t_hat*t_hat));

                c = 1.0 / Math.sqrt(1.0 + t*t);
                s = c*t;
            }

            double c2 = c*c;
            double s2 = s*s;
            double cs = c*s;

            double b11 = c2*a11 + s2*a22 - cs*(a12 + a21);
            double b12 = c2*a12 - s2*a21 + cs*(a11 - a22);
            double b21 = c2*a21 - s2*a12 + cs*(a11 - a22);

            // apply second rotator to make A upper triangular if real eigenvalues
            if (b21*b12 >= 0) {
                if (b12 == 0) {
                    c = 0;
                    s = 1;
                } else {
                    s = Math.sqrt(b21/(b12 + b21));
                    c = Math.sqrt(b12/(b12 + b21));
                }

                cs = c*s;

                a11 = b11 - cs*(b12 + b21);
                a22 = b11 + cs*(b12 + b21);

                lambda[0] = new Complex128(a11*maxAbs);
                lambda[1] = new Complex128(a22*maxAbs);
            } else {
                double im = Math.sqrt(-b21*b12);

                lambda[0] = new Complex128(b11*maxAbs, im*maxAbs);
                lambda[1] = new Complex128(b11*maxAbs, -im*maxAbs);
            }
        }

        return lambda;
    }


    /**
     * Computes the eigenvalues of a 2&times;2 matrix explicitly.
     * @param src The 2&times;2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2&times;2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(CMatrix src) {
        ValidateParameters.ensureAllEqual(2, src.numRows, src.numCols);
        return new CVector(get2x2EigenValues(src.data[0], src.data[1], src.data[2], src.data[3]));
    }


    /**
     * Computes the eigenvalues of a 2&times;2 complex matrix (assumed to be row major).
     * @param a11 The first entry in the 2&times;2 complex matrix
     * @param a12 The second entry in the 2&times;2 complex matrix
     * @param a21 The third entry in the 2&times;2 complex matrix
     * @param a22 The fourth entry in the 2&times;2 complex matrix
     * @return An array containing the eigenvalues of the specified 2&times;2 complex matrix. Eigenvalues will be repeated per their
     * multiplicity.
     */
    public static Complex128[] get2x2EigenValues(Complex128 a11, Complex128 a12,
                                                 Complex128 a21, Complex128 a22) {
        Complex128[] lambda = new Complex128[2];

        // Compute maximum magnitude for scaling.
        double maxAbs = Math.max(Math.max(a11.mag(), a12.mag()), Math.max(a21.mag(), a22.mag()));

        if(maxAbs == 0) {
            lambda[0] = Complex128.ZERO;
            lambda[1] = Complex128.ZERO;
            return lambda;
        }

        // Scale the matrix to avoid over/underflow.
        a11 = a11.div(maxAbs);
        a12 = a12.div(maxAbs);
        a21 = a21.div(maxAbs);
        a22 = a22.div(maxAbs);

        // Trace and determinant for the 2x2 matrix.
        Complex128 trace = a11.add(a22);
        Complex128 det = a11.mult(a22).sub(a12.mult(a21));

        // Compute the middle term of the quadratic equation.
        Complex128 middleTerm = trace.mult(trace).div(4).sub(det);

        // Compute the square root of the middle term.
        Complex128 sqrtMiddle = middleTerm.sqrt();

        // Compute eigenvalues.
        lambda[0] = trace.div(2).add(sqrtMiddle);
        lambda[1] = trace.div(2).sub(sqrtMiddle);

        // Scale back the eigenvalues.
        lambda[0] = lambda[0].mult(new Complex128(maxAbs));
        lambda[1] = lambda[1].mult(new Complex128(maxAbs));

        return lambda;
    }


    /**
     * Computes the eigenvalues of a square real dense matrix. For reproducibility see {@link #getEigenValues(Matrix, long)}. If the
     * algorithm fails to converge within the default maximum number of iterations, {@link #getEigenValues(Matrix, long, int)} can
     * be used to specify a larger number of iterations to attempt.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(Matrix, long)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CVector getEigenValues(Matrix src) {
        RealSchur schur = new RealSchur(false);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a square real dense matrix. If the algorithm fails to converge within the default maximum number
     * of iterations, {@link #getEigenValues(Matrix, long, int)} can be used to specify a larger number of iterations to attempt.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CVector getEigenValues(Matrix src, long seed) {
        RealSchur schur = new RealSchur(false, seed);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a square real dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     */
    public static CVector getEigenValues(Matrix src, long seed, int maxIterationFactor) {
        RealSchur schur = new RealSchur(false, seed).setMaxIterationFactor(maxIterationFactor);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a real dense square matrix.
     * @param src Matrix to compute eigenvalues of.
     * @param schur Schur decomposer to use in the eigenvalue computation.
     * @return The eigenvalues of the {@code src} matrix stored in a complex vector ({@link CVector}).
     */
    private static CVector getEigenValues(Matrix src, RealSchur schur) {
        CVector lambdas = new CVector(src.numRows);
        Matrix T = schur.decompose(src).getT();

        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.data[m] = new Complex128(T.data[m*numRows + m]);
            } else {
                double a11 = T.data[m*numRows + m];
                double a12 = T.data[m*numRows + m + 1];
                double a21 = T.data[(m+1)*numRows + m];
                double a22 = T.data[(m+1)*numRows + m + 1];

                if(Math.abs(a21) > EPS_F64*(Math.abs(a11) + Math.abs(a22))) {
                    // Non-converged 2x2 block found.
                    Complex128[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                    lambdas.data[m] = mu[0];
                    lambdas.data[++m] = mu[1];
                } else {
                    lambdas.data[m] = new Complex128(a11);
                }
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix. For reproducibility see {@link #getEigenValues(CMatrix, long)}.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(CMatrix, long)
     */
    public static CVector getEigenValues(CMatrix src) {
        return getEigenValues(src, new ComplexSchur(false));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues. This allows for reproducibility despite randomness in the
     * eigenvalue computation algorithm.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(CMatrix)
     */
    public static CVector getEigenValues(CMatrix src, long seed) {
        return getEigenValues(src, new ComplexSchur(false, seed));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     */
    public static CVector getEigenValues(CMatrix src, long seed, int maxIterationFactor) {
        return getEigenValues(src,
                new ComplexSchur(false, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param schur The schur decomposer to use in the eigenvalue decomposition.
     * @return The eigenvalues of the {@code src} matrix.
     */
    private static CVector getEigenValues(CMatrix src, ComplexSchur schur) {
        CVector lambdas = new CVector(src.numRows);
        CMatrix T = schur.decompose(src).getT();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.data[m] = T.data[m*numRows + m];
            } else {
                Complex128 a11 = T.data[m*numRows + m];
                Complex128 a12 = T.data[m*numRows + m + 1];
                Complex128 a21 = T.data[(m+1)*numRows + m];
                Complex128 a22 = T.data[(m+1)*numRows + m  +1];

                if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                    // Non-converged 2x2 block found.
                    Complex128[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                    lambdas.data[m] = mu[0];
                    lambdas.data[++m] = mu[1];
                } else {
                    lambdas.data[m] = a11;
                }
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(Matrix, long)
     * @see #getEigenVectors(Matrix, long, int)
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CMatrix getEigenVectors(Matrix src) {
        return getEigenVectors(src, new RealSchur(true));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(Matrix)
     * @see #getEigenVectors(Matrix, long, int)
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CMatrix getEigenVectors(Matrix src, long seed) {
        return getEigenVectors(src, new RealSchur(true, seed));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrix)
     * @see #getEigenVectors(CMatrix, long)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(Matrix src, long seed, int maxIterationFactor) {
        return getEigenVectors(src,
                new RealSchur(true, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @param schur Schur decomposer to use in the computation of the eigenvectors.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    private static CMatrix getEigenVectors(Matrix src, RealSchur schur) {
        schur.decompose(src);
        CMatrix[] complexTU = schur.real2ComplexSchur();
        CMatrix U = complexTU[1];

        if(src.isSymmetric()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrix Q = getEigenVectorsTriu(complexTU[0]); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrix, long)
     * @see #getEigenVectors(CMatrix, long, int)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(CMatrix src) {
        return getEigenVectors(src, new ComplexSchur(true));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrix)
     * @see #getEigenVectors(CMatrix, long, int)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(CMatrix src, long seed) {
        return getEigenVectors(src, new ComplexSchur(true, seed));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrix)
     * @see #getEigenVectors(CMatrix, long)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(CMatrix src, long seed, int maxIterationFactor) {
        return getEigenVectors(src,
                new ComplexSchur(true, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvectors of a square complex dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @param schur Schur decomposer to use when computing the eigenvectors.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    private static CMatrix getEigenVectors(CMatrix src, ComplexSchur schur) {
        schur.decompose(src);
        CMatrix T;
        CMatrix U;

        if(src.numRows == 2) {
            CMatrix[] complexTU = schur.real2ComplexSchur();
            T = complexTU[0];
            U = complexTU[1];
        } else {
            T = schur.getT();
            U = schur.getU();
        }

        if(src.isHermitian()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrix Q = getEigenVectorsTriu(T); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static Matrix getEigenVectorsTriu(Matrix T) {
        RealBackSolver backSolver = new RealBackSolver(false);
        Matrix Q = new Matrix(T.numRows);

        Matrix S_hat;
        Vector r;
        Vector v;

        for(int j=0; j<T.numRows; j++) {
            S_hat = new Matrix(j, j);
            r = new Vector(j);
            makeSystem(T, j, S_hat, r);

            if(S_hat.data.length > 0) {
                v = backSolver.solve(S_hat, r);
                v = v.join(new Vector(1.0));
            } else {
                v = new Vector(1.0);
            }

            v = v.normalize().join(new Vector(T.numRows-v.size));
            Q.setCol(v, j);
        }

        return Q;
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static CMatrix getEigenVectorsTriu(CMatrix T) {
        ComplexBackSolver backSolver = new ComplexBackSolver(false).setCheckSingular(false);
        CMatrix Q = new CMatrix(T.numRows);

        CMatrix S_hat;
        CVector r;
        CVector v;

        for(int j=0; j<T.numRows; j++) {
            S_hat = CMatrix.getEmpty(j, j);
            r = CVector.getEmpty(j);
            makeSystem(T, j, S_hat, r);

            if(S_hat.data.length > 0) {
                v = backSolver.solve(S_hat, r);
                // TODO: Should have an append(Complex128...) method for appending scalars to a vector to avoid having to wrap them
                //  in a vector first. Or maybe just an overloaded method.
                v = v.join(new CVector(Complex128.ONE));
            } else {
                v = new CVector(Complex128.ONE);
            }

            v = v.normalize().join(new CVector(T.numRows - v.size));
            Q.setCol(v, j);
        }

        return Q;
    }


    /**
     * Constructs the matrix (T - &lambda;*I)[0:j][0:j] for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue &lambda;.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue &lambda; appears.
     */
    private static void makeSystem(CMatrix T, int j, CMatrix S_hat, CVector r) {
        Complex128 lam = T.data[j*T.numCols + j];

        // Copy values from T and subtract eigenvalue from diagonal.
        for(int i=0; i<j; i++) {
            int tRow = i*T.numCols;
            int diffRow = i*S_hat.numCols;
            System.arraycopy(T.data, tRow, S_hat.data, diffRow, j);
            S_hat.data[diffRow + i] = S_hat.data[diffRow + i].sub(lam);
            r.data[i] = T.data[tRow + j].mult(-1);
        }
    }


    /**
     * Constructs the matrix (T - &lambda;*I)[0:j][0:j] for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue &lambda;.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue &lambda; appears.
     */
    private static void makeSystem(Matrix T, int j, Matrix S_hat, Vector r) {
        double lam = T.data[j*T.numCols + j];

        // Copy values from T and subtract eigenvalue from diagonal.
        for(int i=0; i<j; i++) {
            int tRow = i*T.numRows;
            int diffRow = i*j;
            System.arraycopy(T.data, tRow, S_hat.data, diffRow, j);
            S_hat.data[diffRow + i] -= (lam);
            r.data[i] = -T.data[tRow + j];
        }
    }


    /**
     * <p>Computes the eigenvalues and eigenvectors of a square complex matrix.
     *
     * <p>This method calculates the eigenvalues and eigenvectors of the given complex matrix {@code src}.
     * The eigenvalues are returned as a single-row matrix, while the eigenvectors are returned as a
     * matrix where each column corresponds to an eigenvector of the input matrix.
     *
     * <p>Random number generation is used internally during the computation to apply random shifts,
     * which enhance numerical stability and improve convergence. Because of this, repeated calls
     * to this method with the same matrix may produce slightly different results.
     *
     * <p>If reproducible results are required (e.g., for testing purposes), use the overloaded method
     * {@link #getEigenPairs(CMatrix, long)} that accepts a seed for the random number generator.
     *
     * @param src The input matrix for which to compute the eigenvalues and eigenvectors.
     *            This matrix must be square ({@code src.numRows == src.numCols}).
     *
     * @return An array containing two matrices:
     *         <ol>
     *             <li>The first matrix is a 1&times;{@code src.numCols} matrix containing the eigenvalues of {@code src}.</li>
     *             <li>The second matrix is an {@code src.numRows}&times;{@code src.numCols} matrix, where each column is an
     *                 eigenvector corresponding to an eigenvalue in the first matrix.</li>
     *         </ol>
     *
     * @throws IllegalArgumentException if the input matrix {@code src} is not square (e.g. {@code src.numRows != src.numCols}).
     *
     * @see #getEigenPairs(Matrix, long)
     */
    public static CMatrix[] getEigenPairs(Matrix src) {
        return getEigenPairs(src, new RealSchur(true));
    }


    /**
     * <p>Computes the eigenvalues and eigenvectors of a square real matrix.
     *
     * <p>This method calculates the eigenvalues and eigenvectors of the given real matrix {@code src}.
     * The eigenvalues are collected and returned as a single-row matrix, while the eigenvectors are collected as the columns of
     * matrix.
     *
     * <p>The computation uses a random shift algorithm to enhance numerical stability and improve convergence.
     * By providing a {@code seed}, the random shifts can be made deterministic, ensuring reproducible results
     * for the same input matrix. For most applications, specifying a seed is optional and generally not required.
     *
     * @param src The input matrix for which to compute the eigenvalues and eigenvectors.
     *            This matrix must be square ({@code src.numRows == src.numCols}).
     * @param seed A seed value for random shifts used in the algorithm. Providing a seed ensures that
     *             the results are reproducible when the same matrix is used as input.
     *             If reproducibility is not a concern, this value can be omitted or set to any arbitrary number.
     *
     * @return An array containing two matrices:
     *         <ol>
     *             <li>The first matrix is a 1&times;{@code src.numCols} matrix containing the eigenvalues of {@code src}.</li>
     *             <li>The second matrix is an {@code src.numRows}&times;{@code src.numCols} matrix, where each column is an
     *                 eigenvector corresponding to an eigenvalue in the first matrix.</li>
     *         </ol>
     *
     * @throws IllegalArgumentException if the input matrix {@code src} is not square (e.g. {@code src.numRows != src.numCols}).
     *
     * @see #getEigenPairs(Matrix)
     */
    public static CMatrix[] getEigenPairs(Matrix src, long seed) {
        return getEigenPairs(src, new RealSchur(true, seed));
    }


    /**
     * <p>Computes the eigenvalues and eigenvectors of a square real matrix.
     *
     * <p>This method calculates the eigenvalues and eigenvectors of the given real matrix {@code src}.
     * The eigenvalues are returned as a single-row matrix, while the eigenvectors are returned as a
     * matrix where each column corresponds to an eigenvector of the input matrix.
     *
     * <p>Random number generation is used internally during the computation to apply random shifts,
     * which enhance numerical stability and improve convergence. Because of this, repeated calls
     * to this method with the same matrix may produce slightly different results.
     *
     * <p>If reproducible results are required (e.g., for testing purposes), use the overloaded method
     * {@link #getEigenPairs(CMatrix, long)} that accepts a seed for the random number generator.
     *
     * @param src The input matrix for which to compute the eigenvalues and eigenvectors.
     *            This matrix must be square ({@code src.numRows == src.numCols}).
     *
     * @return An array containing two matrices:
     *         <ol>
     *             <li>The first matrix is a 1&times;{@code src.numCols} matrix containing the eigenvalues of {@code src}.</li>
     *             <li>The second matrix is an {@code src.numRows}&times;{@code src.numCols} matrix, where each column is an
     *                 eigenvector corresponding to an eigenvalue in the first matrix.</li>
     *         </ol>
     *
     * @throws IllegalArgumentException if the input matrix {@code src} is not square (e.g. {@code src.numRows != src.numCols}).
     *
     * @see #getEigenPairs(CMatrix, long)
     */
    public static CMatrix[] getEigenPairs(CMatrix src) {
        return getEigenPairs(src, new ComplexSchur(true));
    }


    /**
     * <p>Computes the eigenvalues and eigenvectors of a square complex matrix.
     *
     * <p>This method calculates the eigenvalues and eigenvectors of the given complex matrix {@code src}.
     * The eigenvalues are collected and returned as a single-row matrix, while the eigenvectors are collected as the columns of
     * matrix.
     *
     * <p>The computation uses a random shift algorithm to enhance numerical stability and improve convergence.
     * By providing a {@code seed}, the random shifts can be made deterministic, ensuring reproducible results
     * for the same input matrix. For most applications, specifying a seed is optional and generally not required.
     *
     * @param src The input matrix for which to compute the eigenvalues and eigenvectors.
     *            This matrix must be square ({@code src.numRows == src.numCols}).
     * @param seed A seed value for random shifts used in the algorithm. Providing a seed ensures that
     *             the results are reproducible when the same matrix is used as input.
     *             If reproducibility is not a concern, this value can be omitted or set to any arbitrary number.
     *
     * @return An array containing two matrices:
     *         <ol>
     *             <li>The first matrix is a 1&times;{@code src.numCols} matrix containing the eigenvalues of {@code src}.</li>
     *             <li>The second matrix is an {@code src.numRows}&times;{@code src.numCols} matrix, where each column is an
     *                 eigenvector corresponding to an eigenvalue in the first matrix.</li>
     *         </ol>
     *
     * @throws IllegalArgumentException if the input matrix {@code src} is not square (e.g. {@code src.numRows != src.numCols}).
     *
     * @see #getEigenPairs(CMatrix)
     */
    public static CMatrix[] getEigenPairs(CMatrix src, long seed) {
        return getEigenPairs(src, new ComplexSchur(true, seed));
    }


    /**
     * Helper method to compute the eigenvalues and eigenvectors of a square complex matrix.
     * @param src The matrix to compute the eigenvalues and eigenvectors of.
     * @param schur The Schur decomposer to use in computing the eigenvalues and eigenvectors
     * @return An array containing two matrices. The first matrix has shape 1&times;{@code src.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRows}&times;{@code src.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    private static CMatrix[] getEigenPairs(CMatrix src, ComplexSchur schur) {
        CMatrix lambdas = new CMatrix(1, src.numRows);

        schur.decompose(src);
        int numRows = src.numRows;
        CMatrix T;
        CMatrix U;

        if(src.numRows == 2) {
            CMatrix[] complexTU = schur.real2ComplexSchur();
            T = complexTU[0];
            U = complexTU[1];
        } else {
            T = schur.getT();
            U = schur.getU();
        }

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++)
            lambdas.data[i] = T.data[i*numRows + i];

        // If the src matrix is Hermitian, then U will contain the eigenvectors.
        if(!src.isHermitian())
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.

        return new CMatrix[]{lambdas, U};
    }


    /**
     * Helper method to compute the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and eigenvectors of.
     * @param schur The Schur decomposer to use in computing the eigenvalues and eigenvectors
     * @return An array containing two matrices. The first matrix has shape 1&times;{@code src.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRows}&times;{@code src.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    private static CMatrix[] getEigenPairs(Matrix src, RealSchur schur) {
        int numRows = src.numRows;
        CMatrix lambdas = new CMatrix(1, src.numRows);

        schur.decompose(src);
        CMatrix[] complexTU = schur.real2ComplexSchur();
        CMatrix T = complexTU[0];
        CMatrix U = complexTU[1];

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++)
            lambdas.data[i] = T.data[i*numRows + i];

        // If the source matrix is symmetric, then U will contain its eigenvectors.
        if(!src.isSymmetric())
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.

        return new CMatrix[]{lambdas, U};
    }
}
