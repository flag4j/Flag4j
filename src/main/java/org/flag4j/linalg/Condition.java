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

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;

import java.util.function.BiFunction;


/**
 * <p>Utility class for computing the condition number of a matrix.
 *
 * <p>The condition number of a matrix A is defined as the norm of A times the norm of A<sup>-1</sup> (i.e., the norm of the inverse
 * of A). That is, cond(A) = ||A|| * ||A<sup>-1</sup>|| where ||A|| may be any matrix norm (generally taken to be the L2-norm).
 *
 * <p> Conditions numbers are associated with a linear equation Ax = b and provides a bound on how inaccurate the solution x will be
 * after approximation before round-off errors are taken into account. If the condition number is large, then a small error in b may
 * result in large errors in x. Subsequently, if the condition number is small, then the error in x will not be much bigger than the
 * error in b.
 *
 * <p>The condition number is more precisely defined to be the maximum ratio of the relative error in x to the relative error in b.
 * Let <i>e</i> be the error in b and A be a nonsingular matrix. Then,
 * <pre>
 *     cond(A) = max<sub>e,b&ne;0</sub>{ (||b|| / ||A<sup>-1</sup>b||) * (||A<sup>-1</sup>e|| / ||<i>e</i>||) }
 *             = ||A|| * ||A<sup>-1</sup>|| as stated.</pre>
 *
 * <p>When the L2-norm is used to compute the condition number then,
 * <pre>
 *     cond(A) = σ<sub>max</sub>(A) / σ<sub>min</sub>(A)</pre>
 * where σ<sub>max</sub>(A) and σ<sub>min</sub>(A) are the maximum and minimum singular values of the matrix A.
 *
 * <p>This class supports the computation of the condition number of a real or complex matrix using the following norms.
 * <ul>
 *     <li>Induced (operator) norm: {@link #cond(Matrix, double)} and {@link #cond(CMatrix, double)}.</li>
 *     <li>Schatten norm: {@link #condSchatten(Matrix, double)} and {@link #condSchatten(CMatrix, double)}.</li>
 *     <li>Frobenius norm: {@link #condFro(Matrix)} and {@link #condFro(CMatrix)}.</li>
 *     <li>Entry-wise norm: {@link #condEntryWise(Matrix, double)} and {@link #condEntryWise(CMatrix, double)}</li>
 * </ul>
 */
public final class Condition {

    private Condition() {
        // Hide default constructor for utility class
    }


    /**
     * <p>Computes the condition number of a matrix.
     *
     * <p>This method computes the condition number using the matrix operator norm induced by the
     * vector p-norm ({@link MatrixNorms#inducedNorm(Matrix, double)}). {@code p} must be one of the following:
     * <ul>
     *     <li>{@code p=1}: Maximum absolute column sum of the matrix.</li>
     *     <li>{@code p=-1}: Minimum absolute column sum of the matrix.</li>
     *     <li>{@code p=2}: Spectral norm. Equivalent to the maximum singular value of the matrix.</li>
     *     <li>{@code p=-2}: Equivalent to the minimum singular value of the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: Maximum absolute row sum of the matrix.</li>
     *     <li>{@code p=Double.NEGATIVE_INFINITY}: Minimum absolute row sum of the matrix.</li>
     * </ul>
     * When {@code p < 1}, the "norm" is not a true mathematical norm but may still serve useful numerical purposes.
     *
     * <p>To compute the condition number using other norms see one the below methods:
     * <ul>
     *     <li>Schatten norm: {@link #condSchatten(Matrix, double)}.</li>
     *     <li>Frobenius norm: {@link #condFro(Matrix)}</li>
     *     <li>Entry-wise norm: {@link #condEntryWise(Matrix, double)}</li>
     * </ul>
     *
     * @param src The matrix to compute the condition number of.
     * @param p The p-value to use in the induced norm during condition number computation.
     * Must be one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2}, {@link Double#POSITIVE_INFINITY} or
     * {@link Double#NEGATIVE_INFINITY}.
     * @return The condition number of {@code src} as computed using the matrix operator norm induced by vector p-norm.
     */
    public static double cond(Matrix src, double p) {
        if(p == 2.0 || p == -2.0) {
            // Special case for spectral norm. No need to invert matrix explicitly.
            Vector s = new RealSVD(false).decompose(src).getSingularValues();
            return (p==2) ? s.max()/s.min() : s.min()/s.max();
        }

        return cond(src, p, MatrixNorms::inducedNorm);
    }


    /**
     * <p>Computes the condition number of a matrix.
     *
     * <p>This method computes the condition number using the matrix operator norm induced by the
     * vector p-norm ({@link MatrixNorms#inducedNorm(CMatrix, double)}). {@code p} must be one of the following:
     * <ul>
     *     <li>{@code p=1}: Maximum absolute column sum of the matrix.</li>
     *     <li>{@code p=-1}: Minimum absolute column sum of the matrix.</li>
     *     <li>{@code p=2}: Spectral norm. Equivalent to the maximum singular value of the matrix.</li>
     *     <li>{@code p=-2}: Equivalent to the minimum singular value of the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: Maximum absolute row sum of the matrix.</li>
     *     <li>{@code p=Double.NEGATIVE_INFINITY}: Minimum absolute row sum of the matrix.</li>
     * </ul>
     * When {@code p < 1}, the "norm" is not a true mathematical norm but may still serve useful numerical purposes.
     *
     * <p>To compute the condition number using other norms see one the below methods:
     * <ul>
     *     <li>Schatten norm: {@link #condSchatten(CMatrix, double)}.</li>
     *     <li>Frobenius norm: {@link #condFro(CMatrix)}</li>
     *     <li>Entry-wise norm: {@link #condEntryWise(CMatrix, double)}</li>
     * </ul>
     *
     * @param src The matrix to compute the condition number of.
     * @param p The p-value to use in the induced norm during condition number computation.
     * Must be one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2}, {@link Double#POSITIVE_INFINITY} or
     * {@link Double#NEGATIVE_INFINITY}.
     * @return The condition number of {@code src} as computed using the matrix operator norm induced by vector p-norm.
     */
    public static double cond(CMatrix src, double p) {
        if(p == 2.0 || p == -2.0) {
            // Special case for spectral norm. No need to invert matrix explicitly.
            Vector s = new ComplexSVD(false).decompose(src).getSingularValues();
            return (p==2) ? s.max()/s.min() : s.min()/s.max();
        }

        return cond(src, p, MatrixNorms::inducedNorm);
    }


    /**
     * Computes the condition number of a matrix using the {@link MatrixNorms#schattenNorm(Matrix, double) Schatten norm}.
     * @param src Matrix to compute the condition number of.
     * @param p The p value in the Schatten norm.
     * @return The condition number of {@code src}.
     */
    public static double condSchatten(Matrix src, double p) {
        return cond(src, p, MatrixNorms::schattenNorm);
    }


    /**
     * Computes the condition number of a matrix using the {@link MatrixNorms#schattenNorm(CMatrix, double) Schatten norm}.
     * @param src Matrix to compute the condition number of.
     * @param p The p value in the Schatten norm.
     * @return The condition number of {@code src}.
     */
    public static double condSchatten(CMatrix src, double p) {
        return cond(src, p, MatrixNorms::schattenNorm);
    }


    /**
     * Computes the condition number of a matrix using the Frobenius norm.
     * @param src Matrix to compute the condition number of.
     * @return The condition number of {@code src}.
     */
    public static double condFro(Matrix src) {
        return cond(src, 2, MatrixNorms::schattenNorm);
    }


    /**
     * Computes the condition number of a matrix using the Frobenius norm.
     * @param src Matrix to compute the condition number of.
     * @return The condition number of {@code src}.
     */
    public static double condFro(CMatrix src) {
        return cond(src, 2, MatrixNorms::schattenNorm);
    }


    /**
     * Computes the condition number of a matrix using an {@link MatrixNorms#entryWiseNorm(Matrix, double) entry-wise norm}.
     * @param src Matrix to compute the condition number of.
     * @return The condition number of {@code src}.
     */
    public static double condEntryWise(Matrix src, double p) {
        return cond(src, p, MatrixNorms::entryWiseNorm);
    }


    /**
     * Computes the condition number of a matrix using an {@link MatrixNorms#entryWiseNorm(CMatrix, double) entry-wise norm}.
     * @param src Matrix to compute the condition number of.
     * @return The condition number of {@code src}.
     */
    public static double condEntryWise(CMatrix src, double p) {
        return cond(src, p, MatrixNorms::entryWiseNorm);
    }


    /**
     * Computes the condition number of a matrix using the specified norm.
     * @param src Matrix to compute the condition number of.
     * @param p The p-value in the norm.
     * @param norm The norm to apply when computing the condition number.
     * @return
     */
    private static double cond(Matrix src, double p, BiFunction<Matrix, Double, Double> norm) {
        return norm.apply(src, p) * norm.apply(Invert.inv(src), p);
    }


    /**
     * Computes the condition number of a matrix using the specified norm.
     * @param src Matrix to compute the condition number of.
     * @param p The p-value in the norm.
     * @param norm The norm to apply when computing the condition number.
     * @return
     */
    private static double cond(CMatrix src, double p, BiFunction<CMatrix, Double, Double> norm) {
        return norm.apply(src, p) * norm.apply(Invert.inv(src), p);
    }
}
