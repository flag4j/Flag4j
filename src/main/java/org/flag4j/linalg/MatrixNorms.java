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

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingMatrix;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.linalg.ops.sparse.coo.real.RealSparseNorms;
import org.flag4j.linalg.ops.sparse.coo.ring_ops.CooRingNorms;
import org.flag4j.numbers.Ring;
import org.flag4j.util.ValidateParameters;

import java.util.function.Function;


/**
 * <p>A utility class providing a range of matrix norm computations for both real and complex matrices.</p>
 *
 * <h2>Overview</h2>
 * <p>This class includes static methods to compute:</p>
 * <ul>
 *   <li><strong>Schatten norms</strong> (p-norms of the singular values), including:
 *       <ul>
 *         <li>Nuclear norm (p=1)</li>
 *         <li>Frobenius norm (p=2)</li>
 *         <li>Spectral norm (p = {@link Double#POSITIVE_INFINITY})</li>
 *       </ul>
 *   </li>
 *   <li><strong>Induced norms</strong> (operator norms) for specific values of p:
 *       <ul>
 *         <li>p = 1 or -1 (maximum/minimum absolute column sums)</li>
 *         <li>p = 2 or -2 (largest/smallest singular value)</li>
 *         <li>p = {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}
 *             (maximum/minimum absolute row sum)</li>
 *       </ul>
 *   </li>
 *   <li><strong>L<sub>p,q</sub> norms</strong>
 *   for both dense and sparse (COO/CSR) matrices.</li>
 *   <li>Common norms like the <strong>Frobenius norm</strong>, <strong>maximum absolute value</strong> (max norm),
 *       and <strong>infinite norm</strong> (maximum row sum) for real and complex matrices.</li>
 *   <li><strong>Entry-wise p-norms</strong>, computed by flattening the matrix and computing the vector p-norm.</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * Matrix A = ...; // some real matrix
 * double fro = MatrixNorms.norm(A); // Frobenius norm
 * double nuc = MatrixNorms.shattenNorm(A, 1.0); // nuclear norm
 * double spec = MatrixNorms.inducedNorm(A, Double.POSITIVE_INFINITY); // spectral norm
 * }</pre>
 *
 * <p>For complex matrices, use the corresponding overloads that accept a {@code CMatrix} or other complex matrix type.</p>
 */
public final class MatrixNorms {

    private MatrixNorms() {
        // Hide default constructor for utility class
    }


    /**
     * <p>Computes the Schatten p-norm of a real dense matrix. This is equivalent to the p-norm of the vector of singular values of the
     * matrix.
     *
     * <p>This method accepts values of p which are negative. When {@code p < 0} the result is not a true norm but may still have
     * numerical uses.
     *
     * @param src The matrix to compute the norm of.
     * @param p The p value in the Schatten p-norm. Some common cases include:
     * <ul>
     *     <li>{@code p=1}: The nuclear (or trace) norm. Equivalent to the sum of singular values.</li>
     *     <li>{@code p=2}: Frobenius (or <span class="latex-inline">L<sub>2, 2</sub></span>) norm.
     *     Equivalent to the square root of the sum of the absolute squares
     *     of all entries in the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: The spectral norm. Equivalent to the maximum singular value.</li>
     *     <li>{@code p=Double.NEGATIVE_INFINITY}: The minimum singular value.</li>
     * </ul>
     * @return The Schatten p-norm of {@code src}.
     */
    public static double schattenNorm(Matrix src,  double p) {
        if(p == 1.0) {
            return nuclearNorm(src); // Nuclear norm.
        } else if(p == 2.0) {
            return VectorNorms.norm(src.data); // Frobenius norm.
        } else if(p == Double.POSITIVE_INFINITY) {
            return svdBasedNorm(src, RealProperties::max); // Spectral norm.
        } else {
            Vector sigmas = new RealSVD(false, true).decompose(src).getSingularValues();
            return VectorNorms.norm(sigmas.data, p);
        }
    }


    /**
     * <p>Computes the Schatten p-norm of a complex dense matrix. This is equivalent to the p-norm of
     * the vector of singular values of the matrix.
     *
     * @param src The matrix to compute the norm of.
     * @param p The p value in the Schatten p-norm. Must be greater than or equal to 1. Some common cases include:
     * <ul>
     *     <li>{@code p=1}: The nuclear (or trace) norm. Equivalent to the sum of singular values.</li>
     *     <li>{@code p=2}: Frobenius (or <span class="latex-inline">L<sub>2, 2</sub></span>) norm.
     *     Equivalent to the square root of the sum of the absolute squares
     *     of all entries in the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: The spectral norm. Equivalent to the largest singular value.</li>
     * </ul>
     * @return The Schatten p-norm of {@code src}.
     * @throws IllegalArgumentException If {@code p < 1}.
     */
    public static double schattenNorm(CMatrix src,  double p) {
        ValidateParameters.ensureGreaterEq(1, p, "p");

        if(p == 1.0) {
            return nuclearNorm(src); // Nuclear norm.
        } else if(p == 2.0) {
            return VectorNorms.norm(src.data); // Frobenius norm.
        } else if(p == Double.POSITIVE_INFINITY) {
            return svdBasedNorm(src, RealProperties::max); // Spectral norm.
        } else {
            Vector sigmas = new ComplexSVD(false, true).decompose(src).getSingularValues();
            return VectorNorms.norm(sigmas.data, p);
        }
    }


    /**
     * <p>Computes the matrix operator norm of a real dense matrix "induced" by the vector p-norm.
     * Specifically, this method computes the operator norm of the matrix as:
     * <span class="latex-replace"><pre>
     *     ||A||<sub>p</sub> = sup<sub>x&ne;0</sub>(||Ax||<sub>p</sub> / ||x||<sub>p</sub>).</pre></span>
     *
     * <!-- LATEX: \[ ||A||_p = \sup_{x \ne 0} \cfrac{||Ax||_p}{||x||_p} \] -->
     *
     * <p>This method supports a limited set of {@code p} values which yield simple formulas. When {@code p < 1}, the result this method
     * returns is not a true mathematical norm. However, these values may still be useful for numerical purposes.
     * <ul>
     *     <li>{@code p=1}: The maximum absolute column sum.</li>
     *     <li>{@code p=-1}: The minimum absolute column sum.</li>
     *     <li>{@code p=2}: The spectral norm. Equivalent to the largest singular value of the matrix.</li>
     *     <li>{@code p=-2}: The smallest singular value of the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: The maximum absolute row sum.</li>
     *     <li>{@code p=Double.NEGATIVE_INFINITY}: The minimum absolute row sum.</li>
     * </ul>
     *
     * @param src Matrix to compute the norm of.
     * @param p The p value in the "induced" p-norm. Must be one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2},
     * {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * @return Norm of the matrix.
     * @throws IllegalArgumentException If {@code p} is not one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2},
     * {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     */
    public static double inducedNorm(Matrix src, double p) {
        if(p == 1.0) {
            return colBasedNorm(src.shape, src.data, RealProperties::max);
        } else if(p == -1.0) {
            return colBasedNorm(src.shape, src.data, RealProperties::min);
        } else if(p == 2.0) {
            return svdBasedNorm(src, RealProperties::max);
        } else if(p == -2.0) {
            return svdBasedNorm(src, RealProperties::min);
        } else if(p == Double.POSITIVE_INFINITY) {
            return rowBasedNorm(src.shape, src.data, RealProperties::max);
        } else if(p == Double.NEGATIVE_INFINITY) {
            return rowBasedNorm(src.shape, src.data, RealProperties::min);
        } else {
            throw new IllegalArgumentException("Unsupported norm type: p = " + p + ".\n"
                    + "Supported values are: 1, -1, 2, -2, Double.POSITIVE_INFINITY, and Double.NEGATIVE_INFINITY.");
        }
    }


    /**
     * <p>Computes the matrix operator norm of a complex dense matrix "induced" by the vector p-norm.
     * Specifically, this method computes the operator norm of the matrix as:
     * <span class="latex-replace"><pre>
     *     ||A||<sub>p</sub> = sup<sub>x&ne;0</sub>(||Ax||<sub>p</sub> / ||x||<sub>p</sub>).</pre></span>
     *
     * <!-- LATEX: \[ ||A||_p = \sup_{x \ne 0} \cfrac{||Ax||_p}{||x||_p} \] -->
     *
     * <p>This method supports a limited set of {@code p} values which yield simple formulas. When {@code p < 1}, the result this method
     * returns is not a true mathematical norm. However, these values may still be useful for numerical purposes.
     * <ul>
     *     <li>{@code p=1}: The maximum absolute column sum.</li>
     *     <li>{@code p=-1}: The minimum absolute column sum.</li>
     *     <li>{@code p=2}: The spectral norm. Equivalent to the largest singular value of the matrix.</li>
     *     <li>{@code p=-2}: The smallest singular value of the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: The maximum absolute row sum.</li>
     *     <li>{@code p=Double.NEGATIVE_INFINITY}: The minimum absolute row sum.</li>
     * </ul>
     *
     * @param src Matrix to compute the norm of.
     * @param p The p value in the "induced" p-norm. Must be one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2},
     * {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * @return Norm of the matrix.
     * @throws IllegalArgumentException If {@code p} is not one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2},
     * {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     */
    public static double inducedNorm(CMatrix src, double p) {
        if(p == 1.0) {
            return colBasedNorm(src.shape, src.data, RealProperties::max);
        } else if(p == -1.0) {
            return colBasedNorm(src.shape, src.data, RealProperties::min);
        } else if(p == 2.0) {
            return svdBasedNorm(src, RealProperties::max);
        } else if(p == -2.0) {
            return svdBasedNorm(src, RealProperties::min);
        } else if(p == Double.POSITIVE_INFINITY) {
            return rowBasedNorm(src.shape, src.data, RealProperties::max);
        } else if(p == Double.NEGATIVE_INFINITY) {
            return rowBasedNorm(src.shape, src.data, RealProperties::min);
        } else {
            throw new IllegalArgumentException("Unsupported norm type: p = " + p + ".\n"
                    + "Supported values are: 1, -1, 2, -2, Double.POSITIVE_INFINITY, and Double.NEGATIVE_INFINITY.");
        }
    }


    /**
     * <p>Computes the Frobenius (or <span class="latex-inline">L<sub>2, 2</sub></span>) norm of a real dense matrix.
     *
     * <p>The Frobenius norm is defined as the square root of the sum of absolute squares of all entries in the matrix.
     *
     * <p>This method is equivalent to {@link #norm(Matrix, double, double) norm(src, 2, 2)}.
     * However, this method should generally be preferred over
     * {@link #norm(Matrix, double, double)} as it <em>may</em> be slightly more efficient.
     *
     * @param src Matrix to compute theFrobenius norm of.
     *
     * @return the Frobenius of this tensor.
     * @see #norm(Matrix, double, double)
     */
    public static double norm(Matrix src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * <p>Computes the Frobenius (or <span class="latex-inline">L<sub>2, 2</sub></span>) norm of a dense
     * {@link org.flag4j.numbers.Ring} matrix.
     *
     * <p>The Frobenius norm is defined as the square root of the sum of absolute squares of all entries in the matrix.
     *
     * <p>This method is equivalent to {@link #norm(AbstractDenseRingMatrix, double, double) norm(src, 2, 2)}.
     * However, this method should generally be preferred over
     * {@link #norm(AbstractDenseRingMatrix, double, double)} as it <em>may</em> be slightly more efficient.
     *
     * @param src Matrix to compute theFrobenius norm of.
     *
     * @return the Frobenius of this tensor.
     * @see #norm(AbstractDenseRingMatrix, double, double)
     */
    public static double norm(AbstractDenseRingMatrix<?, ?, ?> src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * Computes the maximum norm of this matrix. That is, the maximum value in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The maximum norm of this matrix.
     * @see #infNorm(Matrix)
     */
    public static double maxNorm(Matrix src) {
        return RealProperties.maxAbs(src.data);
    }


    /**
     * Computes the maximum norm of this matrix. That is, the maximum value in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The maximum norm of this matrix.
     * @see #infNorm(AbstractDenseRingMatrix)
     */
    public static double maxNorm(AbstractDenseRingMatrix<?, ?, ?> src) {
        return CompareRing.maxAbs(src.data);
    }


    /**
     * Computes the infinite norm of this matrix. That is the maximum row sum in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The infinite norm of this matrix.
     * @see #maxNorm(Matrix)
     */
    public static double infNorm(Matrix src) {
        return rowBasedNorm(src.shape, src.data, RealProperties::max);
    }


    /**
     * Computes the infinite norm of this matrix. That is the maximum row sum in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The infinite norm of this matrix.
     * @see #maxNorm(Matrix)
     */
    public static double infNorm(AbstractDenseRingMatrix<?, ?, ?> src) {
        return rowBasedNorm(src.shape, src.data, RealProperties::max);
    }


    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of a real dense matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector of singular values of
     *     the matrix.</li>
     * </ul>
     *
     * <p>The <span class="latex-inline">L<sub>p,q</sub></span> norm is computed as if by:
     * <pre>{@code
     *      double norm = 0;
     *      for(int j=0; j<src.numCols; j++) {
     *          double sum = 0;
     *          for(int i=0; i<src.numRows; i++)
     *              sum += Math.pow(Math.abs(src.get(i, j)), p);
     *
     *          norm += Math.pow(sum, q / p);
     *      }
     *
     *      return Math.pow(norm, 1.0 / q);
     * }</pre>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public static double norm(Matrix src, double p, double q) {
        if(p == q) return VectorNorms.norm(src.data, p);
        return matrixNormLpq(src.data, src.shape, p, q);
    }


    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of a dense {@link Ring} matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector of singular values of
     *     the matrix.</li>
     * </ul>
     *
     * <p>The <span class="latex-inline">L<sub>p,q</sub></span> norm is computed as if by:
     * <pre>{@code
     *      double norm = 0;
     *      for(int j=0; j<src.numCols; j++) {
     *          double sum = 0;
     *          for(int i=0; i<src.numRows; i++)
     *              sum += Math.pow(Math.abs(src.get(i, j)), p);
     *
     *          norm += Math.pow(sum, q / p);
     *      }
     *
     *      return Math.pow(norm, 1.0 / q);
     * }</pre>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public static double norm(AbstractDenseRingMatrix<?, ?, ?> src, double p, double q) {
        if(p == q) return VectorNorms.norm(src.data, p);
        return matrixNormLpq(src.data, src.shape, p, q);
    }


    /**
     * <p>Computes the entry-wise p-norm of a real dense matrix.
     *
     * <p>The entry-wise p-norm of a matrix is equivalent to the
     * vector &ell;<sup>p</sup> norm computed on the flattened matrix as if by {@code src.toVector().norm(p);}.
     *
     * @param src The matrix to compute the entry-wise norm of.
     * @param p The p value in the &ell;<sup>p</sup> vector norm.
     * @return The entry-wise norm of {@code src}.
     */
    public static double entryWiseNorm(Matrix src, double p) {
        return VectorNorms.norm(src.data, p);
    }


    /**
     * <p>Computes the entry-wise p-norm of a complex dense matrix.
     *
     * <p>The entry-wise p-norm of a matrix is equivalent to the
     * vector &ell;<sup>p</sup> norm computed on the flattened matrix as if by {@code src.toVector().norm(p);}.
     *
     * @param src The matrix to compute the entry-wise norm of.
     * @param p The p value in the &ell;<sup>p</sup> vector norm.
     * @return The entry-wise norm of {@code src}.
     */
    public static double entryWiseNorm(CMatrix src, double p) {
        return VectorNorms.norm(src.data, p);
    }


    // ------------------------------ Sparse COO Matrices ------------------------------
    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of a real COO matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector of singular values of
     *     the matrix.</li>
     * </ul>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public static double norm(CooMatrix src, double p, double q) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.getSparsity()>=0.95 ? RealSparseNorms.matrixNormLpq(src, p, q) :
                norm(src.toDense(), p, q);
    }


    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of a complex COO matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector of singular values of
     *     the matrix.</li>
     * </ul>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public static double norm(CooCMatrix src, double p, double q) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.getSparsity()>=0.95 ? CooRingNorms.matrixNormLpq(src, p, q) :
                norm(src.toDense(), p, q);
    }


    /**
     * Computes the Frobenius (<span class="latex-inline">L<sub>2, 2</sub></span>) norm of this complex COO matrix. This is equivalent to
     * {@link #norm(CooMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the <span class="latex-inline">L<sub>2, 2</sub></span> norm of.
     * @return the Frobenius (<span class="latex-inline">L<sub>2, 2</sub></span>) norm of this tensor.
     */
    public static double norm(CooMatrix src) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.getSparsity()>=0.95 ? RealSparseNorms.matrixNormL2(src) :
                norm(src.toDense());
    }


    /**
     * Computes the Frobenius (<span class="latex-inline">L<sub>2, 2</sub></span>) norm of this complex COO matrix. This is equivalent to
     * {@link #norm(CooCMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the <span class="latex-inline">L<sub>2, 2</sub></span> norm of.
     * @return the Frobenius (<span class="latex-inline">L<sub>2, 2</sub></span>) norm of this tensor.
     */
    public static double norm(CooCMatrix src) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.getSparsity()>=0.95 ? CooRingNorms.matrixNormL22(src) :
                norm(src.toDense());
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooCMatrix src) {
        return CompareRing.maxAbs(src.data);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooMatrix src) {
        return RealProperties.maxAbs(src.data);
    }

    // ------------------------------ Sparse CSR Matrices ------------------------------

    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of a real CSR matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector of singular values of
     *     the matrix.</li>
     * </ul>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public static double norm(CsrMatrix src, double p, double q) {
        if(p == 0 || q == 0)
            throw new IllegalArgumentException("p and q must be non-zero for norm.");

        double norm = 0;
        double qOverP = q / p;

        // stores intermediate column norms.
        double[] colNorms = new double[src.numCols];

        // Accumulate column-wise norms.
        for (int row = 0; row < src.numRows; row++) {
            int start = src.rowPointers[row];
            int end = src.rowPointers[row + 1];

            for (int idx = start; idx < end; idx++) {
                int col = src.colIndices[idx];
                double value = src.data[idx];

                colNorms[col] += Math.pow(Math.abs(value), p);
            }
        }

        // Compute the q-norm of the column norms.
        for (double colNorm : colNorms)
            if (colNorm > 0) norm += Math.pow(colNorm, qOverP);

        return Math.pow(norm, 1.0 / q);
    }


    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of a complex CSR matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector of singular values of
     *     the matrix.</li>
     * </ul>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public static double norm(CsrCMatrix src, double p, double q) {
        if(p == 0 || q == 0)
            throw new IllegalArgumentException("p and q must be non-zero for norm.");

        double norm = 0;
        double qOverP = q / p;

        // stores intermediate column norms.
        double[] colNorms = new double[src.numCols];

        // Accumulate column-wise norms.
        for (int row = 0; row < src.numRows; row++) {
            int start = src.rowPointers[row];
            int end = src.rowPointers[row + 1];

            for (int idx = start; idx < end; idx++) {
                int col = src.colIndices[idx];
                double value = src.data[idx].mag();

                colNorms[col] += Math.pow(value, p);
            }
        }

        // Compute the q-norm of the column norms.
        for (double colNorm : colNorms)
            if (colNorm > 0) norm += Math.pow(colNorm, qOverP);

        return Math.pow(norm, 1.0 / q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CsrMatrix src) {
        return RealProperties.maxAbs(src.data);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CsrCMatrix src) {
        return CompareRing.maxAbs(src.data);
    }


    /**
     * Computes the Frobenius (<span class="latex-inline">L<sub>2, 2</sub></span>) of this matrix. This is equivalent to {@link #norm(CsrMatrix, double, double) norm
     * (src, 2, 2)}.
     *
     * @param src Matrix to compute the norm of.
     * @return the Frobenius of this matrix.
     */
    public static double norm(CsrMatrix src) {
        return VectorNorms.norm(src.data); // Zeros do not contribute to this norm.
    }


    /**
     * Computes the Frobenius of this matrix. This is equivalent to {@link #norm(CsrCMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the norm of.
     * @return the Frobenius of this matrix.
     */
    public static double norm(CsrCMatrix src) {
        return VectorNorms.norm(src.data); // Zeros do not contribute to this norm.
    }

    // -------------------------------------------------- Low-level implementations --------------------------------------------------

    /**
     * Compute the <span class="latex-inline">L<sub>p,q</sub></span> norm of a matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p First parameter in <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q Second parameter in <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of the matrix.
     */
    private static double matrixNormLpq(double[] src, Shape shape, double p, double q) {
        if(p == 0 || q == 0)
            throw new IllegalArgumentException("p and q must be non-zero for norm but got p=" + p + " and q=" + q + ".");

        double norm = 0;
        double colSum;
        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++)
                colSum += Math.pow(Math.abs(src[i*cols + j]), p);
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1.0/q);
    }


    /**
     * Compute the <span class="latex-inline">L<sub>p,q</sub></span> norm of a matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p First parameter in <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q Second parameter in <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of the matrix.
     */
    private static <T extends Ring<T>> double matrixNormLpq(T[] src, Shape shape, double p, double q) {
        if(p == 0 || q == 0)
            throw new IllegalArgumentException("p and q must be non-zero for norm but got p=" + p + " and q=" + q + ".");

        double norm = 0;
        double colSum;
        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++)
                colSum += Math.pow(src[i*cols + j].mag(), p);
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1.0/q);
    }


    /**
     * Helper method for computing a matrix norm which is based on the absolute row sums.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param aggregator Operation to apply to absolute row sums.
     *
     * @return The row-based matrix norm.
     */
    private static double rowBasedNorm(Shape shape, double[] src, Function<double[], Double> aggregator) {
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                rowSums[i] += Math.abs(src[rowOffset + j]);
        }

        return aggregator.apply(rowSums);
    }


    /**
     * Helper method for computing a matrix norm which is based on the absolute row sums.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param aggregator Operation to apply to absolute row sums.
     *
     * @return The row-based matrix norm.
     */
    private static <T extends Ring<T>> double rowBasedNorm(Shape shape, T[] src, Function<double[], Double> aggregator) {
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                rowSums[i] += src[rowOffset + j].mag();
        }

        return aggregator.apply(rowSums);
    }


    /**
     * Helper for computing an SVD based norm of a real dense matrix.
     *
     * @param src Matrix for which to compute SVD based norm.
     * @param aggregator Operation to apply to the vector of singular values.
     *
     * @return The result of applying the {@code aggregator} function to the singular values of {@code src}.
     */
    private static double svdBasedNorm(Matrix src, Function<double[], Double> aggregator) {
        Vector sigmas = new RealSVD(false, true).decompose(src).getSingularValues();
        return aggregator.apply(sigmas.data);
    }


    /**
     * Helper for computing an SVD based norm of a complex dense matrix.
     *
     * @param src Matrix for which to compute SVD based norm.
     * @param aggregator Operation to apply to the vector of singular values.
     *
     * @return The result of applying the {@code aggregator} function to the singular values of {@code src}.
     */
    private static double svdBasedNorm(CMatrix src, Function<double[], Double> aggregator) {
        Vector sigmas = new ComplexSVD(false, true).decompose(src).getSingularValues();
        return aggregator.apply(sigmas.data);
    }


    /**
     * Helper method for computing a matrix norm which is based on the absolute column sums.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param aggregator Operation to apply to absolute column sums.
     *
     * @return The column-based matrix norm.
     */
    private static double colBasedNorm(Shape shape, double[] src, Function<double[], Double> aggregator) {
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] colSums = new double[cols];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                colSums[j] += Math.abs(src[rowOffset + j]);
        }

        return aggregator.apply(colSums);
    }


    /**
     * Helper method for computing a matrix norm which is based on the absolute column sums.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param aggregator Operation to apply to absolute column sums.
     *
     * @return The column-based matrix norm.
     */
    private static <T extends Ring<T>> double colBasedNorm(Shape shape, T[] src, Function<double[], Double> aggregator) {
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] colSums = new double[cols];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                colSums[j] += src[rowOffset + j].mag();
        }

        return aggregator.apply(colSums);
    }


    /**
     * Computes the nuclear norm of a real dense matrix. Equivalent to the sum of singular values of the matrix.
     * @param src The matrix to compute the norm of.
     * @return The nuclear norm of {@code src}.
     */
    private static double nuclearNorm(Matrix src) {
        return new RealSVD(false, true)
                .decompose(src)
                .getSingularValues()
                .sum();
    }


    /**
     * Computes the nuclear norm of a complex dense matrix. Equivalent to the sum of singular values of the matrix.
     * @param src The matrix to compute the norm of.
     * @return The nuclear norm of {@code src}.
     */
    private static double nuclearNorm(CMatrix src) {
        return new ComplexSVD(false, true)
                .decompose(src)
                .getSingularValues()
                .sum();
    }
}
