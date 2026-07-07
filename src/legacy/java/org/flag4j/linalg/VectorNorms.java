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

import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.numbers.RingElement;


/**
 * A utility class for computing vector norms, including various types of
 * <span class="latex-inline">&ell;<sup>p</sup></span> norms,
 * with support for both dense and sparse vectors. This class provides methods to compute norms
 * for vectors with real entries as well as vectors with entries that belong to a {@link RingElement}.
 *
 * <p>The methods in this class use scaling internally when computing the
 * <span class="latex-inline">&ell;<sup>p</sup></span> norm to protect against
 * overflow and underflow for very large or very small values of {@code p} (in absolute value).
 *
 * <p><strong>Note:</strong> When {@code p < 1}, the results of the
 * <span class="latex-inline">&ell;<sup>p</sup></span> norm methods are not
 * technically true mathematical norms but may still be useful for numerical tasks. However, {@code p = 0}
 * will result in {@link Double#NaN}.
 *
 * <p>This class is designed to be stateless and is not intended to be instantiated.
 */
public final class VectorNorms {

    // TODO: This class currently uses scaling to avoid potential over/underflow issues. This works for most
    //  reasonable inputs. However, for very large (or small) values over/underflow may still occur.
    //  To better combat this, multiple accumulators can be used with different scaling as is done by LAPACK/BLAS drnm2.
    //  Specifically, a "big" accumulator which scales values down to ovoid overflow, a "small" accumulator which scales values up
    //  to avoid underflow, and a "medium" accumulator which applies no scaling.

    private VectorNorms() {
        // Hide default constructor for utility class
    }

    /**
     * <p>Computes the Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm of a real dense or sparse vector.
     * <p>Zeros do not contribute to this norm, so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero items if the vector is sparse) to compute norm of.
     * @return Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm.
     */
    public static double norm(double... src) {
        // Some quick returns.
        if (src.length == 0) return 0;
        else if (src.length == 1) return Math.abs(src[0]);
        else if (src.length == 2) {
            double s0 = src[0];
            double s1 = src[1];
            return Math.sqrt(s0*s0 + s1*s1);
        } else if (src.length == 3) {
            double s0 = src[0];
            double s1 = src[1];
            double s2 = src[2];
            return Math.sqrt(s0*s0 + s1*s1 + s2*s2);
        }

        return scaledL2Norm(src);
    }


    /**
     * <p>Computes the squared Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm of a real dense or sparse
     * vector.
     * <p>Zeros do not contribute to this norm, so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero items if the vector is sparse) to compute norm of.
     * @return The squared Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm.
     */
    public static double normSquared(double... src) {
        // Some quick returns.
        if (src.length == 0) return 0;
        else if (src.length == 1) return src[0];
        else if (src.length == 2) {
            double s0 = src[0];
            double s1 = src[1];
            return s0*s0 + s1*s1;
        }
        else if (src.length == 2) {
            double s0 = src[0];
            double s1 = src[1];
            double s2 = src[2];
            return s0*s0 + s1*s1 + s2*s2;
        }

        return scaledL2NormSquared(src);
    }


    /**
     * <p>Computes the Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm of a dense or sparse vector whose entries are members of a
     * {@link RingElement}.
     * <p>Zeros do not contribute to this norm, so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero items if the vector is sparse) to compute norm of.
     * @return Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm
     */
    public static <T extends RingElement<T>> double norm(T... src) {
        // Some quick returns.
        if (src.length == 0) return 0;
        else if (src.length == 1) return src[0].abs();
        else if (src.length == 2) {
            double s0 = src[0].abs();
            double s1 = src[1].abs();
            return Math.sqrt(s0*s0 + s1*s1);
        }
        else if (src.length == 3) {
            double s0 = src[0].abs();
            double s1 = src[1].abs();
            double s2 = src[2].abs();
            return Math.sqrt(s0*s0 + s1*s1 + s2*s2);
        }

        return scaledL2Norm(src);
    }


    /**
     * <p>Computes the squared Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm of a real dense or sparse
     * vector.
     * <p>Zeros do not contribute to this norm, so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero items if the vector is sparse) to compute norm of.
     * @return The squared Euclidean (<span class="latex-inline">&ell;<sup>2</sup></span>) norm.
     */
    public static <T extends RingElement<T>> double normSquared(T... src) {
        // Some quick returns.
        if (src.length == 0) return 0;
        else if (src.length == 1) return src[0].abs();
        else if (src.length == 2) {
            double s0 = src[0].abs();
            double s1 = src[1].abs();
            return s0*s0 + s1*s1;
        }
        else if (src.length == 2) {
            double s0 = src[0].abs();
            double s1 = src[1].abs();
            double s2 = src[2].abs();
            return s0*s0 + s1*s1 + s2*s2;
        }

        return scaledL2NormSquared(src);
    }


    /**
     * <p>Computes the <span class="latex-inline">&ell;<sup>p</sup></span> norm (or p-norm) of a real dense or sparse vector.
     * <p>Some common norms:
     * <ul>
     *     <li>{@code p=1}: The taxicab, city block, or Manhattan norm.</li>
     *     <li>{@code p=2}: The Euclidean or <span class="latex-inline">&ell;<sup>2</sup></span> norm.</li>
     * </ul>
     *
     * <p>Zeros do not contribute to this norm, so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero items if the vector is sparse).
     * @param p The {@code p} value in the p-norm. When {@code p < 1}, the result of this method is not technically a
     * true mathematical norm. However, it may be useful for various numerical tasks.
     * <ul>
     *     <li>If {@code p} is finite, then the norm is computed as if by:
     *     <pre>{@code
     *     int norm = 0;
     *
     *     for(double v : src)
     *         norm += Math.pow(Math.abs(v), p);
     *
     *     return Math.pow(norm, 1.0/p);
     *     }</pre>
     *     </li>
     *     <li>If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.</li>
     *     <li>If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.</li>
     * </ul>
     *
     * <p>Warning, if {@code p} is very large in absolute value, overflow errors may occur.
     * @return The p-norm of the vector.
     */
    public static double norm(double[] src, double p) {
        if(p == 0.0) {
            throw new IllegalArgumentException("p must be non-zero.");
        }

        // Some quick returns.
        if (src.length == 0) return 0;
        else if (src.length == 1) return Math.abs(src[0]);

        if(p == Double.POSITIVE_INFINITY) {
            return RealProperties.maxAbs(src); // Maximum norm.
        } else if(p == Double.NEGATIVE_INFINITY) {
            return RealProperties.minAbs(src); // Minimum "norm".
        } else if(p == 1) {
            double norm = 0;
            for(double v : src)
                norm += Math.abs(v);

            return norm;
        } else if(p == 2) {
            return scaledL2Norm(src);
        } else {
            return (p > 0.0) ? scaledLpNorm(src, p) : scaledLpPseudoNorm(src, p);
        }
    }


    /**
     * <p>Computes the <span class="latex-inline">&ell;<sup>p</sup></span> norm (or p-norm) of a dense or sparse vector whose
     * entries are members of a {@link RingElement}.
     * <p>Some common norms:
     * <ul>
     *     <li>{@code p=1}: The taxicab, city block, or Manhattan norm.</li>
     *     <li>{@code p=2}: The Euclidean or <span class="latex-inline">&ell;<sup>2</sup></span> norm.</li>
     * </ul>
     *
     * <p>Zeros do not contribute to this norm, so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero items if the vector is sparse).
     * @param p The {@code p} value in the p-norm. When {@code p < 1}, the result of this method is not technically a
     * true mathematical norm. However, it may be useful for various numerical tasks.
     * <ul>
     *     <li>If {@code p} is finite, then the norm is computed as if by:
     *     <pre>{@code
     *     int norm = 0;
     *
     *     for(double v : src)
     *         norm += Math.pow(Math.abs(v), p);
     *
     *     return Math.pow(norm, 1.0/p);
     *     }</pre>
     *     </li>
     *     <li>If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.</li>
     *     <li>If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.</li>
     * </ul>
     *
     * <p>Warning, if {@code p} is very large in absolute value, overflow errors may occur.
     * @return The p-norm of the vector.
     */
    public static <T extends RingElement<T>> double norm(T[] src, double p) {
        if(p == 0.0) {
            throw new IllegalArgumentException("p must be non-zero.");
        }

        if (src.length == 0) return 0;
        else if (src.length == 1) return src[0].abs();

        if(p == Double.POSITIVE_INFINITY) {
            return CompareRing.maxAbs(src); // Maximum norm.
        } else if(p == Double.NEGATIVE_INFINITY) {
            return CompareRing.minAbs(src); // Minimum "norm".
        } else if(p == 1) {
            double norm = 0.0;
            for(T v : src)
                norm += v.abs();

            return norm;
        } else if(p == 2.0) {
            return scaledL2Norm(src);
        } else {
            return (p > 0.0) ? scaledLpNorm(src, p) : scaledLpPseudoNorm(src, p);
        }
    }


    // TODO: docs
    private static double scaledLpPseudoNorm(double[] src, double p) {
        if (p == 0.0 || Double.isNaN(p))
            throw new IllegalArgumentException("p must be non‑zero and finite");

        double q = -p;
        double scaleRec = 0.0;
        double sspRec  = 1.0;

        for(double v : src) {
            if(v == 0.0)
                return Double.POSITIVE_INFINITY;

            double y = 1.0 / v;

            if(y > scaleRec) {
                double t = scaleRec/y;
                sspRec = 1.0 + sspRec*Math.pow(t, q);
                scaleRec = y;
            } else {
                double t = y/scaleRec;
                sspRec += Math.pow(t, q);
            }
        }

        return 1.0 / (scaleRec*Math.pow(sspRec, 1.0/q));
    }


    /**
     * Computes the scaled <span class="latex-inline">&ell;<sup>p</sup></span> norm of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the <span class="latex-inline">&ell;<sup>p</sup></span> norm for large or small values of {@code p}.
     *
     * @param src The input vector (or non-zero values if the vector is sparse)
     * whose <span class="latex-inline">&ell;<sup>p</sup></span> norm is to be computed.
     * @param p The value of {@code p} for the <span class="latex-inline">&ell;<sup>p</sup></span> norm.
     * Here {@code p} must be positive.
     * @return The scaled <span class="latex-inline">&ell;<sup>p</sup></span> norm of the input vector.
     */
    private static double scaledLpNorm(double[] src, double p) {
        double scale = 0.0;  // running scale
        double ssp = 1.0;  // running scaled sum

        for (double v : src) {
            if (v == 0.0) continue;
            double ax = Math.abs(v);

            if (scale < ax) {
                double t = scale / ax;      // 0 < t < 1
                ssp = 1.0 + ssp*Math.pow(t, p);
                scale = ax;
            } else {
                double t = ax / scale;      // 0 <= t <= 1
                ssp += Math.pow(t, p);
            }
        }
        return (scale == 0.0) ? 0.0 : scale*Math.pow(ssp, 1.0 / p);
    }


    // TODO: Docs.
    private static <T extends RingElement<T>> double scaledLpPseudoNorm(T[] src, double p) {
        if (p == 0.0 || Double.isNaN(p))
            throw new IllegalArgumentException("p must be non‑zero and finite");

        double q = -p;
        double scaleRec = 0.0;
        double sspRec  = 1.0;

        for(T v : src) {
            if(v.isZero())
                return Double.POSITIVE_INFINITY;

            double y = 1.0/v.abs();

            if(y > scaleRec) {
                double t = scaleRec/y;
                sspRec = 1.0 + sspRec*Math.pow(t, q);
                scaleRec = y;
            } else {
                double t = y/scaleRec;
                sspRec += Math.pow(t, q);
            }
        }

        return 1.0 / (scaleRec*Math.pow(sspRec, 1.0/q));
    }



    /**
     * Computes the scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm (Euclidean norm) of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the <span class="latex-inline">&ell;<sup>2</sup></span> norm for vectors with very large or very small values.
     *
     * @param src The input vector (or non-zero entries if the vector is sparse) whose <span class="latex-inline">&ell;<sup>2</sup></span> norm is to be computed.
     * @return The scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm of the input vector.
     */
    private static double scaledL2Norm(double[] src) {
        double scale = 0.0;  // running scale
        double ssq = 1.0;  // running scaled sum of squares

        for (double v : src) {
            if (v == 0.0) continue;
            double ax = Math.abs(v);

            if (scale < ax) {
                double t = scale / ax;   // t < 1
                ssq = 1.0 + ssq*t*t;
                scale = ax;
            } else {
                double t = ax / scale;   // t <= 1
                ssq += t*t;
            }
        }

        return (scale == 0.0) ? 0.0 : scale*Math.sqrt(ssq);
    }


    /**
     * Computes the scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm (Euclidean norm) of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the <span class="latex-inline">&ell;<sup>2</sup></span> norm for vectors with very large or very small values.
     *
     * @param src The input vector (or non-zero entries if the vector is sparse)
     * whose <span class="latex-inline">&ell;<sup>2</sup></span> norm is to be computed.
     * @return The scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm of the input vector.
     */
    private static double scaledL2NormSquared(double[] src) {
        double scale = 0.0;   // running scale
        double ssq   = 1.0;  // running scaled sum of squares

        for (double v : src) {
            if (v == 0.0) continue;
            double ax = Math.abs(v);

            if (scale < ax) {
                double t = scale / ax;   // 0 < t < 1
                ssq = 1.0 + ssq*t*t;
                scale = ax;
            } else {
                double t = ax / scale;   // 0 <= t <= 1
                ssq += t*t;
            }
        }

        return (scale == 0.0) ? 0.0 : scale*scale*ssq;
    }


    /**
     * Computes the scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm (Euclidean norm) of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the <span class="latex-inline">&ell;<sup>2</sup></span> norm for vectors with very large or very small values.
     *
     * @param src The input vector (or non-zero entries if the vector is sparse)
     * whose <span class="latex-inline">&ell;<sup>2</sup></span> norm is to be computed.
     * @return The scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm of the input vector.
     */
    private static <T extends RingElement<T>> double scaledL2NormSquared(T[] src) {
        double scale = 0.0;   // running scale
        double ssq   = 1.0;  // running scaled sum of squares

        for (T v : src) {
            if (v.isZero()) continue;
            double ax = v.abs();

            if (scale < ax) {
                double t = scale / ax;   // 0 < t < 1
                ssq = 1.0 + ssq*t*t;
                scale = ax;
            } else {
                double t = ax / scale;   // 0 <= t <= 1
                ssq += t*t;
            }
        }

        return (scale == 0.0) ? 0.0 : scale*scale*ssq;
    }


    /**
     * Computes the scaled <span class="latex-inline">&ell;<sup>p</sup></span> norm of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the <span class="latex-inline">&ell;<sup>p</sup></span> norm for large or small values of {@code p}.
     *
     * @param src The input vector (or non-zero values if the vector is sparse)
     * whose <span class="latex-inline">&ell;<sup>p</sup></span> norm is to be computed.
     * @param p The value of {@code p} for the <span class="latex-inline">&ell;<sup>p</sup></span> norm.
     * @return The scaled <span class="latex-inline">&ell;<sup>p</sup></span> norm of the input vector.
     */
    private static <T extends RingElement<T>> double scaledLpNorm(T[] src, double p) {
        double scale = 0.0;  // running scale
        double ssp = 1.0;  // running scaled sum

        for (T v : src) {
            if (v.isZero()) continue;
            double ax = v.abs();

            if (scale < ax) {
                double t = scale / ax;      // 0 < t < 1
                ssp = 1.0 + ssp * Math.pow(t, p);
                scale = ax;
            } else {
                double t = ax / scale;      // 0 <= t <= 1
                ssp += Math.pow(t, p);
            }
        }

        return (scale == 0.0) ? 0.0 : scale*Math.pow(ssp, 1.0 / p);
    }


    /**
     * Computes the scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm (Euclidean norm) of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the <span class="latex-inline">&ell;<sup>2</sup></span> norm for vectors with very large or very small values.
     *
     * @param src The input vector (or non-zero entries if the vector is sparse) whose <span class="latex-inline">&ell;<sup>2</sup></span> norm is to be computed.
     * @return The scaled <span class="latex-inline">&ell;<sup>2</sup></span> norm of the input vector.
     */
    private static <T extends RingElement<T>> double scaledL2Norm(T[] src) {
        double scale = 0.0;  // running scale
        double ssq = 1.0;  // running scaled sum of squares

        for (T v : src) {
            if (v.isZero()) continue;
            double ax = v.abs();

            if (scale < ax) {
                double t = scale / ax;   // t < 1
                ssq = 1.0 + ssq*t*t;
                scale = ax;
            } else {
                double t = ax / scale;   // t <= 1
                ssq += t*t;
            }
        }

        return (scale == 0.0) ? 0.0 : scale*Math.sqrt(ssq);
    }


    /**
     * <p>Computes the <span class="latex-inline">&ell;<sup>2</sup></span> (Euclidean) norm of a sub-vector within {@code src},
     * starting at index {@code start} and considering {@code n} elements spaced by {@code stride}.
     *
     * <p>More formally, this method examines and computes the norm of the elements at indices:
     * {@code start}, {@code start + stride}, {@code start + 2*stride}, ..., {@code start + (n-1)*stride}.
     *
     * <p>This method may be used to compute the norm of a row or column in a
     * {@link org.flag4j.arrays.dense.Matrix matrix} {@code a} as follows:
     * <ul>
     *     <li>Norm of row {@code i}:
     *     <pre>{@code norm(a.items, i*a.numCols, a.numCols, 1);}</pre></li>
     *     <li>Norm of column {@code j}:
     *     <pre>{@code norm(a.items, j, a.numRows, a.numRows);}</pre></li>
     * </ul>
     *
     * @param src The array to containing sub-vector elements to compute norm of.
     * @param start The starting index in {@code src} to search. Must be positive but this is not explicitly enforced.
     * @param n The number of elements to consider within {@code src1}. Must be positive but this is not explicitly enforced.
     * @param stride The gap (in indices) between consecutive elements of the sub-vector within {@code src}.
     * Must be positive but this is not explicitly enforced.
     * @return The <span class="latex-inline">&ell;<sup>2</sup></span> (Euclidean) norm of the specified sub-vector of {@code src}.
     *
     * @throws IndexOutOfBoundsException If {@code start + (n-1)*stride} exceeds {@code src.length - 1}.
     */
    public static double norm(double[] src, final int start, final int n, final int stride) {
        if (src.length == 0) return 0;
        else if (src.length == 1) return Math.abs(src[0]);

        final int end = start + n*stride;
        double scale = 0.0;   // running scale
        double ssq   = 1.0;  // running scaled sum of squares

        for (int i=start; i<end; i+=stride) {
            double v = src[i];
            if (v == 0.0) continue;
            double ax = Math.abs(v);

            if (scale < ax) {
                double t = scale / ax;   // t < 1
                ssq = 1.0 + ssq*t*t;
                scale = ax;
            } else {
                double t = ax / scale;   // t <= 1
                ssq += t*t;
            }
        }

        return (scale == 0.0) ? 0.0 : scale * Math.sqrt(ssq);
    }


    /**
     * <p>Computes the <span class="latex-inline">&ell;<sup>2</sup></span> (Euclidean) norm of a sub-vector within {@code src},
     * starting at index {@code start} and considering {@code n} elements spaced by {@code stride}.
     *
     * <p>More formally, this method examines and computes the norm of the elements at indices:
     * {@code start}, {@code start + stride}, {@code start + 2*stride}, ..., {@code start + (n-1)*stride}.
     *
     * <p>This method may be used to compute the norm of a row or column in a
     * {@link org.flag4j.arrays.dense.Matrix matrix} {@code a} as follows:
     * <ul>
     *     <li>Norm of row {@code i}:
     *     <pre>{@code norm(a.items, i*a.numCols, a.numCols, 1);}</pre></li>
     *     <li>Norm of column {@code j}:
     *     <pre>{@code norm(a.items, j, a.numRows, a.numRows);}</pre></li>
     * </ul>
     *
     * @param src The array to containing sub-vector elements to compute norm of.
     * @param start The starting index in {@code src} to search. Must be positive but this is not explicitly enforced.
     * @param n The number of elements to consider within {@code src1}. Must be positive but this is not explicitly enforced.
     * @param stride The gap (in indices) between consecutive elements of the sub-vector within {@code src}.
     * Must be positive but this is not explicitly enforced.
     * @return The <span class="latex-inline">&ell;<sup>2</sup></span> (Euclidean) norm of the specified sub-vector of {@code src}.
     *
     * @throws IndexOutOfBoundsException If {@code start + (n-1)*stride} exceeds {@code src.length - 1}.
     */
    public static <T extends RingElement<T>> double norm(T[] src, final int start, final int n, final int stride) {
        if (src.length == 0) return 0;
        else if (src.length == 1) return src[0].abs();

        final int end = start + n*stride;
        double scale = 0.0;   // running scale
        double ssq   = 1.0;  // running scaled sum of squares

        for (int i=start; i<end; i+=stride) {
            T v = src[i];
            if (v.isZero()) continue;
            double ax = v.abs();

            if (scale < ax) {
                double t = scale / ax;   // 0 < t < 1
                ssq = 1.0 + ssq*t*t;
                scale = ax;
            } else {
                double t = ax / scale;   // 0 <= t <= 1
                ssq += t*t;
            }
        }

        return (scale == 0.0) ? 0.0 : scale * Math.sqrt(ssq);
    }
}
