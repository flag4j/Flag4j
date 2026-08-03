/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
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

package org.flag4jv3.linalg.internal.kernels.dense.complex;

import org.flag4jv3.algebra.elements.Complex128;
import org.flag4jv3.math.util.ComplexStability;


/// Low-level kernels for computing exponential, powers, and logarithms of double-precision complex values and dense complex nD-arrays.
///
/// Complex nD-array values represented by an interleaved {@code double[]}
/// buffer:
/// ```
/// [re0, im0, re1, im1,...]
/// ```
///
/// Layout offsets and strides are measured in complex elements rather than raw positions in the interleaved buffer.
///
/// <blockquote style="color: #c29d9d; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
///     <strong>Warning:</strong> This class contains low-level implementations primarly intended for internal use.
///     As such, the methods in this class perform minimal validation of input parameters. Users of this class are responsible
///     for ensuring that the input data is valid and consistent with the intended operation. Malformed inputs <em>may</em>
///     result in undefined behavior, incorrect results, or exceptions.
/// </blockquote>
///
/// ### Stability & Safety Notes:
/// The implementations in this class generally avoids intermediate overflow and
/// significant underflow by scaling values whose magnitudes fall outside a safe ranges.
/// It also handles infinities, NaNs, and signed zeros where possible.
public class DeCm128Exponential {

    private DeCm128Exponential() {
        // Hide default constructor for utility class.
    }


    // -------------------------------- nD-arrays --------------------------------


    // -------------------------------- Scalar values --------------------------------


    /// Evaluates the [complex exponential](https://en.wikipedia.org/wiki/Exponential_function#Complex_exponential) function:
    /// <span class="latex-inline">exp(z) = <em>e</em><sup>z</sup></span>.
    ///
    /// For <span class="latex-inline">z = x + iy</span>, this is computed as
    /// <span class="latex-replace">exp(z) = e<sup>x</sup>(cos(y) + i&middot;sin(y))</span>
    ///
    /// Internal scaling is used so that a finite result is returned whenever one is representable, even if
    /// <span class="latex-inline">e<sup>x</sup></span> alone would overflow. A result which is genuinely too large to
    /// represent will have one or both components infinite.
    ///
    /// **Special Cases:**
    /// - If `z` is real, the result is `Math.exp(z.re())` with the sign of the zero imaginary
    /// component preserved.
    /// - If `z.im()` is infinite or [NaN][Double#NaN], the result is [Complex128#NaN] unless `z.re()` is infinite.
    /// - All remaining special cases follow C99 Annex G (&sect;G.6.3.1).
    ///
    /// @param z The complex value to evaluate the exponential function on.
    /// @return The value <span class="latex-inline"><em>e</em><sup>z</sup></span>.
    public static Complex128 cexp(Complex128 z) {
        double re = z.re();
        double im = z.im();

        if (im == 0.0) {
            return new Complex128(Math.exp(re), im);
        }

        if (!Double.isFinite(im)) {
            if (re == Double.NEGATIVE_INFINITY) {
                return new Complex128(0.0, Math.copySign(0.0, im));
            }

            if (re == Double.POSITIVE_INFINITY) {
                return new Complex128(Double.POSITIVE_INFINITY, Double.NaN);
            }

            return Complex128.NaN; // Finite or NaN real component with non-finite im.
        }

        double cos = Math.cos(im);
        double sin = Math.sin(im);

        if (re == 0.0) {
            return new Complex128(cos, sin);
        }

        if (re <= ComplexStability.LOG_MAX_128 || !Double.isFinite(re)) {
            double scale = Math.exp(re);
            return new Complex128(scale*cos, scale*sin);
        }

        int k = (int) Math.floor(re*ComplexStability.INV_LN_2_128);
        double remainder = Math.fma(-k, ComplexStability.LN_2_128, re);
        double scale = Math.exp(remainder);

        return new Complex128(
                Math.scalb(scale*cos, k),
                Math.scalb(scale*sin, k)
        );
    }
}
