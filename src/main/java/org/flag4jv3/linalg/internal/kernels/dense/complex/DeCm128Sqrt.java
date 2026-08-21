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
import org.flag4jv3.linalg.internal.kernels.DenseKernelSupport;
import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.Shape;
import org.flag4jv3.ndarrays.dense.DenseDoubleData;

import static org.flag4jv3.math.util.ComplexStability.SAFE_EXP_HI_128;
import static org.flag4jv3.math.util.ComplexStability.SAFE_EXP_LO_128;

// TODO NOW: FINISH DOCS, review comments

/// Low-level kernels for computing principal square roots of double-precision complex values and dense complex nD-arrays.
///
/// Complex nD-array values represented by an interleaved {@code double[]}
/// buffer:
/// ```
/// [re0, im0, re1, im1,...]
/// ```
///
/// Layout offsets and strides are measured in complex elements rather than raw positions in the interleaved buffer.
///
/// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
///     <strong>Warning:</strong> This class contains low-level implementations primarly intended for internal use.
///     As such, the methods in this class perform minimal validation of input parameters. Users of this class are responsible
///     for ensuring that the input data is valid and consistent with the intended operation. Malformed inputs <em>may</em>
///      result in undefined behavior, incorrect results, or exceptions.
/// </blockquote>
///
/// ### Stability & Safety Notes:
/// The complex square-root implementation in this class avoids intermediate overflow and
/// significant underflow by scaling values whose magnitudes fall outside a safe exponent range.
/// It also handles infinities, NaNs, signed zeros, and the negative-real-axis branch cut according to the
///  principal complex square root.
public final class DeCm128Sqrt {
    private DeCm128Sqrt() {
        // Hide default constructor for utility class.
    }


    // -------------------------------- nD-arrays --------------------------------


    /// Computes the element-wise principal square roots of a complex nD-array.
    ///
    /// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> No check is made to ensure that {@code srcLayout} is a valid layout for the {@code src} array.
    ///     Similarly for {@code outLayout} and {@code out}. If either of these are invalid, the behavior of this method is undefined.
    /// </blockquote>
    ///
    /// @param src The nD-array data buffer containing the complex values to compute the square roots of.
    /// The real and complex components of a complex number are interleaved in this array. That is, `src[2*i]` is the real component,
    /// and `src[2*i + 1]` is the complex component of the `i`-th complex number in the array.
    /// @param srcLayout The layout of the source array.
    /// @param out The destination buffer to store the result in. This *may* be the same object as `src` if and only if
    /// `!srcLayout.equals(outLayout)`.
    /// @param outLayout
    public static void csqrt(DenseDoubleData src, DenseDoubleData out) {
        DenseKernelSupport.requireWritable(out.layout());
        DenseKernelSupport.requireSameShape(out.layout(), src.layout());

        final var srcLayout = src.layout();
        final var srcBuffer = src.buffer();
        final var outLayout = out.layout();
        final var outBuffer = out.buffer();

        if (srcBuffer == outBuffer && !srcLayout.equals(outLayout)) {
            // TODO NOW: Similarly to resolveOut, add a resolveSrc that checks for this and makes temp copy of src
            // Allowing inplace computation with different layouts is not supported because it would
            // require copying the data to a temporary buffer to be done safely.
            throw new IllegalArgumentException(
                    "Source and destination share the same backing array "
                            + "but use different layouts.");
        }

        int count = srcLayout.shape().numelIntValueExact();

        if (count == 0) return;

        if (Layout.areContiguousAndMatchOrder(srcLayout, outLayout)) {
            csqrtContiguous(srcBuffer, srcLayout.offset(), outBuffer, outLayout.offset(), count);
        } else {
            csqrtStrided(srcBuffer, srcLayout, outBuffer, outLayout, count);
        }
    }


    private static void csqrtContiguous(double[] src,
                                        int srcOffset,
                                        double[] dest,
                                        int destOffset,
                                        int count) {
        int srcIndex = 2*srcOffset;
        int destIndex = 2*destOffset;

        for (int element = 0; element < count; element++) {
            csqrtInto(
                    src[srcIndex],
                    src[srcIndex + 1],
                    dest,
                    destIndex);

            srcIndex += 2;
            destIndex += 2;
        }
    }


    private static void csqrtStrided(double[] src,
                                     Layout srcLayout,
                                     double[] dest,
                                     Layout destLayout,
                                     int count) {
        final Shape shape = srcLayout.shape();
        final int rank = shape.rank();

        int srcElementIndex = srcLayout.offset();
        int destElementIndex = destLayout.offset();

        if (rank == 0) {
            csqrtInto(src[2*srcElementIndex], src[2*srcElementIndex + 1], dest, 2*destElementIndex);
            return;
        }

        final int[] srcStrides = srcLayout.strides();
        final int[] destStrides = destLayout.strides();
        final int[] indices = new int[rank];

        for (int element = 0; element < count; element++) {
            int srcIndex = 2*srcElementIndex;
            int destIndex = 2*destElementIndex;

            csqrtInto(src[srcIndex], src[srcIndex + 1], dest, destIndex);

            if (element + 1 == count) {
                break;
            }

            for (int axis = rank - 1; axis >= 0; axis--) {
                final int axisSize = shape.getSize(axis);

                indices[axis]++;

                if (indices[axis] < axisSize) {
                    srcElementIndex += srcStrides[axis];
                    destElementIndex += destStrides[axis];
                    break;
                }

                indices[axis] = 0;
                srcElementIndex -= (axisSize - 1)*srcStrides[axis];
                destElementIndex -= (axisSize - 1)*destStrides[axis];
            }
        }
    }


    private static void csqrtInto(double re, double im, double[] out, int destOffset) {
        // Triage for infinities and NaNs.
        if (!(Double.isFinite(re) && Double.isFinite(im))) {
            nonFiniteValues(re, im, out, destOffset);
            return;
        }

        // Pure-real fast path, including all signed-zero combinations.
        if (im == 0.0) {
            if (re > 0.0) {
                out[destOffset] = Math.sqrt(re);
                out[destOffset + 1] = im;
            } else if (re == 0.0) {
                out[destOffset] = 0.0;
                out[destOffset + 1] = im;
            } else {
                out[destOffset] = 0.0;
                out[destOffset + 1] = Math.copySign(Math.sqrt(-re), im);
            }

            return;
        }

        final double ax = Math.abs(re);
        final double ay = Math.abs(im);
        final int e = Math.getExponent(Math.max(ax, ay));

        // This is the larger component of the result:
        // Re(sqrt(z)) for re >= 0, or |Im(sqrt(z))| for re < 0.
        final double t = computeLargeComponent(ax, ay, e);

        if (re >= 0.0) {
            out[destOffset] = t;
            out[destOffset + 1] = im/(2.0*t);
        } else {
            out[destOffset] = ay/(2.0*t);
            out[destOffset + 1] = Math.copySign(t, im);
        }
    }


    private static void nonFiniteValues(
            double re,
            double im,
            double[] dest,
            int offset) {

        if (Double.isInfinite(im)) {
            dest[offset] = Double.POSITIVE_INFINITY;
            dest[offset + 1] = im;
            return;
        }

        if (Double.isNaN(re)) {
            dest[offset] = Double.NaN;
            dest[offset + 1] = Double.NaN;
            return;
        }

        if (Double.isInfinite(re)) {
            if (re > 0.0) {
                dest[offset] = Double.POSITIVE_INFINITY;
                dest[offset + 1] = Double.isNaN(im)
                        ? Double.NaN
                        : Math.copySign(0.0, im);
            } else if (Double.isNaN(im)) {
                // The sign of the infinite imaginary part is unspecified.
                dest[offset] = Double.NaN;
                dest[offset + 1] = Double.POSITIVE_INFINITY;
            } else {
                dest[offset] = 0.0;
                dest[offset + 1] = Math.copySign(Double.POSITIVE_INFINITY, im);
            }

            return;
        }

        // re is finite, so im must be NaN.
        dest[offset] = Double.NaN;
        dest[offset + 1] = Double.NaN;
    }

    // -------------------------------- Scalar values --------------------------------


    /**
     * Computes the principal square root of the complex number {@code re + im*i}.
     *
     * <p>The result {@code w} satisfies {@code Re(w) >= +0.0} (never {@code -0.0}),
     * lies on the closed right half-plane, and follows the C99 Annex G special-value
     * table for all infinity, NaN, and signed-zero inputs.
     *
     * @param re Real component of the value to take the square root of.
     * @param im Imaginary component of the value to take the square root of.
     * @return The principal square root {@code w} as {@code (Re(w), Im(w))}.
     */
    public static Complex128 csqrt(Complex128 z) {
        double re = z.re();
        double im = z.im();

        if (!(Double.isFinite(re) && Double.isFinite(im))) {
            return nonFiniteValues(re, im);
        }

        // Pure-real fast path. Also covers all four signed-zero-only inputs.
        // The general path below handles im == ±0.0 correctly too; this exists so
        // that real-valued data (scalar in a linear-algebra library) pays one sqrt,
        // and so sqrt(x + 0i) is bit-exact against Math.sqrt(x).
        if (im == 0.0) {
            if (re > 0.0) return new Complex128(Math.sqrt(re), im);
            if (re == 0.0) return new Complex128(0.0, im);          // sqrt(±0 ± 0i) = +0 ± 0i
            return new Complex128(0.0, Math.copySign(Math.sqrt(-re), im));
        }

        final double ax = Math.abs(re);
        final double ay = Math.abs(im);
        final int e = Math.getExponent(Math.max(ax, ay));

        // t == Re(sqrt(z)) if re >= 0, else |Im(sqrt(z))|.
        // Always a normal double for nonzero finite z: |z|^(1/2) lies in
        // [2^-537, 2^513), comfortably inside the normal range.
        final double t = computeLargeComponent(ax, ay, e);

        // Small component from the exact identity 2*Re(w)*Im(w) = im, evaluated
        // entirely in ORIGINAL coordinates: one correctly rounded division of
        // representable operands, subnormal only when the true result is.
        // 2t cannot overflow (t < 2^513) and cannot be zero.
        return (re >= 0.0)
                ? new Complex128(t, im/(2.0*t))
                : new Complex128(ay/(2.0*t), Math.copySign(t, im));
    }


    /// Handles Infinity/NaN triage per the C99 Annex G `csqrt` table (G.6.4.2).
    ///
    /// This method assumes that at least one of `re` or `im` is non-finite.
    ///
    /// @param re The real component of the complex number.
    /// @param im The imaginary component of the complex number.
    private static Complex128 nonFiniteValues(double re, double im) {
        // sqrt(x ± inf*i) = +inf ± inf*i for every x, including NaN.
        if (Double.isInfinite(im)) {
            return new Complex128(Double.POSITIVE_INFINITY, im);
        }

        // im is finite or NaN from here on.
        if (Double.isNaN(re)) {
            return new Complex128(Double.NaN, Double.NaN);
        }

        if (Double.isInfinite(re)) {
            if (re > 0.0) {
                // sqrt(+inf + y*i) = +inf + (±0)*i for finite y; +inf + NaN*i for NaN y.
                return Double.isNaN(im)
                        ? new Complex128(Double.POSITIVE_INFINITY, Double.NaN)
                        : new Complex128(Double.POSITIVE_INFINITY, Math.copySign(0.0, im));
            }
            // sqrt(-inf + y*i) = +0 + (±inf)*i for finite y; NaN ± inf*i for NaN y
            return Double.isNaN(im)
                    ? new Complex128(Double.NaN, Double.POSITIVE_INFINITY)
                    : new Complex128(0.0, Math.copySign(Double.POSITIVE_INFINITY, im));
        }

        // re finite, so im is NaN.
        return new Complex128(Double.NaN, Double.NaN);
    }


    private static double computeLargeComponent(double ax, double ay, int exponent) {
        if (exponent >= SAFE_EXP_LO_128 && exponent <= SAFE_EXP_HI_128) {
            double modulus = Math.sqrt(ax*ax + ay*ay);
            return Math.sqrt(0.5*(modulus + ax));
        }

        int k = exponent >> 1;
        double sx = Math.scalb(ax, -2*k);
        double sy = Math.scalb(ay, -2*k);
        double modulus = Math.sqrt(sx*sx + sy*sy);

        return Math.scalb(Math.sqrt(0.5*(modulus + sx)), k);
    }
}
