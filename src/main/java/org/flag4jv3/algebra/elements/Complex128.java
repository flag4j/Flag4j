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

package org.flag4jv3.algebra.elements;

import org.flag4jv3.algebra.Field;
import org.flag4jv3.io.parsing.ComplexNumberParser;
import org.flag4jv3.linalg.internal.kernels.dense.complex.DeCm128Exponential;
import org.flag4jv3.linalg.internal.kernels.dense.complex.DeCm128Sqrt;
import org.flag4jv3.util.tuples.DoublePair;
import org.flag4jv3.util.tuples.IntPair;

import java.math.BigInteger;

// TODO Valhalla: when relevant features of Valhalla are completed, this should become a value record.
// TODO CodeGen: Tag this class to generate Complex64

/// Represents a complex number with a real and imaginary part.
///
/// The real and imaginary parts are represented as doubles (i.e., 63-bit floating point numbers). Hence, this implementation is
/// referred to as a 128-bit complex number.
///
public record Complex128(double re, double im) implements F64AnalyticElement<Complex128> {

    // TODO NOW: What other constants do we want?
    public static final Complex128 ZERO = new Complex128(0);
    public static final Complex128 ONE = new Complex128(1);
    public static final Complex128 I = new Complex128(0, 1);
    public static final Complex128 INF = new Complex128(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
    public static final Complex128 NEG_INF = new Complex128(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
    public static final Complex128 NaN = new Complex128(Double.NaN, Double.NaN);


    public Complex128(double re) {
        this(re, 0);
    }


    public Complex128() {
        this(0, 0);
    }


    public Complex128(DoublePair value) {
        this(value.first(), value.second());
    }


    public Complex128(IntPair value) {
        this(value.first(), value.second());
    }


    public Complex128(String value) {
        this(ComplexNumberParser.getComponents(value));
    }


    /// Checks if another object is equal to this complex number.
    ///
    /// An object is considered equal to a complex number if it is an instance of [Complex128]
    /// and the real and imaginary parts are equal to this complex number according to the semantics of
    /// [Double#equals(Object)] and *not* according to
    public boolean equals(Object other) {
        if (this == other) return true;
        if (other instanceof Complex128 that) {
            if (that.isNaN()) {
                return true;
            } else {
                return Double.valueOf(that.re).equals(re)
                        && Double.valueOf(that.im).equals(im);
            }
        }

        return false;
    }


    @Override
    public Complex128 sqrt() {
        return DeCm128Sqrt.csqrt(re, im);
    }


    @Override
    public Complex128 root(double n) {
        return F64AnalyticElement.super.root(n);
    }


    @Override
    public Complex128 log(double base) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex128 log10() {
        return F64AnalyticElement.super.log10();
    }


    @Override
    public Complex128 pow(double a) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex128 add(double a) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex128 sub(double a) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex128 mult(double b) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex128 div(double a) {
        // TODO NOW: Implement this method
        return null;
    }


    /// (<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/2
    @Override
    public Complex128 sinh() {
        return F64AnalyticElement.super.sinh();
    }


    /// (<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)/2
    @Override
    public Complex128 cosh() {
        return F64AnalyticElement.super.cosh();
    }


    /// Computes the magnitude of `this` scalar.
    ///
    /// @return The magnitude value of `this` scalar.
    ///
    /// @see #magSquared()
    @Override
    public double mag() {
        return Math.hypot(re, im);
    }


    @Override
    public double magSquared() {
        return re*re + im*im;
    }


    /// Computes the principal n<sup>th</sup> root of `this` scalar.
    ///
    /// Note: this need not be defined or be unique for all elements of the field.
    ///
    /// @param n
    /// @return The principal n<sup>th</sup> root of `this` scalar if defined.
    ///
    /// @throws ArithmeticException If the n<sup>th</sup> root is not defined for `this` scalar.
    @Override
    public Complex128 root(Complex128 n) {
        return F64AnalyticElement.super.root(n);
    }


    /// Evaluates the exponential function on `this` scalar.
    ///
    /// @return The output of the exponential function on `this` scalar.
    ///
    /// @see #ln()
    /// @see #pow(Complex128)
    @Override
    public Complex128 exp() {
        return DeCm128Exponential.cexp(this);
    }


    /// Evaluates the natural logarithm on `this` scalar.
    ///
    /// @return The output of the natural logarithm on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #exp()
    /// @see #log2()
    /// @see #log10()
    /// @see #log(Complex128)
    @Override
    public Complex128 ln() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the natural logarithm on `this` scalar plus [1][#one()]: `add(one()).ln()`.
    ///
    /// @return The output of the natural logarithm on `this` scalar plus [1][#one()].
    ///
    /// @see #ln()
    /// @see #exp()
    /// @see #log2()
    /// @see #log10()
    /// @see #log(Complex128)
    @Override
    public Complex128 ln1p() {
        return F64AnalyticElement.super.ln1p();
    }


    /// Evaluates the logarithm with the specified `base` on `this` scalar.
    ///
    /// @param base The base of the logarithm.
    /// @return The output of logarithm with the specified `base` on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #log2()
    /// @see #log10()
    /// @see #ln
    /// @see #pow(Complex128)
    @Override
    public Complex128 log(Complex128 base) {
        return F64AnalyticElement.super.log(base);
    }


    /// Evaluates the logarithm base2 on `this` scalar.
    ///
    /// @return The output of logarithm base 2 on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #log10()
    /// @see #log(Complex128)
    /// @see #ln
    @Override
    public Complex128 log2() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes `this` scalar raised to the specified `exponent`.
    ///
    /// @param exponent The exponent to raise `this` scalar to.
    /// @return The output of raising `this` scalar to the specified `exponent`.
    ///
    /// @see #exp()
    /// @see #log(Complex128)
    @Override
    public Complex128 pow(Complex128 exponent) {
        return F64AnalyticElement.super.pow(exponent);
    }


    /// Evaluates the sine function, <span class="latex-inline">sin</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">sin</span> of `this` scalar.
    @Override
    public Complex128 sin() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the cosine function, <span class="latex-inline">cos</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">cos</span> of `this` scalar.
    @Override
    public Complex128 cos() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the tangent function, <span class="latex-inline">tan</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">tan</span> of `this` scalar.
    @Override
    public Complex128 tan() {
        return F64AnalyticElement.super.tan();
    }


    /// Evaluates the cotangent function, <span class="latex-inline">cot</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">cot</span> of `this` scalar.
    @Override
    public Complex128 cot() {
        return F64AnalyticElement.super.cot();
    }


    /// Evaluates the secant function, <span class="latex-inline">sec</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">sec</span> of `this` scalar.
    @Override
    public Complex128 sec() {
        return F64AnalyticElement.super.sec();
    }


    /// Evaluates the cosecant function, <span class="latex-inline">csc</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">csc</span> of `this` scalar.
    @Override
    public Complex128 csc() {
        return F64AnalyticElement.super.csc();
    }


    /// Evaluates the arc sine (inverse sine) function, <span class="latex-inline">asin</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">asin</span> of `this` scalar.
    @Override
    public Complex128 asin() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc cosine (inverse cosine) function, <span class="latex-inline">acos</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acos</span> of `this` scalar.
    @Override
    public Complex128 acos() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc tangent (inverse tangent) function, <span class="latex-inline">atan</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">atan</span> of `this` scalar.
    @Override
    public Complex128 atan() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc cotangent (inverse cotangent) function, <span class="latex-inline">acot</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acot</span> of `this` scalar.
    @Override
    public Complex128 acot() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc secant (inverse secant) function, <span class="latex-inline">asec</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">asec</span> of `this` scalar.
    @Override
    public Complex128 asec() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc cosecant (inverse cosecant) function, <span class="latex-inline">acsc</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acsc</span> of `this` scalar.
    @Override
    public Complex128 acsc() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the hyperbolic tangent, <span class="latex-inline">tanh</span> on `this` scalar.
    /// This is defined as:
    /// <span class="latex-eq-align">
    /// <pre>
    ///     tanh(x) = 1/coth(x)
    ///             = sinh(x)/cosh(x)
    ///             = (<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/(<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)</pre>
    /// </span>
    ///
    /// @return <span class="latex-inline">tanh</span> of `this` scalar.
    @Override
    public Complex128 tanh() {
        return F64AnalyticElement.super.tanh();
    }


    /// Evaluates the hyperbolic secant, <span class="latex-inline">sech</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">1/cosh(x)</span>.
    ///
    /// @return <span class="latex-inline">sech</span> of `this` scalar.
    ///
    /// @see #cosh()
    @Override
    public Complex128 sech() {
        return F64AnalyticElement.super.sech();
    }


    /// Evaluates the hyperbolic cosecant, <span class="latex-inline">csch</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">1/sinh(x)</span>.
    ///
    /// @return <span class="latex-inline">csch</span> of `this` scalar.
    ///
    /// @see #sinh()
    @Override
    public Complex128 csch() {
        return F64AnalyticElement.super.csch();
    }


    /// Evaluates the hyperbolic cotangent, <span class="latex-inline">coth</span> on `this` scalar.
    /// This is defined as:
    /// <span class="latex-eq-align">
    /// <pre>
    ///     coth(x) = 1/tanh(x)
    ///             = cosh(x)/sinh(x)
    ///             = (<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)/(<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)</pre>
    /// </span>
    ///
    /// @return <span class="latex-inline">coth</span> of `this` scalar.
    ///
    /// @see #tanh()
    @Override
    public Complex128 coth() {
        return F64AnalyticElement.super.coth();
    }


    /// Evaluates the signum function on `this` scalar.
    ///
    /// @return The signum function evaluated on `this` scalar.
    @Override
    public Complex128 sgn() {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Field<Complex128> structure() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Divides `this` by `b`.
    ///
    /// @param b the divisor. Must not be zero.
    /// @return The quotient of `this` and `b`.
    ///
    /// @throws ArithmeticException If `b` is [zero][#isZero()].
    /// @implSpec By default, this is implemented as `mult(b.inv())`. Because of this, if an implementation of
    /// [#inv()] relies on [this method][#div(Complex128)], then this method must *also* be overridden.
    /// Otherwise, there would be infinite recursion.
    /// @see #inv()
    @Override
    public Complex128 div(Complex128 b) {
        return F64AnalyticElement.super.div(b);
    }


    /// Multiplies `this` scalar by the long `n`.
    /// This is defined as a repeated addition of this scalar with itself `n` times if `n` is non-negative.
    /// If `n` is negative, then the result is the [additive inverse][#negate] of the product of
    /// `this` and `|n|`.
    ///
    /// @param n The long to multiply by.
    /// @return The product of `this` and `n`. If `n==0`, then [#zero()] is returned.
    ///
    /// @implNote The default implementation relies on [SemiringElement#mult(long)],
    /// which is a "double and add" algorithm that is <span class="latex-inline">O(log(n))</span>.
    /// Implementations should generally override this method for better performance.
    @Override
    public Complex128 mult(long n) {
        return F64AnalyticElement.super.mult(n);
    }


    /// Gets the [characteristic][#characteristic()] of the ring this scalar belongs to.
    ///
    /// @return The characteristic of the ring this scalar belongs to.
    @Override
    public BigInteger characteristic() {
        return F64AnalyticElement.super.characteristic();
    }


    /// Subtracts `b` from `this`.
    ///
    /// @param b The value to subtract from `this`.
    /// @return The difference of `this` and `b`.
    ///
    /// @implSpec By default, this is implemented as `add(b.negate())`. Because of this, if an implementation of
    /// [#negate()] relies on [this method][#div(Complex128)], then this method must *also* be overridden.
    /// Otherwise, there would be infinite recursion.
    /// @see #negate()
    @Override
    public Complex128 sub(Complex128 b) {
        return F64AnalyticElement.super.sub(b);
    }


    /// Adds `b` to `this`.
    ///
    /// @param b The element to add.
    /// @return The sum of `this` and `b`.
    @Override
    public Complex128 add(Complex128 b) {
        // TODO NOW: Implement this method
        return null;
    }


    /// Multiplies `this` by `b`.
    ///
    /// @param b The element to multiply by.
    /// @return The product of `this` and `b`.
    @Override
    public Complex128 mult(Complex128 b) {
        // TODO NOW: Implement this method
        return null;
    }


    /// Checks if this semiring element is the [additive identity][#zero()].
    ///
    /// @return `true` if `this` is equal to [#zero()]; `false` otherwise.
    ///
    /// @see #zero()
    /// @see #isOne()
    @Override
    public boolean isZero() {
        return F64AnalyticElement.super.isZero();
    }


    /// Checks if this semiring element is the [multiplicative identity][#one()].
    ///
    /// @return `true` if `this` is equal to [#one()]; `false` otherwise.
    ///
    /// @see #one()
    /// @see #isZero()
    @Override
    public boolean isOne() {
        return F64AnalyticElement.super.isOne();
    }


    /// Computes the square of `this`.
    ///
    /// @return The square of `this`.
    ///
    /// @see #pow(long)
    @Override
    public Complex128 square() {
        return F64AnalyticElement.super.square();
    }


    /// Negates `this.` That is, computes the additive inverse of `this` scalar.
    ///
    /// @return The negation of `this`.
    ///
    /// @see #sub(Complex128)
    @Override
    public Complex128 negate() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes the multiplicative inverse of `this`.
    ///
    /// @return The multiplicative inverse of `this`.
    ///
    /// @throws ArithmeticException If `this` is [zero][#isZero()].
    /// @see #div(Complex128)
    @Override
    public Complex128 inv() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes `this` raised to the power `n`. If `n` is positive, this is defined as repeated multiplication of
    /// `this` scalar with itself `n` times. If `n` is negative, then the result is the [multaplicative inverse][#inv]
    /// of the positive case.
    ///
    /// @param n The exponent.
    /// @return `this` raised to the specified `n`.
    ///
    /// @implNote The default implementation uses a repeated square and multiply algorithm
    /// that is <span class="latex-inline">O(log(n))</span>.
    @Override
    public Complex128 pow(long n) {
        return F64AnalyticElement.super.pow(n);
    }


    /// Computes `this` raised to the power `n`. If `n` is positive, this is defined as repeated multiplication of
    /// `this` scalar with itself `n` times. If `n` is negative, then the result is the [multaplicitive inverse][#inv]
    /// of the positive case.
    ///
    /// @param n The exponent.
    /// @return `this` raised to the specified `n`.
    ///
    /// @implNote The default implementation uses a repeated square and multiply algorithm
    /// that is <span class="latex-inline">O(log(n))</span>.
    @Override
    public Complex128 pow(BigInteger n) {
        return F64AnalyticElement.super.pow(n);
    }


    /// Computes the conjugate value of `this` scalar.
    ///
    /// @return The conjugate value of `this` scalar.
    ///
    /// @implSpec The implementation must satisfy:
    /// 1. __Involutive__: <span class="latex-inline">(a<sup>\*</sup>)<sup>\*</sup> = a</span>
    /// 2. __Additive__: <span class="latex-inline">(a + b)<sup>\*</sup> = a<sup>\*</sup> + b<sup>\*</sup></span>
    /// 3. __Anti-multiplicative__: <span class="latex-inline">(ab)<sup>\*</sup> = b<sup>\*</sup>a<sup>\*</sup></span>
    ///
    @Override
    public Complex128 conjugate() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes the norm of `this` scalar. This is defined as `mult(conjugate())`.
    ///
    /// @return The norm of `this` scalar.
    @Override
    public Complex128 norm() {
        return F64AnalyticElement.super.norm();
    }


    /// Checks if this complex number is NaN. A complex number is NaN if *either* the real *or* imaginary part is NaN.
    ///
    /// @return `true` if this complex number is NaN; `false` otherwise.
    @Override
    public boolean isNaN() {
        return Double.isNaN(re) || Double.isNaN(im);
    }


    /// Checks if this complex number is finite. A complex number is finite if *both* the real *and* imaginary part is
    /// finite.
    ///
    /// @return `true` if this complex number is finite; `false` otherwise.
    @Override
    public boolean isFinite() {
        return Double.isFinite(re) && Double.isFinite(im);
    }


    /// Checks if this complex number is infinite. A complex number is infinite if *either* the real *or* imaginary part is
    /// infinite.
    ///
    /// @return `true` if this complex number is infinite; `false` otherwise.
    @Override
    public boolean isInfinite() {
        return Double.isInfinite(re) || Double.isInfinite(im);
    }


    /// Converts this complex number to a human-readable string.
    ///
    /// @return A human-readable string representation of this complex number.
    @Override
    public String toString() {
        // Try for some quick returns.
        if (isNaN()) return "NaN";
        if (isInfinite()) return "Infinity";
        if (isZero()) return "0";

        String realPart = "";
        String imagPart = "";
        String sign;
        double imAbs = Math.abs(im);

        if (re != 0.0) {
            realPart = re%1 == 0 ? String.valueOf((int) re) : String.valueOf(re);
            sign = im > 0.0 ? "+" : "-";
        } else {
            sign = im > 0.0 ? "" : "-";
        }

        if (imAbs == 1.0) {
            imagPart = sign + (im < 0.0 ? "i" : "i");
        } else if (im != 0.0) {
            imagPart = sign + (im%1 == 0 ? String.valueOf((int) imAbs) : String.valueOf(imAbs)) + "i";
        }

        return realPart + imagPart;
    }
}
