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
import org.flag4jv3.util.tuples.FloatPair;

import java.math.BigInteger;

public final record Complex64(float re, float im) implements F32AnalyticElement<Complex64> {

    public static final Complex64 ZERO = new Complex64(0f, 0f);
    public static final Complex64 ONE = new Complex64(1f, 0f);
    public static final Complex64 I = new Complex64(0f, 1f);
    public static final Complex64 INF = new Complex64(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
    public static final Complex64 NEG_INF = new Complex64(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
    public static final Complex64 NaN = new Complex64(Float.NaN, Float.NaN);


    public Complex64() {
        this(0f, 0f);
    }


    public Complex64(float re) {
        this(re, 0f);
    }


    public Complex64(FloatPair value) {
        this(value.first(), value.second());
    }


    public Complex64(String value) {
        this(ComplexNumberParser.getComponentsAsFloats(value));
    }


    @Override
    public Complex64 sqrt() {
        return F32AnalyticElement.super.sqrt();
    }


    @Override
    public Complex64 root(float n) {
        return F32AnalyticElement.super.root(n);
    }


    @Override
    public Complex64 log(float base) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex64 log10() {
        return F32AnalyticElement.super.log10();
    }


    @Override
    public Complex64 pow(float a) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex64 add(float a) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex64 sub(float a) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex64 mult(float b) {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Complex64 div(float a) {
        // TODO NOW: Implement this method
        return null;
    }


    /// (<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/2
    @Override
    public Complex64 sinh() {
        return F32AnalyticElement.super.sinh();
    }


    /// (<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)/2
    @Override
    public Complex64 cosh() {
        return F32AnalyticElement.super.cosh();
    }


    /// Computes the magnitude of `this` scalar.
    ///
    /// @return The magnitude value of `this` scalar.
    ///
    /// @see #magSquared()
    @Override
    public double mag() {
        // TODO NOW: Implement this method
        return 0;
    }


    @Override
    public double magSquared() {
        // TODO NOW: Implement this method
        return 0;
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
    public Complex64 root(Complex64 n) {
        return F32AnalyticElement.super.root(n);
    }


    /// Evaluates the exponential function on `this` scalar.
    ///
    /// @return The output of the exponential function on `this` scalar.
    ///
    /// @see #ln()
    /// @see #pow(AnalyticElement)
    @Override
    public Complex64 exp() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the natural logarithm on `this` scalar.
    ///
    /// @return The output of the natural logarithm on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #exp()
    /// @see #log2()
    /// @see #log10()
    /// @see #log(AnalyticElement)
    @Override
    public Complex64 ln() {
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
    /// @see #log(AnalyticElement)
    @Override
    public Complex64 ln1p() {
        return F32AnalyticElement.super.ln1p();
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
    /// @see #pow(AnalyticElement)
    @Override
    public Complex64 log(Complex64 base) {
        return F32AnalyticElement.super.log(base);
    }


    /// Evaluates the logarithm base2 on `this` scalar.
    ///
    /// @return The output of logarithm base 2 on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #log10()
    /// @see #log(AnalyticElement)
    /// @see #ln
    @Override
    public Complex64 log2() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes `this` scalar raised to the specified `exponent`.
    ///
    /// @param exponent The exponent to raise `this` scalar to.
    /// @return The output of raising `this` scalar to the specified `exponent`.
    ///
    /// @see #exp()
    /// @see #log(AnalyticElement)
    @Override
    public Complex64 pow(Complex64 exponent) {
        return F32AnalyticElement.super.pow(exponent);
    }


    /// Evaluates the sine function, <span class="latex-inline">sin</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">sin</span> of `this` scalar.
    @Override
    public Complex64 sin() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the cosine function, <span class="latex-inline">cos</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">cos</span> of `this` scalar.
    @Override
    public Complex64 cos() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the tangent function, <span class="latex-inline">tan</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">tan</span> of `this` scalar.
    @Override
    public Complex64 tan() {
        return F32AnalyticElement.super.tan();
    }


    /// Evaluates the cotangent function, <span class="latex-inline">cot</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">cot</span> of `this` scalar.
    @Override
    public Complex64 cot() {
        return F32AnalyticElement.super.cot();
    }


    /// Evaluates the secant function, <span class="latex-inline">sec</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">sec</span> of `this` scalar.
    @Override
    public Complex64 sec() {
        return F32AnalyticElement.super.sec();
    }


    /// Evaluates the cosecant function, <span class="latex-inline">csc</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">csc</span> of `this` scalar.
    @Override
    public Complex64 csc() {
        return F32AnalyticElement.super.csc();
    }


    /// Evaluates the arc sine (inverse sine) function, <span class="latex-inline">asin</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">asin</span> of `this` scalar.
    @Override
    public Complex64 asin() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc cosine (inverse cosine) function, <span class="latex-inline">acos</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acos</span> of `this` scalar.
    @Override
    public Complex64 acos() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc tangent (inverse tangent) function, <span class="latex-inline">atan</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">atan</span> of `this` scalar.
    @Override
    public Complex64 atan() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc cotangent (inverse cotangent) function, <span class="latex-inline">acot</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acot</span> of `this` scalar.
    @Override
    public Complex64 acot() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc secant (inverse secant) function, <span class="latex-inline">asec</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">asec</span> of `this` scalar.
    @Override
    public Complex64 asec() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Evaluates the arc cosecant (inverse cosecant) function, <span class="latex-inline">acsc</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acsc</span> of `this` scalar.
    @Override
    public Complex64 acsc() {
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
    public Complex64 tanh() {
        return F32AnalyticElement.super.tanh();
    }


    /// Evaluates the hyperbolic secant, <span class="latex-inline">sech</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">1/cosh(x)</span>.
    ///
    /// @return <span class="latex-inline">sech</span> of `this` scalar.
    ///
    /// @see #cosh()
    @Override
    public Complex64 sech() {
        return F32AnalyticElement.super.sech();
    }


    /// Evaluates the hyperbolic cosecant, <span class="latex-inline">csch</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">1/sinh(x)</span>.
    ///
    /// @return <span class="latex-inline">csch</span> of `this` scalar.
    ///
    /// @see #sinh()
    @Override
    public Complex64 csch() {
        return F32AnalyticElement.super.csch();
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
    public Complex64 coth() {
        return F32AnalyticElement.super.coth();
    }


    /// Evaluates the signum function on `this` scalar.
    ///
    /// @return The signum function evaluated on `this` scalar.
    @Override
    public Complex64 sgn() {
        // TODO NOW: Implement this method
        return null;
    }


    @Override
    public Field<Complex64> structure() {
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
    /// [#inv()] relies on [this method][#div(T)], then this method must *also* be overridden.
    /// Otherwise, there would be infinite recursion.
    /// @see #inv()
    @Override
    public Complex64 div(Complex64 b) {
        return F32AnalyticElement.super.div(b);
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
    public Complex64 mult(long n) {
        return F32AnalyticElement.super.mult(n);
    }


    /// Gets the [characteristic][Ring#characteristic()] of the ring this scalar belongs to.
    ///
    /// @return The characteristic of the ring this scalar belongs to.
    @Override
    public BigInteger characteristic() {
        return F32AnalyticElement.super.characteristic();
    }


    /// Subtracts `b` from `this`.
    ///
    /// @param b The value to subtract from `this`.
    /// @return The difference of `this` and `b`.
    ///
    /// @implSpec By default, this is implemented as `add(b.negate())`. Because of this, if an implementation of
    /// [#negate()] relies on [this method][#div(T)], then this method must *also* be overridden.
    /// Otherwise, there would be infinite recursion.
    /// @see #negate()
    @Override
    public Complex64 sub(Complex64 b) {
        return F32AnalyticElement.super.sub(b);
    }


    /// Gets a reference to `this` object.
    ///
    /// @return A reference to `this` object.
    @Override
    public Complex64 self() {
        return this;
    }


    /// Adds `b` to `this`.
    ///
    /// @param b The element to add.
    /// @return The sum of `this` and `b`.
    @Override
    public Complex64 add(Complex64 b) {
        // TODO NOW: Implement this method
        return null;
    }


    /// Multiplies `this` by `b`.
    ///
    /// @param b The element to multiply by.
    /// @return The product of `this` and `b`.
    @Override
    public Complex64 mult(Complex64 b) {
        // TODO NOW: Implement this method
        return null;
    }


    /// The additive identity of the [Semiring] this element belongs to (i.e., zero).
    ///
    /// @return The additive identity of the [Semiring] this element belongs to.
    ///
    /// @see #isZero()
    /// @see #one()
    @Override
    public Complex64 zero() {
        return F32AnalyticElement.super.zero();
    }


    /// The multiplicative identity of the [Semiring] this element belongs to (i.e., one).
    ///
    /// @return The multiplicative identity of the [Semiring] this element belongs to.
    ///
    /// @see #isOne()
    /// @see #zero()
    @Override
    public Complex64 one() {
        return F32AnalyticElement.super.one();
    }


    /// Checks if this semiring element is the [additive identity][#zero()].
    ///
    /// @return `true` if `this` is equal to [#zero()]; `false` otherwise.
    ///
    /// @see #zero()
    /// @see #isOne()
    @Override
    public boolean isZero() {
        return F32AnalyticElement.super.isZero();
    }


    /// Checks if this semiring element is the [multiplicative identity][#one()].
    ///
    /// @return `true` if `this` is equal to [#one()]; `false` otherwise.
    ///
    /// @see #one()
    /// @see #isZero()
    @Override
    public boolean isOne() {
        return F32AnalyticElement.super.isOne();
    }


    /// Computes the square of `this`.
    ///
    /// @return The square of `this`.
    ///
    /// @see #pow(long)
    @Override
    public Complex64 square() {
        return F32AnalyticElement.super.square();
    }


    /// Checks if another object is equal to `this` semiring element.
    ///
    /// @param b The other object to compare to `this` semiring element.
    /// @return `true` if `b` is a semiring element and is equal to `this`; `false` otherwise.
    @Override
    public boolean equals(Object b) {
        // TODO NOW: Implement this method
        return false;
    }


    @Override
    public int hashCode() {
        // TODO NOW: Implement this method
        return 0;
    }


    /// Negates `this.` That is, computes the additive inverse of `this` scalar.
    ///
    /// @return The negation of `this`.
    ///
    /// @see #sub(RingElement)
    @Override
    public Complex64 negate() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes the multiplicative inverse of `this`.
    ///
    /// @return The multiplicative inverse of `this`.
    ///
    /// @throws ArithmeticException If `this` is [zero][#isZero()].
    /// @see #div(FieldElement)
    @Override
    public Complex64 inv() {
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
    public Complex64 pow(long n) {
        return F32AnalyticElement.super.pow(n);
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
    public Complex64 pow(BigInteger n) {
        return F32AnalyticElement.super.pow(n);
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
    public Complex64 conjugate() {
        // TODO NOW: Implement this method
        return null;
    }


    /// Computes the norm of `this` scalar. This is defined as `mult(conjugate())`.
    ///
    /// @return The norm of `this` scalar.
    @Override
    public Complex64 norm() {
        return F32AnalyticElement.super.norm();
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
        float imAbs = Math.abs(im);

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
