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

/// Represents a [scalar][FieldElement] belonging to a [field][org.flag4jv3.algebra.Field] that is analytic.
/// That is, it belongs to a field <span class="latex-inline">F</span> such that locally convergent Taylor series can be defined for
/// some subset of functions mapping <span class="latex-inline">F</span> to <span class="latex-inline">F</span>.
/// For instance, elements from fields that *are* or *behave like* the real or complex numbers.
///
/// This allows for functions like [exp][#exp()] and [sin][#sin()]/[cos][#cos()] to be defined on the field.
///
/// @see FieldElement
/// @see org.flag4jv3.algebra.Field
public interface AnalyticElement<T extends AnalyticElement<T>> extends FieldElement<T>, InvolutiveRingElement<T> {


    /// Computes the magnitude of `this` scalar.
    ///
    /// @return The magnitude value of `this` scalar.
    ///
    /// @see #magSquared()
    double mag();


    /// Computes the square of the magnitude of `this` scalar.
    ///
    /// @return The square of the magnitude of `this` scalar.
    ///
    /// @implSpec The default implementation is essentially `mag()*mag()`. However, many elements may find computing the
    /// squared magnitude is actually more efficient than the standard magnitude. As such, it is *highly* encouraged to
    /// override this in such cases.
    /// @see #mag()
    default double magSquared() {
        double mag = mag();
        return mag*mag;
    }


    /// Computes the principal square root of `this` scalar.
    ///
    /// Note: this need not be defined or be unique for all elements of the field.
    ///
    /// @return The principal square root of `this` scalar if defined.
    ///
    /// @throws ArithmeticException If the square root is not defined for `this` scalar.
    T sqrt();


    /// Computes the principal n<sup>th</sup> root of `this` scalar.
    ///
    /// Note: this need not be defined or be unique for all elements of the field.
    ///
    /// @return The principal n<sup>th</sup> root of `this` scalar if defined.
    ///
    /// @throws ArithmeticException If the n<sup>th</sup> root is not defined for `this` scalar.
    default T root(T n) {
        return pow(n.inv());
    }


    /// Evaluates the exponential function on `this` scalar.
    ///
    /// @return The output of the exponential function on `this` scalar.
    ///
    /// @see #ln()
    /// @see #pow(AnalyticElement)
    T exp();


    /// Evaluates the natural logarithm on `this` scalar.
    ///
    /// @return The output of the natural logarithm on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #exp()
    /// @see #log2()
    /// @see #log10()
    /// @see #log(AnalyticElement)
    T ln();


    /// Evaluates the natural logarithm on `this` scalar plus [1][#one()]: `add(one()).ln()`.
    ///
    /// @return The output of the natural logarithm on `this` scalar plus [1][#one()].
    ///
    /// @see #ln()
    /// @see #exp()
    /// @see #log2()
    /// @see #log10()
    /// @see #log(AnalyticElement)
    default T ln1p() {
        return add(one()).ln();
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
    default T log(T base) {
        return ln().div(base.ln());
    }


    /// Evaluates the logarithm base 10 on `this` scalar.
    ///
    /// @return The output of logarithm base 10 on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #log2()
    /// @see #log(AnalyticElement)
    /// @see #ln
    T log10();


    /// Evaluates the logarithm base2 on `this` scalar.
    ///
    /// @return The output of logarithm base 2 on `this` scalar.
    ///
    /// @see #ln1p()
    /// @see #log10()
    /// @see #log(AnalyticElement)
    /// @see #ln
    T log2();


    /// Computes `this` scalar raised to the specified `exponent`.
    ///
    /// @param exponent The exponent to raise `this` scalar to.
    /// @return The output of raising `this` scalar to the specified `exponent`.
    ///
    /// @see #exp()
    /// @see #log(AnalyticElement)
    default T pow(T exponent) {
        return exponent.mult(this.ln()).exp();
    }


    /// Evaluates the sine function, <span class="latex-inline">sin</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">sin</span> of `this` scalar.
    T sin();


    /// Evaluates the cosine function, <span class="latex-inline">cos</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">cos</span> of `this` scalar.
    T cos();


    /// Evaluates the tangent function, <span class="latex-inline">tan</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">tan</span> of `this` scalar.
    default T tan() {
        return sin().div(cos());
    }


    /// Evaluates the cotangent function, <span class="latex-inline">cot</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">cot</span> of `this` scalar.
    default T cot() {
        return cos().div(sin());
    }


    /// Evaluates the secant function, <span class="latex-inline">sec</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">sec</span> of `this` scalar.
    default T sec() {
        return cos().inv();
    }


    /// Evaluates the cosecant function, <span class="latex-inline">csc</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">csc</span> of `this` scalar.
    default T csc() {
        return sin().inv();
    }


    /// Evaluates the arc sine (inverse sine) function, <span class="latex-inline">asin</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">asin</span> of `this` scalar.
    T asin();


    /// Evaluates the arc cosine (inverse cosine) function, <span class="latex-inline">acos</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acos</span> of `this` scalar.
    T acos();


    /// Evaluates the arc tangent (inverse tangent) function, <span class="latex-inline">atan</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">atan</span> of `this` scalar.
    T atan();


    /// Evaluates the arc cotangent (inverse cotangent) function, <span class="latex-inline">acot</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acot</span> of `this` scalar.
    T acot();


    /// Evaluates the arc secant (inverse secant) function, <span class="latex-inline">asec</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">asec</span> of `this` scalar.
    T asec();


    /// Evaluates the arc cosecant (inverse cosecant) function, <span class="latex-inline">acsc</span>, on `this` scalar.
    ///
    /// @return <span class="latex-inline">acsc</span> of `this` scalar.
    T acsc();


    /// Evaluates the hyperbolic sine, <span class="latex-inline">sinh</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">(<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/2</span>
    ///
    /// @return <span class="latex-inline">sinh</span> of `this` scalar.
    T sinh();


    /// Evaluates the hyperbolic cosine, <span class="latex-inline">cosh</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">(<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)/2</span>.
    ///
    /// @return <span class="latex-inline">cosh</span> of `this` scalar.
    T cosh();


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
    default T tanh() {
        T exp = exp();
        T negExp = exp.inv();

        return exp.sub(negExp).div(exp.add(negExp));
    }


    /// Evaluates the hyperbolic secant, <span class="latex-inline">sech</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">1/cosh(x)</span>.
    ///
    /// @return <span class="latex-inline">sech</span> of `this` scalar.
    ///
    /// @see #cosh()
    default T sech() {
        return cosh().inv();
    }


    /// Evaluates the hyperbolic cosecant, <span class="latex-inline">csch</span> on `this` scalar.
    /// This is defined as: <span class="latex-inline">1/sinh(x)</span>.
    ///
    /// @return <span class="latex-inline">csch</span> of `this` scalar.
    ///
    /// @see #sinh()
    default T csch() {
        return sinh().inv();
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
    default T coth() {
        T exp = exp();
        T negExp = exp.inv();

        return exp.add(negExp).div(exp.sub(negExp));
    }


    /// Evaluates the signum function on `this` scalar.
    ///
    /// @return The signum function evaluated on `this` scalar.
    T sgn();


    /// Checks if `this` is finite.
    ///
    /// @return `true` if `this` is finite; otherwise, `false`.
    ///
    /// @implNote By default, this is implemented as `!isInfinite() && !isNaN()`
    default boolean isFinite() {
        return !isInfinite() && !isNaN();
    }


    /// Checks if `this` is infinite.
    ///
    /// @return `true` if `this` is infinite; otherwise, `false`.
    ///
    /// @implNote By default, this is implemented by checking if this element's [squared magnitude][#magSquared()]
    /// is infinite according [Double#isNaN(double)]. However, for some fields, it may be useful to define this directly
    /// on the field element and not rely on the squared magnitude.
    default boolean isInfinite() {
        return Double.isInfinite(magSquared());
    }


    /// Checks if `this` is NaN.
    ///
    /// @return `true` if `this` is NaN; otherwise, `false`.
    ///
    /// @implNote By default, this is implemented by checking if this element's [squared magnitude][#magSquared()] is NaN according to
    /// [Double#isNaN(double)]. However, for some fields, it may be useful to define this directly on the field element and not
    /// rely on the squared magnitude.
    default boolean isNaN() {
        return Double.isNaN(magSquared());
    }
}
