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

package org.flag4jv3.scalars;

/// Scalars which are analytic and so functions like `exp` and the trig functions can be defined generally
/// for the field as the locally convergent Taylor series for the respective function. Since polynomials only rely
/// on operations defined for the field, this gives a valid definition for these functions for a general field.
public interface AnalyticScalar<T extends AnalyticScalar<T>> extends FieldScalar<T> {

    default double abs() {
        return mag();
    }

    double mag();

    T sqrt();

    default T root(T n) {
        return pow(n);
    }

    T exp();

    T ln();

    default T log(T base) {
        return ln().div(base.ln());
    }

    T log10();

    T log2();

    T log1p();

    default T pow(T exponent) {
        return exponent.mult(this.ln()).exp();
    }

    // TODO NOW: We can default some of these (e.g., tan(x) = sin(x)/cos(x))
    T sin();

    T cos();

    default T tan() {
        return sin().div(cos());
    }

    default T cot() {
        return cos().div(sin());
    }

    default T sec() {
        return cos().inv();
    }

    default T csc() {
        return sin().inv();
    }

    T asin();

    T acos();

    T atan();

    T asec();

    T acsc();


    /// (<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/2
    T sinh();

    /// (<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)/2
    T cosh();

    /// (<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/(<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)
    default T tanh() {
        T exp = exp();
        T negExp = exp.negate();

        return exp.sub(negExp).div(exp.add(negExp));
    }


    default T sech() {
        return cosh().inv();
    }


    default T csch() {
        return sinh().inv();
    }


    default T coth() {
        T exp = exp();
        T negExp = exp.negate();

        return exp.add(negExp).div(exp.sub(negExp));
    }
}
