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

/// Scalars that behave similarly to real numbers should implement this interface.
///
/// For example, complex numbers behave similarly to reals in many ways.
public interface AnalyticScalar<T extends AnalyticScalar<T>> extends FieldElement<T> {

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

    default T pow(T exponent) {
        return exponent.mult(this.ln()).exp();
    }


    T sin();

    T cos();

    T tan();

    T asin();

    T acos();

    // TODO NOW: Can add defults to these.
    T atan();

    T sinh();

    T cosh();

    T tanh();
}
