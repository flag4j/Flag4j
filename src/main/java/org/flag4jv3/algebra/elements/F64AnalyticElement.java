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

// TODO NOW: Docs

/// Scalars which represent a real number with 64-bit floating point precision or an extension of real numbers
/// with 64-bit floating point precision should implement this class.
///
/// For example, a complex number represented internally as two 64-bit floating points (for real and imaginary
/// components) should implement this interface.
public interface F64AnalyticElement<T extends F64AnalyticElement<T>> extends AnalyticElement<T> {

    @Override
    default T sqrt() {
        return pow(0.5);
    }

    default T root(double n) {
        if (n == 0.0) {
            throw new ArithmeticException("Cannot take the zeroth root.");
        }

        return pow(1.0/n);
    }

    T log(double base);

    @Override
    default T log10() {
        return ln().div(Math.log(10));
    }

    T pow(double a);

    T add(double a);

    T sub(double a);

    T mult(double b);

    T div(double a);

    /// (<i>e<sup>x</sup>&nbsp;-&nbsp;e<sup>-x</sup></i>)/2
    @Override
    default T sinh() {
        return exp().sub(negate().exp()).div(2.0);
    }

    /// (<i>e<sup>x</sup>&nbsp;+&nbsp;e<sup>-x</sup></i>)/2
    @Override
    default T cosh() {
        return exp().add(negate().exp()).div(2.0);
    }
}
