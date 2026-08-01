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

import java.math.BigInteger;


/// Represents a scalar element of a [Field].
///
/// An element of a [Field] is a value supporting all the axioms of a [Field].
///
/// ### Implementation notes
/// - Implementations should be **immutable**: every operation returns a new instance
///   rather than mutating `this`.
/// - [#equals(Object)] and [#hashCode()] must be consistent with the field's notion
///   of equality, since [#isZero()] and [#isOne()] default to comparing against
///   [#zero()]/[#one()] via `equals`.
/// - [#structure()] should return the same [Field] instance (or an [equals][#equals(Object)]-consistent
///   one) for every element that actually belongs to that field.
///
/// @param <T> The type of the field element.
/// @see Field
/// @see SemiringElement
/// @see RingElement
/// @see FiniteFieldElement
public interface FieldElement<T extends FieldElement<T>> extends RingElement<T> {

    @Override
    Field<T> structure(); // Override for return type.

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
    default T div(T b) {
        if (b.isZero()) {
            throw new ArithmeticException("Division by zero element.");
        }

        return mult(b.inv());
    }


    /// Computes the multiplicative inverse of `this`.
    ///
    /// @return The multiplicative inverse of `this`.
    ///
    /// @throws ArithmeticException If `this` is [zero][#isZero()].
    /// @see #div(FieldElement)
    T inv();


    /// Computes `this` raised to the power `n`. If `n` is positive, this is defined as repeated multiplication of
    /// `this` scalar with itself `n` times. If `n` is negative, then the result is the [multaplicative inverse][#inv]
    /// of the positive case.
    ///
    /// @param n The exponent.
    /// @return `this` raised to the specified `n`.
    ///
    /// @implNote The default implementation uses a repeated square and multiply algorithm
    /// that is <span class="latex-inline">O(log(n))</span>.
    @Override // Overriding here to support negative exponents for fields.
    default T pow(long n) {
        if (n == Long.MIN_VALUE) { // -n would overflow; fallback to BigInteger.
            return pow(BigInteger.valueOf(n));
        }

        return (n < 0)
                ? RingElement.super.pow(-n).inv()
                : RingElement.super.pow(n);
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
    @Override // Overriding here to support negative exponents for fields.
    default T pow(BigInteger n) {
        return (n.signum() == -1)
                ? RingElement.super.pow(n.negate()).inv()
                : RingElement.super.pow(n);
    }
}
