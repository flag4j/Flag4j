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

import org.flag4jv3.algebra.Semiring;

/// Represents an element of a [Semiring].
///
/// An element of a [Semiring] is a value supporting associative/commutative addition and
/// associative multiplication that distributes over addition, with additive/multiplicative identities `0` and `1`.
/// Semirings do not guarantee additive inverses, so subtraction is not defined here; see
/// [RingScalar] for that.
///
/// ### Implementation notes
/// - Implementations should be **immutable**: every operation returns a new instance
///   rather than mutating `this`.
/// - [#equals(Object)] and [#hashCode()] must be consistent with the semiring's notion
///   of equality, since [#isZero()] and [#isOne()] default to comparing against
///   [#zero()]/[#one()] via `equals`.
/// - [#structure()] should return the same [Semiring] instance (or an `equals`-consistent
///   one) for every element that actually belongs to that semiring.
///
/// @param <T> the concrete element type.
/// @see Semiring
/// @see RingScalar
public interface SemiringScalar<T extends SemiringScalar<T>> {
    Semiring<T> structure();

    /// Gets a reference to `this` object.
    ///
    /// @return A reference to `this` object.
    @SuppressWarnings("unchecked")
    default T self() {
        return (T) this;
    }


    /// Adds `b` to `this`.
    ///
    /// @param b The element to add.
    /// @return The sum of `this` and `b`.
    T add(T b);

    /// Multiplies `this` by `b`.
    ///
    /// @param b The element to multiply by.
    /// @return The product of `this` and `b`.
    T mult(T b);


    /// The additive identity of the [Semiring] this element belongs to (i.e., zero).
    ///
    /// @return The additive identity of the [Semiring] this element belongs to.
    ///
    /// @see #isZero()
    /// @see #one()
    default T zero() {
        return structure().zero();
    }


    /// The multiplicative identity of the [Semiring] this element belongs to (i.e., one).
    ///
    /// @return The multiplicative identity of the [Semiring] this element belongs to.
    ///
    /// @see #isOne()
    /// @see #zero()
    default T one() {
        return structure().one();
    }


    /// Checks if this semiring element is the [additive identity][#zero()].
    ///
    /// @return `true` if `this` is equal to [#zero()]; `false` otherwise.
    ///
    /// @see #zero()
    /// @see #isOne()
    default boolean isZero() {
        return equals(zero());
    }


    /// Checks if this semiring element is the [multiplicative identity][#one()].
    ///
    /// @return `true` if `this` is equal to [#one()]; `false` otherwise.
    ///
    /// @see #one()
    /// @see #isZero()
    default boolean isOne() {
        return equals(one());
    }

    /// Computes `this` raised to the power `power`.
    ///
    /// @param power The power to raise `this` to. Must be non-negative.
    /// @return `this` raised to the specified `power`.
    ///
    /// @see #square()
    default T pow(int power) {
        return DefaultHelpers.pow(self(), power);
    }


    /// Computes the square of `this`.
    ///
    /// @return The square of `this`.
    ///
    /// @see #pow(power)
    default T square() {
        return mult(self());
    }


    /// Checks if another object is equal to `this` semiring element.
    ///
    /// @param b The other object to compare to `this` semiring element.
    /// @return `true` if `b` is a semiring element and is equal to `this`; `false` otherwise.
    @Override
    boolean equals(Object b);


    @Override
    int hashCode();
}
