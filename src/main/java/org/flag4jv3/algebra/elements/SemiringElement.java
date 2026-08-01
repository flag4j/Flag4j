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

import org.flag4jv3.algebra.Semiring;

import java.io.Serializable;
import java.math.BigInteger;

/// Represents a scalar element of a [Semiring].
///
/// An element of a [Semiring] is a value supporting all the axioms of a [Semiring].
///
/// ### Implementation notes
/// - Implementations should be **immutable**: every operation returns a new instance
///   rather than mutating `this`.
/// - [#equals(Object)] and [#hashCode()] must be consistent with the semiring's notion
///   of equality, since [#isZero()] and [#isOne()] default to comparing against
///   [#zero()]/[#one()] via `equals`.
/// - [#structure()] should return the same [Semiring] instance (or an [equals][#equals(Object)]-consistent
///   one) for every element that actually belongs to that semiring.
///
/// @param <T> The type of the semiring element.
/// @see Semiring
/// @see RingElement
/// @see FieldElement
/// @see FiniteFieldElement
public interface SemiringElement<T extends SemiringElement<T>> extends Serializable {

    /// Gets the backing algebraic structure of `this` scalar.
    ///
    /// @return The backing algebraic structure of `this` scalar.
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


    /// Multiplies `this` scalar by the long `n`. This is defined as repeated addition of this scalar with itself `n` times.
    ///
    /// @param n The non-negative long to multiply by.
    /// @return The product of `this` and `n`. If `n==0`, then [#zero()] is returned.
    ///
    /// @throws IllegalArgumentException If `n` is negative.
    /// @implNote The default implementation is a repeated addition algorithm that is <span class="latex-inline">O(log(n))</span>.
    /// Implementations should generally override this method for better performance.
    default T mult(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative.");
        }

        // Use a double and add algorithm to compute the result in O(log n) time.
        T result = zero();
        T addend = self();

        while (n != 0) {
            if ((n & 1) != 0) {
                result = result.add(addend);
            }

            n >>>= 1;

            if (n != 0) {
                addend = addend.add(addend);
            }
        }

        return result;
    }


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

    /// Computes `this` raised to the power `n`.
    ///
    /// @param n The exponent. Must be non-negative.
    /// @return `this` raised to the specified `n`.
    ///
    /// @throws IllegalArgumentException If `n` is negative.
    /// @implNote The default implementation uses a repeated square and multiply algorithm
    /// that is <span class="latex-inline">O(log(n))</span>.
    /// @see #square()
    default T pow(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative.");
        } else if (n == 0) {
            return one();
        }

        T result = one();
        T base = self();

        // Exponentiation by squaring.
        while (n > 0) {
            if ((n & 1) == 1) {
                result = result.mult(base);
            }

            base = base.mult(base);
            n >>= 1;
        }

        return result;
    }


    /// Computes `this` raised to the power `n`.
    ///
    /// @param n The exponent. Must be non-negative.
    /// @return `this` raised to the specified `n`.
    ///
    /// @throws IllegalArgumentException If `n` is negative.
    /// @implNote The default implementation uses a repeated square and multiply algorithm
    /// that is <span class="latex-inline">O(log(n))</span>.
    /// @see #square()
    default T pow(BigInteger n) {
        if (n.signum() < 0) {
            throw new IllegalArgumentException("n must be non-negative.");
        } else if (n.bitLength() <= 63) {
            return pow(n.longValue()); // Safe for 64-bit integers. Also covers the 0 case.
        }

        T result = one();
        T base = self();

        // Exponentiation by squaring.
        while (n.signum() > 0) {
            if (n.testBit(0)) {
                result = result.mult(base);
            }

            base = base.mult(base);
            n = n.shiftRight(1);
        }

        return result;
    }


    /// Computes the square of `this`.
    ///
    /// @return The square of `this`.
    ///
    /// @see #pow(long)
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
