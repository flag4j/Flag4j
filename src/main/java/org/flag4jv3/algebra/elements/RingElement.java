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


import org.flag4jv3.algebra.Ring;

import java.math.BigInteger;

/// Represents a scalar element of a [Ring].
///
/// An element of a [Ring] is a value supporting all the axioms of a [Ring].
///
/// ### Implementation notes
/// - Implementations should be **immutable**: every operation returns a new instance
///   rather than mutating `this`.
/// - [#equals(Object)] and [#hashCode()] must be consistent with the ring's notion
///   of equality, since [#isZero()] and [#isOne()] default to comparing against
///   [#zero()]/[#one()] via `equals`.
/// - [#structure()] should return the same [Ring] instance (or an [equals][#equals(Object)]-consistent
///   one) for every element that actually belongs to that ring.
///
/// @param <T> The type of the ring element.
/// @see org.flag4jv3.algebra.Ring
/// @see SemiringElement
/// @see FieldElement
/// @see FiniteFieldElement
public interface RingElement<T extends RingElement<T>> extends SemiringElement<T> {

    @Override
    Ring<T> structure(); // Override for return type.

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
    @Override // Override to support negative values.
    default T mult(long n) {
        if (n >= 0) {
            return SemiringElement.super.mult(n);
        } else if (n == Long.MIN_VALUE) { // -n would overflow.
            T neg = negate();
            return neg.mult(Long.MAX_VALUE).add(neg);
        } else {
            return negate().mult(-n);
        }
    }


    /// Gets the [characteristic][Ring#characteristic()] of the ring this scalar belongs to.
    ///
    /// @return The characteristic of the ring this scalar belongs to.
    default BigInteger characteristic() {
        return structure().characteristic();
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
    default T sub(T b) {
        return add(b.negate());
    }


    /// Negates `this.` That is, computes the additive inverse of `this` scalar.
    ///
    /// @return The negation of `this`.
    ///
    /// @see #sub(RingElement)
    T negate();
}
