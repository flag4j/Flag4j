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

import org.flag4jv3.algebra.FiniteField;

import java.math.BigInteger;
import java.util.List;


/// Represents a scalar element of a [FiniteField].
///
/// An element of a [FiniteField] is a value supporting all the axioms of a [FiniteField].
///
/// ### Implementation notes
/// - Implementations should be **immutable**: every operation returns a new instance
///   rather than mutating `this`.
/// - [#equals(Object)] and [#hashCode()] must be consistent with the field's notion
///   of equality, since [#isZero()] and [#isOne()] default to comparing against
///   [#zero()]/[#one()] via `equals`.
/// - [#structure()] should return the same [FiniteField] instance (or an [equals][#equals(Object)]-consistent
///   one) for every element that actually belongs to that field.
///
/// @param <T> The type of the field element.
/// @see FiniteField
/// @see SemiringElement
/// @see RingElement
/// @see FieldElement
public interface FiniteFieldElement<T extends FiniteFieldElement<T>> extends FieldElement<T> {
    @Override
    FiniteField<T> structure(); // Override for return type.

    /// Gets the [order][FiniteField#order()] of this finite field.
    ///
    /// @return The [order][FiniteField#order()] of this finite field.
    default BigInteger order() {
        return structure().order();
    }

    /// Gets the [degree][FiniteField#degree()] of this finite field.
    ///
    /// @return The [degree][FiniteField#degree()] of this finite field.
    default int degree() {
        return structure().degree();
    }


    /// The coefficients of the monic irreducible polynomial defining this scalar's field representation.
    ///
    /// The returned list satisfies the following:
    /// - Coefficients appear in ascending order of degree: index `i` holds the coefficient of
    ///   <span class="latex-inline">x<sup>i</sup></span>.
    /// - The list has length `degree() + 1` and its last element is [BigInteger#ONE] (the modulus is monic).
    /// - Every coefficient lies in `[0, p)` where `p` is this field's [characteristic][#characteristic()].
    /// - The list is immutable.
    ///
    /// For prime fields (i.e. `degree() == 1`) the modulus is conventionally
    /// <span class="latex-inline">m(x) = x</span>, so this method returns `[0, 1]`.
    ///
    /// @return An immutable list of the coefficients of this field's modulus, in ascending order of degree.
    ///
    default List<BigInteger> modulusCoefficients() {
        return structure().modulusCoefficients();
    }


    /// Gets a [primitive element][FiniteField#primitiveElement()] of the finite field this scalar belongs to.
    /// That is, a fixed generator of the field's multiplicative group: an element whose powers enumerate every
    /// non-zero element of the field.
    ///
    /// @return A primitive element of the finite field this scalar belongs to.
    ///
    /// @see FiniteField#primitiveElement()
    default T primitiveElement() {
        return structure().primitiveElement();
    }


    /// Applies the Frobenius map of this scalar's [finite field][#structure()] to `this`.
    /// Specifically, this is defined as `pow(characteristic())`.
    ///
    /// @return The result of applying the Frobenius map to `this`.
    default T frobenius() {
        return degree() == 1 ? self() : pow(characteristic());
    }


    /// Iteratively applies the [Frobenius map][#frobenius()] of this scalar's [finite field][#structure()] `n` times to `this`.
    ///
    /// @param n The number of times to apply the [Frobenius map][#frobenius()]. Must be non-negative.
    /// @return The result of applying the [Frobenius map][#frobenius()] `n` times to `this`.
    ///
    /// @throws IllegalArgumentException If `n` is negative.
    default T frobenius(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative.");
        }

        n %= degree();

        T result = self();
        for (int i = 0; i < n; i++) {
            result = result.frobenius();
        }

        return result;
    }
}
