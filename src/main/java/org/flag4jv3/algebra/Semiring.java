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

package org.flag4jv3.algebra;

import org.flag4jv3.scalars.SemiringScalar;

/// Defines the mathematical structure of a [semiring](https://en.wikipedia.org/wiki/Semiring).
///
/// A **semiring**, `(R, +, *, 0, 1)`, is a set `R` with addition (`+`) and multiplication (`*`), but no
/// requirement that additive inverses exist. Subtraction is therefore not generally
/// defined; this is what distinguishes a semiring from a [Ring].
///
/// ### Semiring Axioms
/// For all `a, b, c` in `R`:
/// - Addition is associative and commutative, with identity `0`: `a + 0 = a`
/// - Multiplication is associative, with identity `1`: `a * 1 = a`
/// - Multiplication distributes over addition on both sides
/// - `0` is absorbing under multiplication: `a * 0 = 0 * a = 0`
///
/// ### Examples
/// - The natural scalars `ℕ` under ordinary `+` and `*`
/// - The booleans `{false, true}` under logical OR and AND
/// - The tropical semiring: reals extended with `+∞`, where "addition" is `min` and
///   "multiplication" is ordinary real addition
///
/// This interface represents the algebraic *structure* itself.
/// Each element of the semiring exposes a reference back to its
/// structure via [SemiringScalar#structure()], so that `zero()`/`one()` are always
/// consistent across elements and reachable without an element instance in hand.
///
/// @param <T> the type of element belonging to this semiring.
/// @see SemiringScalar
public interface Semiring<T extends SemiringScalar<T>> {
    /// The additive identity element `0` of this semiring satisfying `a + 0 = a`
    /// for every element `a` in this semiring.
    ///
    /// @return The additive identity of this semiring.
    T zero();

    /// The multiplicative identity element `1` of this semiring satisfying `a * 1 = a`
    /// for every element `a` in this semiring.
    ///
    /// @return The multiplicative identity of this semiring.
    T one();


    /// Checks if this semiring is equal to another object.
    ///
    /// Two semirings are considered equal if they represent the same algebraic structure.
    ///
    /// @param b The other object to compare with this semiring.
    /// @return `true` if the other object is a semiring and represents the same algebraic
    ///                 structure as this semiring; `false` otherwise.
    @Override
    boolean equals(Object b);

    @Override
    int hashCode();
}
