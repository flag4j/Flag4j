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

import org.flag4jv3.algebra.elements.SemiringElement;

import java.io.Serializable;
import java.math.BigInteger;

/// Defines the algebraic structure of a [semiring](https://en.wikipedia.org/wiki/Semiring).
///
/// A **semiring**, <span class="latex-inline">(R, +, &middot;, 0, 1)</span>, is a set <span class="latex-inline">R</span>
/// equipped with two binary operations: addition (<span class="latex-inline">+</span>) and multiplication
/// (<span class="latex-inline">&middot;</span>).
/// Semirings generalize [rings][Ring] by dropping the requirement that every element in
/// <span class="latex-inline">R</span> must have an additive inverse.
///
/// ### Semiring Axioms
/// For all <span class="latex-replace">a, b, c in R</span><!-- LATEX: $a, b, c \in R$ -->:
/// - __Additive identity exists__: There exists an element <span class="latex-inline">0&isin;R</span>
/// such that <span class="latex-inline">a + 0 = a</span>
/// - __Multiplicative identity exists__: There exists an element <span class="latex-inline">1&isin;R</span> such that
/// <span class="latex-inline">a &middot; 1 = a</span>
/// - __Addition is associative__: <span class="latex-inline">a + (b + c) = (a + b) + c</span>
/// - __Addition is commutative__: <span class="latex-inline">a + b = b + a</span>
/// - __Multiplication is associative__: <span class="latex-inline">a &middot; (b &middot; c) = (a &middot; b) &middot; c</span>
/// - __Multiplication is distributive over addition__:
///     - __Left distributivity__: <span class="latex-inline">a &middot; (b + c) = a &middot; b + a &middot; c</span>
///     - __Right distributivity__: <span class="latex-inline">(a + b) &middot; c = a &middot; c + b &middot; c</span>
/// - <span class="latex-inline">0</span> __is absorbing under multiplication__: <span class="latex-inline">a &middot; 0 = 0 &middot; a = 0</span>
///
/// ### Examples of Semirings
/// - The natural numbers <span class="latex-inline">ℕ</span> under ordinary addition and multiplication
/// - The booleans `true` and `false` under logical OR and AND as addition and multiplication respectively
/// - The tropical semiring: reals extended with +&infin; as additive identity, 0 as the multiplicative identity
/// and where "addition" is <span class="latex-inline">min</span> and "multiplication" is ordinary real addition
///
/// This interface represents the algebraic *structure* itself.
/// Each element of the semiring exposes a reference back to its
/// structure via [SemiringElement#structure()], so that [#zero()]/[#one()] are always
/// consistent across elements and reachable without an element instance in hand.
///
/// @param <T> the type of element belonging to this semiring.
/// @see SemiringElement
/// @see Ring
/// @see Field
/// @see FiniteField
public interface Semiring<T extends SemiringElement<T>> extends Serializable {

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
    abstract boolean equals(Object b);

    @Override
    abstract int hashCode();


    /// The canonical image of the natural number `n` in this semiring; that is, `n·1`
    /// (the sum of `n` copies of [#one()], with `valueOf(0)` yielding [#zero()]).
    ///
    /// This is the unique semiring homomorphism from the natural numbers into this semiring:
    /// `valueOf(a + b) = valueOf(a) + valueOf(b)` and `valueOf(a * b) = valueOf(a) * valueOf(b)`.
    /// Note that it need not be injective: in a structure of non-zero characteristic `p`,
    /// `valueOf(n)` is equal to `valueOf(n mod p)`.
    ///
    /// @throws IllegalArgumentException if `n < 0`. (Semirings need not contain additive inverses;
    /// see [Ring#valueOf(long)] which lifts this restriction.)
    /// @implSpec The default implementation performs double-and-add, using O(log n) additions.
    /// Implementations which admit a more direct construction (e.g. modular reduction) should override
    /// this method. Overriding implementations should override all `valueOf` overloads consistently.
    /// @see #valueOf(int)
    default T valueOf(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("Cannot map negative value into a semiring: " + n);
        }

        T result = zero();
        T powerOfTwo = one();

        while (n != 0) {
            if ((n & 1L) == 1L) result = result.add(powerOfTwo);
            n >>>= 1;
            if (n != 0) powerOfTwo = powerOfTwo.add(powerOfTwo);
        }

        return result;
    }


    /// The canonical image of the natural number `n` in this semiring; that is, `n·1`
    /// (the sum of `n` copies of [#one()], with `valueOf(0)` yielding [#zero()]).
    ///
    /// This is the unique semiring homomorphism from the natural numbers into this semiring:
    /// `valueOf(a + b) = valueOf(a) + valueOf(b)` and `valueOf(a * b) = valueOf(a) * valueOf(b)`.
    /// Note that it need not be injective: in a structure of non-zero characteristic `p`,
    /// `valueOf(n)` is equal to `valueOf(n mod p)`.
    ///
    /// @throws IllegalArgumentException if `n < 0`. (Semirings need not contain additive inverses;
    /// see [Ring#valueOf(BigInteger)] which lifts this restriction.)
    /// @implSpec The default implementation performs double-and-add, using O(log n) additions.
    /// Implementations which admit a more direct construction (e.g. modular reduction) should override
    /// this method. Overriding implementations should override all `valueOf` overloads consistently.
    /// @see #valueOf(long)
    default T valueOf(BigInteger n) {
        if (n.signum() < 0) {
            throw new IllegalArgumentException("Cannot map negative value into a semiring: " + n);
        }

        T result = zero();
        T powerOfTwo = one();
        final int len = n.bitLength();

        for (int i = 0; i < len; i++) {
            if (n.testBit(i)) result = result.add(powerOfTwo);
            if (i < len - 1) powerOfTwo = powerOfTwo.add(powerOfTwo);
        }

        return result;
    }
}
