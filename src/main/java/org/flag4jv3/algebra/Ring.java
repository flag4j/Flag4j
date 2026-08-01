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

import org.flag4jv3.algebra.elements.RingElement;

import java.math.BigInteger;


/// Defines the algebraic structure of a [ring](https://en.wikipedia.org/wiki/Ring_(mathematics)).
///
/// A **ring**, <span class="latex-inline">(R, +, &middot;, 0, 1)</span>, is a set <span class="latex-inline">R</span>
/// equipped with two binary operations: addition (<span class="latex-inline">+</span>) and multiplication
/// (<span class="latex-inline">&middot;</span>). Rings generalize [fields][Field] by dropping the requirement that every element in
/// <span class="latex-inline">R</span> must have a multiplicative inverse and that multiplication must commute.
/// Rings are also a special case of [semirings][Semiring] with the additional requirement that every element
/// in <span class="latex-inline">R</span> must have an additive inverse.
///
/// ### Ring Axioms
/// For all <span class="latex-inline">a, b, c &isin; R</span>:
/// - __Additive identity exists__: There exists an element <span class="latex-inline">0&isin;R</span>
/// such that <span class="latex-inline">a + 0 = a</span>
/// - __Multiplicative identity exists__: There exists an element <span class="latex-inline">1&isin;R</span> such that
/// <span class="latex-inline">a &middot; 1 = a</span>
/// - __Existence of additive inverses__: For every <span class="latex-inline">a&isin;R</span>, there exists an element
/// <span class="latex-inline">-a&isin;R</span> such that <span class="latex-inline">a + (-a) = 0</span>
/// - __Addition is associative__: <span class="latex-inline">a + (b + c) = (a + b) + c</span>
/// - __Addition is commutative__: <span class="latex-inline">a + b = b + a</span>
/// - __Multiplication is associative__: <span class="latex-inline">a &middot; (b &middot; c) = (a &middot; b) &middot; c</span>
/// - __Multiplication is distributive over addition__:
///     - __Left distributivity__: <span class="latex-inline">a &middot; (b + c) = a &middot; b + a &middot; c</span>
///     - __Right distributivity__: <span class="latex-inline">(a + b) &middot; c = a &middot; c + b &middot; c</span>
///
/// ### Examples of rings
/// - The integers <span class="latex-inline">ℤ</span> under ordinary addition and multiplication
/// - The set of polynomials with real coefficients
/// - The set of <span class="latex-replace">n-by-n</span><!-- LATEX: \(n\times n\) --> matrices.
/// - The set of quaternions <span class="latex-inline">ℍ</span>.
///
/// This interface represents the algebraic *structure* itself.
/// Each element of the ring exposes a reference back to its
/// structure via [RingElement#structure()], so that [#zero()]/[#one()] and the
/// [#characteristic()] are always consistent across elements and reachable without an element instance in hand.
///
/// @param <T> the type of element belonging to this ring.
/// @see RingElement
/// @see Semiring
/// @see Field
/// @see FiniteField
public interface Ring<T extends RingElement<T>> extends Semiring<T> {

    /// The characteristic of the ring.
    ///
    /// The characteristic of a ring is the smallest positive integer <span class="latex-replace">n</span> such that
    /// <span class="latex-replace">n&middot;1=0</span> if such an <span class="latex-replace">n</span> exists.
    /// Otherwise, the characteristic is <span class="latex-replace">0</span>.
    ///
    /// @return The characteristic of this ring.
    ///
    BigInteger characteristic();


    /// The canonical image of the integer `n` in this ring; that is, `n·1`
    /// (the sum of `n` copies of [#one()], with `valueOf(0)` yielding [#zero()]).
    ///
    /// This is the unique ring homomorphism from the integers into this ring:
    /// `valueOf(a + b) = valueOf(a) + valueOf(b)` and `valueOf(a * b) = valueOf(a) * valueOf(b)`.
    /// Note that it need not be injective: in a structure of non-zero characteristic `p`,
    /// `valueOf(n)` is equal to `valueOf(n mod p)`.
    ///
    /// @implSpec The default implementation performs double-and-add, using O(log n) additions.
    /// Implementations which admit a more direct construction (e.g., modular reduction) should override
    /// this method. Overriding implementations should override all `valueOf` overloads consistently.
    /// @see #valueOf(BigInteger)
    @Override
    default T valueOf(long n) {
        if (n >= 0) return Semiring.super.valueOf(n);
        if (n == Long.MIN_VALUE) { // -n would overflow; fallback to BigInteger.
            return Semiring.super.valueOf(Long.MAX_VALUE).add(one()).negate();
        }
        return Semiring.super.valueOf(-n).negate();
    }


    /// The canonical image of the integer `n` in this ring; that is, `n·1`
    /// (the sum of `n` copies of [#one()], with `valueOf(0)` yielding [#zero()]).
    ///
    /// This is the unique ring homomorphism from the integers into this ring:
    /// `valueOf(a + b) = valueOf(a) + valueOf(b)` and `valueOf(a * b) = valueOf(a) * valueOf(b)`.
    /// Note that it need not be injective: in a structure of non-zero characteristic `p`,
    /// `valueOf(n)` is equal to `valueOf(n mod p)`.
    ///
    /// @implSpec The default implementation performs double-and-add, using O(log n) additions.
    /// Implementations which admit a more direct construction (e.g., modular reduction) should override
    /// this method. Overriding implementations should override all `valueOf` overloads consistently.
    /// @see #valueOf(long)
    @Override
    default T valueOf(BigInteger n) {
        return n.signum() >= 0
                ? Semiring.super.valueOf(n)
                : Semiring.super.valueOf(n.negate()).negate();
    }
}
