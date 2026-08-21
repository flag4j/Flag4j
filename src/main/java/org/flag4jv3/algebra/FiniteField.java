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

import org.flag4jv3.algebra.elements.FiniteFieldElement;

import java.math.BigInteger;
import java.util.List;


/// Defines the algebraic structure of a [finite field](https://en.wikipedia.org/wiki/Finite_field). Finite fields are
/// often also referred to as Galois fields.
///
/// A **finite field**, <span class="latex-inline">(F, +, &middot;, 0, 1)</span>, is a finite set <span class="latex-inline">F</span>
/// equipped with two binary operations: addition (<span class="latex-inline">+</span>) and multiplication
/// (<span class="latex-inline">&middot;</span>). Finite fields are a special case of [fields][Field] where the set
/// <span class="latex-inline">F</span> is finite.
///
/// ### Finite Field Axioms
/// For all <span class="latex-replace">a, b, c in F</span><!-- LATEX: $a, b, c \in R$ -->:
/// - __Additive identity exists__: There exists an element <span class="latex-inline">0&isin;F</span>
/// such that <span class="latex-inline">a + 0 = a</span>
/// - __Multiplicative identity exists__: There exists an element <span class="latex-inline">1&isin;F</span> such that
/// <span class="latex-inline">a &middot; 1 = a</span>
/// - __Existence of additive inverses__: For every <span class="latex-inline">a&isin;R</span>, there exists an element
/// <span class="latex-inline">-a&isin;R</span> such that <span class="latex-inline">a + (-a) = 0</span>
/// - __Existence of multiplicative inverses__: For every nonzero <span class="latex-inline">a&isin;F</span>, there exists an element
/// <span class="latex-inline">a<sup>-1</sup>&isin;F</span> such that
/// <span class="latex-inline"> a &middot; a<sup>-1</sup> = 1</span>
/// - __Addition is associative__: <span class="latex-inline">a + (b + c) = (a + b) + c</span>
/// - __Addition is commutative__: <span class="latex-inline">a + b = b + a</span>
/// - __Multiplication is associative__: <span class="latex-inline">a &middot; (b &middot; c) = (a &middot; b) &middot; c</span>
/// - __Multiplication is commutative__: <span class="latex-inline">a &middot; b = b &middot; a</span>
/// - __Multiplication is distributive over addition__:
///     - __Left distributivity__: <span class="latex-inline">a &middot; (b + c) = a &middot; b + a &middot; c</span>
///     - __Right distributivity__: <span class="latex-inline">(a + b) &middot; c = a &middot; c + b &middot; c</span>
///
/// ### Examples of finite fields
/// - The set of integers modulo a prime number <span class="latex-inline">p</span>:
/// <span class="latex-inline">𝔽<sub>p</sub></span> or <span class="latex-replace">GF(p)</span><!-- LATEX: $\text{GF}(p)$ -->
/// - Extension fields: <span class="latex-inline">𝔽<sub>q</sub></span>
/// or <span class="latex-replace">GF(q)</span><!-- LATEX: $\text{GF}(q)$ --> with <span class="latex-inline">q=p<sup>k</sup></span>
/// where <span class="latex-inline">p</span> is a prime and <span class="latex-inline">k</span> is a positive integer.
///
/// This interface represents the algebraic *structure* itself.
/// Each element of the finite field exposes a reference back to its
/// structure via [FiniteFieldElement#structure()], so that [#zero()]/[#one()] and the
/// [#characteristic()] are always consistent across elements and reachable without an element instance in hand.
///
/// @param <T> the type of element belonging to this finite field.
/// @see FiniteFieldElement
/// @see Semiring
/// @see Ring
/// @see Field
public interface FiniteField<T extends FiniteFieldElement<T>> extends Field<T> {

    /// The characteristic of this finite field.
    ///
    /// For a finite field of order <span class="latex-inline">q = p<sup>k</sup></span>, the characteristic is the prime
    /// <span class="latex-inline">p</span>. Equivalently, it is the smallest positive integer
    /// <span class="latex-inline">n</span> such that <span class="latex-inline">n &middot; 1 = 0</span>.
    ///
    /// Unlike general rings and fields, the characteristic of a finite field is never zero.
    ///
    /// @return The characteristic of this finite field; always a prime number.
    @Override
    BigInteger characteristic(); // Re-declared here for more clear documentation.


    /// The degree of this finite field over its prime subfield.
    ///
    /// A finite field of characteristic <span class="latex-inline">p</span> is a finite-dimensional vector space over
    /// its prime subfield <span class="latex-inline">𝔽<sub>p</sub></span>; the degree is that dimension
    /// <span class="latex-inline">k &ge; 1</span>. A degree of `1` indicates a prime field
    /// <span class="latex-inline">𝔽<sub>p</sub></span> while a degree of <span class="latex-inline">k &gt; 1</span>
    /// indicates an extension field <span class="latex-inline">𝔽<sub>p<sup>k</sup></sub></span> of order
    /// <span class="latex-inline">p<sup>k</sup></span>.
    ///
    /// @return The degree of this finite field over its prime subfield; always a positive integer.
    default BigInteger order() {
        return characteristic().pow(degree());
    }


    /// The degree of this finite field over its prime subfield.
    ///
    /// A finite field of characteristic <span class="latex-inline">p</span> is a finite-dimensional vector space over
    /// its prime subfield <span class="latex-inline">𝔽<sub>p</sub></span>; the degree is that dimension
    /// <span class="latex-inline">k &ge; 1</span>. A degree of `1` indicates a prime field
    /// <span class="latex-inline">𝔽<sub>p</sub></span> while a degree of <span class="latex-inline">k &gt; 1</span>
    /// indicates an extension field <span class="latex-inline">𝔽<sub>p<sup>k</sup></sub></span> of order
    /// <span class="latex-inline">p<sup>k</sup></span>.
    ///
    /// @return The degree of this finite field over its prime subfield; always a positive integer.
    int degree();


    /// A primitive element of this finite field. That is, a generator of the field's multiplicative group.
    ///
    /// The multiplicative group of a finite field (all non-zero elements under multiplication) is cyclic, so there
    /// always exists an element <span class="latex-inline">g</span> whose powers
    /// <span class="latex-inline">g<sup>0</sup>, g<sup>1</sup>, &hellip;, g<sup>q-2</sup></span> enumerate every
    /// non-zero element of the field exactly once, where <span class="latex-inline">q</span> is the
    /// [order][#order()] of the field. Equivalently, <span class="latex-inline">g</span> has multiplicative order
    /// <span class="latex-inline">q - 1</span>.
    ///
    /// Primitive elements are generally not unique: a finite field of order <span class="latex-inline">q</span> has
    /// <span class="latex-replace">&phi;(q - 1)</span><!-- LATEX: $\varphi(q - 1)$ --> of them, where
    /// <span class="latex-replace">&phi;</span><!-- LATEX: $\varphi$ --> is Euler's totient function. Which primitive
    /// element is returned is unspecified, but implementations must return the same element on every invocation.
    ///
    /// @return A primitive element (generator of the multiplicative group) of this finite field.
    T primitiveElement();


    /// An iterable over all elements of this finite field.
    ///
    /// The returned iterable yields each element of the field exactly once; the total number of elements yielded is
    /// equal to this field's [order][#order()]. The iteration order is unspecified but must be deterministic.
    /// [Iterable#iterator()] must return a fresh, independent iterator, so the result may be iterated multiple times.
    ///
    /// Implementations must produce elements lazily rather than materializing the full field. Callers should consult
    /// [#order()] before attempting a full iteration as enumerating a large field may be computationally infeasible
    /// even though the returned iterable itself is inexpensive to construct.
    ///
    /// @return An iterable that lazily yields each element of this finite field exactly once.
    Iterable<T> elements();


    /// The coefficients of the monic irreducible polynomial defining this finite field's representation.
    ///
    /// The returned list satisfies the following:
    /// - Coefficients appear in ascending order of degree: slice `i` holds the coefficient of
    ///   <span class="latex-inline">x<sup>i</sup></span>.
    /// - The list has length `degree() + 1` and its last element is [BigInteger#ONE] (the modulus is monic).
    /// - Every coefficient lies in `[0, p)` where `p` is this field's characteristic.
    /// - The list is immutable.
    ///
    /// For prime fields (i.e. `degree() == 1`) the modulus is conventionally
    /// <span class="latex-inline">m(x) = x</span>, so this method returns `[0, 1]`.
    ///
    /// @return An immutable list of the coefficients of this field's modulus, in ascending order of degree.
    ///
    /// @implNote Implementations whose internal representation is not a polynomial basis (e.g. normal bases or
    /// tower constructions) must still report the modulus of the polynomial-basis presentation they expose.
    List<BigInteger> modulusCoefficients();
}
