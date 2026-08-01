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

import org.flag4jv3.algebra.elements.FieldElement;

/// Defines the algebraic structure of a [field](https://en.wikipedia.org/wiki/Field_(mathematics)).
///
/// A **field**, <span class="latex-inline">(F, +, &middot;, 0, 1)</span>, is a set <span class="latex-inline">F</span>
/// equipped with two binary operations: addition (<span class="latex-inline">+</span>) and multiplication
/// (<span class="latex-inline">&middot;</span>). Fields are a special case of [rings][Ring] with the additional requirements
/// that every element in <span class="latex-inline">F</span> must have a multiplicative inverse and multiplication must be
/// commutative.
///
/// ### Field Axioms
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
/// ### Examples of fields
/// - The real numbers <span class="latex-inline">ℝ</span>
/// - The complex numbers <span class="latex-inline">ℂ</span>
/// - The rational numbers <span class="latex-inline">ℚ</span>
/// - The set of integers modulo a prime number (more specifically, this is a [finite field][FiniteField]).
///
/// This interface represents the algebraic *structure* itself.
/// Each element of the field exposes a reference back to its
/// structure via [FieldElement#structure()], so that [#zero()]/[#one()] and the
/// [#characteristic()] are always consistent across elements and reachable without an element instance in hand.
///
/// @param <T> the type of element belonging to this field.
/// @see FieldElement
/// @see Semiring
/// @see Ring
/// @see FiniteField
public interface Field<T extends FieldElement<T>> extends Ring<T> {
    // 
}
