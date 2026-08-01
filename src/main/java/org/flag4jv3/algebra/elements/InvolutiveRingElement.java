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

/// Represents a scalar element belonging to a ring which has an involution (i.e., conjugate) operation defined.
///
/// An involution is a unary operation <span class="latex-inline"><sup>*</sup></span> such that it satisfies:
/// 1. __Involutive__: <span class="latex-inline">(a<sup>\*</sup>)<sup>\*</sup> = a</span>
/// 2. __Additive__: <span class="latex-inline">(a + b)<sup>\*</sup> = a<sup>\*</sup> + b<sup>\*</sup></span>
/// 3. __Anti-multiplicative__: <span class="latex-inline">(ab)<sup>\*</sup> = b<sup>\*</sup>a<sup>\*</sup></span>
///
/// @param <T> The type of the scalar.
/// @see org.flag4jv3.algebra.Ring
/// @see RingElement
public interface InvolutiveRingElement<T extends InvolutiveRingElement<T>>
        extends RingElement<T> {

    /// Computes the conjugate value of `this` scalar.
    ///
    /// @return The conjugate value of `this` scalar.
    ///
    /// @implSpec The implementation must satisfy:
    /// 1. __Involutive__: <span class="latex-inline">(a<sup>\*</sup>)<sup>\*</sup> = a</span>
    /// 2. __Additive__: <span class="latex-inline">(a + b)<sup>\*</sup> = a<sup>\*</sup> + b<sup>\*</sup></span>
    /// 3. __Anti-multiplicative__: <span class="latex-inline">(ab)<sup>\*</sup> = b<sup>\*</sup>a<sup>\*</sup></span>
    ///
    T conjugate();


    /// Computes the norm of `this` scalar. This is defined as `mult(conjugate())`.
    ///
    /// @return The norm of `this` scalar.
    default T norm() {
        return mult(conjugate());
    }
}
