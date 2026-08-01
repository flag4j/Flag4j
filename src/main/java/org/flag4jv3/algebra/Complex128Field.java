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

import org.flag4jv3.algebra.elements.Complex128;

import java.io.Serial;
import java.math.BigInteger;

// TODO NOW: Improve docs.

/// Represents the field of complex numbers, ℂ, backed by two 64-bit floating point numbers (for a total of 128-bits).
public final class Complex128Field extends AlgebraicStructure implements Field<Complex128> {

    @Serial
    private static final long serialVersionUID = 1L;

    /// The singleton instance representing the field of complex numbers.
    public static final Complex128Field INSTANCE = new Complex128Field();


    private Complex128Field() {
        // Guards against reflective construction of a second instance. During class initialization INSTANCE is
        // still null when this constructor first runs, so the singleton itself passes. Deserialization bypasses
        // this constructor entirely; readResolve() handles that path.
        if (INSTANCE != null) {
            throw new IllegalStateException("ComplexField is a singleton; use ComplexField.INSTANCE.");
        }
    }


    /// The additive identity of this field: <span class="latex-inline">0 + 0i</span>.
    @Override
    public Complex128 zero() {
        return Complex128.ZERO;
    }


    /// The multiplicative identity of this field: <span class="latex-inline">1 + 0i</span>.
    @Override
    public Complex128 one() {
        return Complex128.ONE;
    }


    /// The characteristic of the ring.
    ///
    /// The characteristic of a ring is the smallest positive integer <span class="latex-replace">n</span> such that
    /// <span class="latex-replace">n&middot;1=0</span> if such an <span class="latex-replace">n</span> exists.
    /// Otherwise, the characteristic is <span class="latex-replace">0</span>.
    ///
    /// @return The characteristic of this ring.
    ///
    @Override
    public BigInteger characteristic() {
        return BigInteger.ZERO;
    }


    /// The canonical image of `n` in this field: the complex number <span class="latex-inline">n + 0i</span>.
    ///
    /// @implNote Exact for `|n| <= 2`<sup>`53`</sup>; larger magnitudes round to the nearest representable double.
    @Override
    public Complex128 valueOf(long n) {
        return new Complex128(n, 0.0);
    }


    /// The canonical image of `n` in this field: the complex number <span class="latex-inline">n + 0i</span>.
    ///
    /// @implNote Exact for `|n| <= 2`<sup>`53`</sup>; larger magnitudes round to the nearest representable double,
    /// and magnitudes beyond the double range yield an infinite real component.
    @Override
    public Complex128 valueOf(BigInteger n) {
        return new Complex128(n.doubleValue(), 0.0);
    }


    @Override
    protected Object canonicalInstance() {
        return INSTANCE;
    }


    /// Checks if `b` represents the same algebraic structure as this field.
    ///
    /// @return `true` if `b` is an instance of `ComplexField`; `false` otherwise.
    @Override
    public boolean equals(Object b) {
        return b instanceof Complex128Field;
    }


    @Override
    public int hashCode() {
        return Complex128Field.class.hashCode();
    }


    @Override
    public String toString() {
        return "Field (DoublePair)";
    }
}
