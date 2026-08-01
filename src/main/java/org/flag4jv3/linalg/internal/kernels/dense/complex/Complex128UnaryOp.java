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

package org.flag4jv3.linalg.internal.kernels.dense.complex;

// TODO NOW: Make docs reflect that this is generic item represented by two doubles (complex number, 2d point, etc.)

/// Represents an operation on a single 128-bit complex-valued operand that produces a 128-bit complex-valued.
/// This is a type specialization of UnaryOperator for complex numbers represented as two `double` primitives.
///
/// This is a [functional interface][FunctionalInterface] whose functional method is [#apply(double, double, double[], int).
///
/// @see Complex64UnaryOp
@FunctionalInterface
public interface Complex128UnaryOp {

    /// Applies an operation on a single complex-valued operand that produces a complex-valued result.
    ///
    /// @param re The real component of the complex number being operated on.
    /// @param im The imaginary component of the complex number being operated on.
    /// @param out The buffer the operation output should be stored in. This is assumed to store complex
    /// values with their real/imaginary components interleaved (i.e., all even indices contain real components while all odd
    /// indices contain imaginary components).
    /// @param outOffset The offset in `out` to write the output of the operation to. The real component of the output will be
    /// stored in `out[outOffset]` and the imaginary component in `out[outOffset + 1]`.
    void apply(double re, double im, double[] out, int outOffset);
}
