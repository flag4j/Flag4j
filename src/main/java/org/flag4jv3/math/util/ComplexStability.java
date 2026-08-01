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

package org.flag4jv3.math.util;

// TODO NOW: Ensure the docs are correct/clear.

/// Constants useful for performing operations on complex numbers safely (e.g., avoiding under/overflows).
public final class ComplexStability {

    private ComplexStability() {
        // Hide default constructor for utility class.
    }


    /// When complex number is represented as two 64-bit floating point numbers:
    /// Largest exponent of `max(|re|, |im|)`, such that `re*re + im*im` cannot overflow. This is set to `510` because
    /// each square is below `2^1022` and this their sum is below `2^1023 < Double.MAX_VALUE`.
    public static final int SAFE_EXP_HI_128 = 510;


    /// When complex number is represented as two 64-bit floating point numbers:
    /// Smallest exponent of `max(|re|, |im|)` for which every square that can contribute to the 53-bit sum is
    /// a full-precision normal double. This is set to `-484` because the larget square is at least `2^-968` so anything
    /// within 53 bits of it sits at or above `2^-1021 > Double.MIN_NORMAL`
    public static final int SAFE_EXP_LO_128 = -484;

    /// When complex number is represented as two 32-bit floating point numbers:
    /// Largest exponent of `max(|re|, |im|)`, such that `re*re + im*im` cannot overflow. This is set to `63` because
    /// each square is below `2^126` and this their sum is below `2^127 < Float.MAX_VALUE`.
    public static final int SAFE_EXP_HI_64 = 62;

    /// When complex number is represented as two 64-bit floating point numbers:
    /// Smallest exponent of `max(|re|, |im|)` for which every square that can contribute to the 24-bit sum is
    /// a full-precision normal float. This is set to `-51` because the larget square is at least `2^-102` so anything
    /// within 53 bits of it sits at or above `2^-125 > Double.MIN_NORMAL`
    public static final int SAFE_EXP_LO_64 = -51;


    public static final double LOG_MAX_128 = 0x1.62e42fefa39efp+9; // log(Double.MAX_VALUE)

    public static final double LN_2_128 = 0x1.62e42fefa39efp-1;

    public static final double INV_LN_2_128 = 0x1.71547652b82fep+0;
}
