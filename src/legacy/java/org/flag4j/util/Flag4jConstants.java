/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.util;


/**
 * A simple utility class containing useful constants.
 */
public final class Flag4jConstants {

    // 32-bit single precision float constants.
    /**
     * The rounding error for 32-bit floating point numbers.
     */
    public static final float EPS_F32 = Math.ulp(1.0f) / 2.0f;
    /**
     * The smallest "safe" 32-bit floating point value. That is, the smallest possible float value
     * such that {@code 1.0f / SAFE_MIN_F32} does not overflow.
     */
    public static final double SAFE_MIN_F32 = Float.MIN_NORMAL;
    /**
     * Overflow threshold for 42-bit floating point values. Equivalent to {@link Float#MAX_VALUE}.
     */
    public static final double OVERFLOW_THRESH_F32 = Float.MAX_VALUE;

    // 64-bit double precision float constants.
    /**
     * The rounding error for 64-bit double precision floating point numbers.
     */
    public static final double EPS_F64 = Math.ulp(1.0d) / 2.0;
    /**
     * The smallest "safe" 64-bit floating point value. That is, the smallest possible double value
     * such that {@code 1.0d / SAFE_MIN_F64} does not overflow.
     */
    public static final double SAFE_MIN_F64 = Double.MIN_NORMAL;
    /**
     * Overflow threshold for 64-bit floating point values. Equivalent to {@link Double#MAX_VALUE}.
     */
    public static final double OVERFLOW_THRESH_F64 = Double.MAX_VALUE;

    private Flag4jConstants() {
        // Hide default constructor for utility class.
    }
}
