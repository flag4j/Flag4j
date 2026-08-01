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

package org.flag4jv3.util;

/// Useful constants for Flag4j.
public final class Flag4jConstants {
    private Flag4jConstants() {
        // Prevent instantiation for utility class.
    }


    /// Machine precision (i.e., epsilon) for single precision 32-bit floating point numbers. This
    /// is defined as `Math.ulp(1.0f)` which is the spacing between `1.0f` and the next largest float value.
    public static final float EPS_F32 = Math.ulp(1.0f);

    /// Machine precision (i.e., epsilon) for double precision 64-bit floating point numbers. This
    /// is defined as `Math.ulp(1.0d)` which is the spacing between `1.0d` and the next largest double value.
    public static final double EPS_F64 = Math.ulp(1.0d);
}
