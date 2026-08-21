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

import org.flag4jv3.ndarrays.Shape;

// TODO NOW: This needs to replace the old `ValidateParameters`. Also consider if things specific to certain objects should go
//  somewhere else (e.g., `validateBufferSize` and `ensureNonOverlapping` might go in `LayoutUtils` or something similar).
public final class NewValidateParameters {

    private NewValidateParameters() {
        // Hide default constructor for utility class.
    }


    /// Enum representing various parities of real numbers.
    public enum Sign {
        POSITIVE("positive"),
        NEGATIVE("negative"),
        NON_NEGATIVE("non-negative"),
        NON_POSITIVE("non-positive");

        /// Human-readable text representing the sign.
        private final String readableText;


        private Sign(String readableText) {
            this.readableText = readableText;
        }


        @Override
        public String toString() {
            return this.readableText;
        }
    }


    /// Represents inclusivity of bounds for numeric range.
    public enum Inclusivity {
        /// Range with exclusive lower and upper bounds.
        EXCLUSIVE,
        /// Range with inclusive lower bound and exclusive upper bound.
        INCLUSIVE_LOWER,
        /// Range with inclusive upper bound and exclusive lower bound.
        INCLUSIVE_UPPER,
        /// Range with inclusive lower and upper bounds.
        INCLUSIVE;


        /// Formats a range based on this inclusivity.
        ///
        /// @param low The lower bound of the range.
        /// @param high The upper bound of the range.
        /// @return A string representing the specified range with this inclusivity.
        String formatRange(double low, double high) {
            return switch (this) {
                case EXCLUSIVE -> "(" + low + ", " + high + ")";
                case INCLUSIVE_LOWER -> "[" + low + ", " + high + ")";
                case INCLUSIVE_UPPER -> "(" + low + ", " + high + "]";
                case INCLUSIVE -> "[" + low + ", " + high + "]";
            };
        }
    }


    /**
     * Validates that a buffer is at least as large as the specified minimum size.
     *
     * @param minSize The minimum allowed size of the buffer.
     * @param bufferSize The buffer's actual size.
     * @param msg The name of the buffer. If `null`, then a default error message is used.
     */
    public static void validateBufferSize(int minSize, int bufferSize, String msg) {
        if (bufferSize < minSize) {
            msg = replaceIfNull(msg, "Expected buffer size to be greater than " + minSize + " but got " + bufferSize);
            throw new IllegalArgumentException(msg);
        }
    }


    /// Ensures that two [shapes][Shape] are equal.
    ///
    /// @param shape1 First shape to compare.
    /// @param shape2 Second shape to compare.
    /// @param msg The message to display if the shapes are *not* equal. If `null`, then a default error message is used.
    public static void ensureSameShape(Shape shape1, Shape shape2, String msg) {
        if (!shape1.equals(shape2)) {
            msg = replaceIfNull(msg, "Expecting shapes to be equal but got " + shape1 + " and " + shape2);
            throw new IllegalArgumentException(msg);
        }
    }


    /// Ensures if two `double`'s are equal.
    ///
    /// @param a The first `double` to compare.
    /// @param b The second `double` to compare.
    /// @param msg The message to display if `a` and `b` are *not* equal. If `null`, then a default error
    /// message is used.
    /// @throws IllegalArgumentException If `a != b`.
    public static void ensureEqual(double a, double b, String msg) {
        if (a != b) {
            msg = replaceIfNull(msg, "Expecting values to be equal but got " + a + " and " + b);
            throw new IllegalArgumentException(msg);
        }
    }


    /// Ensures that `value > min` is true.
    ///
    /// @param value The value of interest.
    /// @param min The minimum allowed value (exclusives).
    /// @param msg The message to display if `value <= min`. If `null`, then a default error message is used.
    /// @throws IllegalArgumentException If `value <= min`.
    public static void ensureGreaterThan(double value, double min, String msg) {
        if (value <= min) {
            msg = replaceIfNull(msg, "Expecting value to be greater than " + min + " but got " + value);
            throw new IllegalArgumentException(msg);
        }
    }


    /// Ensures that a value is within the specified range.
    ///
    /// @param value The value of interest.
    /// @param low The lower bound of the range.
    /// @param high The upper bound of the range.
    /// @param inclusivity The inclusivity of the range bounds.
    /// @param msg The message to display if the `value` is not within the specified range. If `null`, then a default
    /// error message is used.
    public static void ensureInRange(double value, double low, double high, Inclusivity inclusivity, String msg) {
        boolean inRange = switch (inclusivity) {
            case EXCLUSIVE -> low < value && value < high;
            case INCLUSIVE_LOWER -> low <= value && value < high;
            case INCLUSIVE_UPPER -> low < value && value <= high;
            case INCLUSIVE -> low <= value && value <= high;
        };

        if (!inRange) {
            msg = replaceIfNull(msg, "Expected value to be in range "
                    + inclusivity.formatRange(low, high) + " but got " + value);
            throw new IllegalArgumentException(msg);
        }
    }


    /// Ensures that a `value` has the specified `sign`.
    ///
    /// @param value The value to check sign of.
    /// @param sign The desired sign.
    /// @param msg The error message to display if the `value`'s sign does not match `sign`. If `null`, then a default error
    /// message is used.
    /// @throws IllegalArgumentException If `value`'s sign does not match `sign`.
    /// @see #ensureSign(int[], Sign, String)
    /// @see #ensureSign(float[], Sign, String)
    /// @see #ensureSign(double[], Sign, String)
    public static void ensureSign(double value, Sign sign, String msg) {
        boolean isValid = switch (sign) {
            case POSITIVE -> value > 0;
            case NEGATIVE -> value < 0;
            case NON_NEGATIVE -> value >= 0;
            case NON_POSITIVE -> value <= 0;
        };

        if (!isValid) {
            msg = replaceIfNull(msg, "Expected value to be " + sign + " but got " + value);
            throw new IllegalArgumentException(msg);
        }
    }


    /// Ensures that all elements of `values` have the specified `sign`.
    ///
    /// @param values The values to check the sign of.
    /// @param sign The desired sign.
    /// @param msg The error message to display if any element of `values` sign does not match `sign`.
    /// If `null`, then a default error message is used.
    /// @throws IllegalArgumentException If *any* element of `values` sign does not match `sign`.
    /// @see #ensureSign(double, Sign, String)
    /// @see #ensureSign(float[], Sign, String)
    /// @see #ensureSign(double[], Sign, String)
    public static void ensureSign(int[] values, Sign sign, String msg) {
        for (int i = 0; i < values.length; i++) {
            ensureSign(values[i], sign, msg);
        }
    }


    /// Ensures that all elements of `values` have the specified `sign`.
    ///
    /// @param values The values to check the sign of.
    /// @param sign The desired sign.
    /// @param msg The error message to display if any element of `values` sign does not match `sign`.
    /// If `null`, then a default error message is used.
    /// @throws IllegalArgumentException If *any* element of `values` sign does not match `sign`.
    /// @see #ensureSign(int[], Sign, String)
    /// @see #ensureSign(double, Sign, String)
    /// @see #ensureSign(double[], Sign, String)
    public static void ensureSign(float[] values, Sign sign, String msg) {
        for (int i = 0; i < values.length; i++) {
            ensureSign(values[i], sign, msg);
        }
    }


    /// Ensures that all elements of `values` have the specified `sign`.
    ///
    /// @param values The values to check the sign of.
    /// @param sign The desired sign.
    /// @param msg The error message to display if any element of `values` sign does not match `sign`.
    /// If `null`, then a default error message is used.
    /// @throws IllegalArgumentException If *any* element of `values` sign does not match `sign`.
    /// @see #ensureSign(int[], Sign, String)
    /// @see #ensureSign(float[], Sign, String)
    /// @see #ensureSign(double, Sign, String)
    public static void ensureSign(double[] values, Sign sign, String msg) {
        for (int i = 0; i < values.length; i++) {
            ensureSign(values[i], sign, msg);
        }
    }


    /// If a string is `null`, then returns the specified replacement.
    ///
    /// @param str The string of interest.
    /// @param replacement The replacement.
    /// @return `str` if `str` *is not* `null`; otherwise, `replacement` (even if `replacement` is `null`).
    private static String replaceIfNull(String str, String replacement) {
        return (str == null) ? replacement : str;
    }
}
