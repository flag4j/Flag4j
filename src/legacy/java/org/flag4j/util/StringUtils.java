/*
 * MIT License
 *
 * Copyright (c) 2022-2026. Jacob Watters
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

import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Complex64;
import org.flag4j.numbers.RealF32;
import org.flag4j.numbers.RealF64;

/**
 * A class which provides simple utility methods for {@link String strings}.
 */
public final class StringUtils {

    private StringUtils() {
        // hide public constructor for utility class.
    }


    /**
     * Centers a string within a specified size bin.
     * @param s String to center.
     * @param size Size of the bin to center string within.
     * @return A whitespace string of specified size with the string s centered within it.
     */
    public static String center(String s, int size) {
        return center(s, size, " ");
    }


    /**
     * Centers a string within a specified size bin.
     * @param s String to center.
     * @param size Size of the bin to center string within.
     * @param pad Padding character.
     * @return A string made up of the padding character of specified size with the string s centered within it.
     */
    public static String center(String s, int size, String pad) {
        if (s == null || size <= s.length())
            return s;

        StringBuilder sb = new StringBuilder(size);
        sb.append(String.valueOf(pad).repeat((size - s.length()) / 2));
        sb.append(s);
        while (sb.length() < size) {
            sb.append(pad);
        }

        return sb.toString();
    }


    /**
     * Gets the string representation of a double rounded to the specified precision.
     * @param value Value to convert to String.
     * @param precision Precision to round value to.
     * @return The string representation of a double rounded to the specified precision.
     */
    public static String ValueOfRound(double value, int precision) {
        return String.valueOf(Complex128.round(new Complex128(value), precision));
    }


    /**
     * Gets the string representation of a field element rounded to the specified precision if possible.
     * @param value Value to convert to String.
     * @param precision Precision to round value to if the value can be rounded.
     * @return The string representation of {@code value} rounded to the specified precision if the field type can be rounded.
     */
    public static <T> String ValueOfRound(T value, int precision) {
        String valueOf;
        if(value instanceof Complex128)
            valueOf = Complex128.round((Complex128) value, precision).toString();
        else if(value instanceof Complex64)
            valueOf = Complex64.round((Complex64) value, precision).toString();
        else if(value instanceof RealF64)
            valueOf = RealF64.round((RealF64) value, precision).toString();
        else if(value instanceof RealF32)
            valueOf = RealF32.round((RealF32) value, precision).toString();
        else
            valueOf = value.toString();

        return valueOf;
    }
}
