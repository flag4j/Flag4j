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

public final class NewValidateParameters {

    private NewValidateParameters() {
    } // Hide default constructor for utility class.


    /**
     * Validates that a buffer is at least as large as the specified minimum size.
     *
     * @param minSize The minimum allowed size of the buffer.
     * @param bufferSize The buffers actual size.
     * @param msg The name of the buffer.
     */
    public static void validateBufferSize(int minSize, int bufferSize, String msg) {
        msg = replaceIfNull(msg, "Expected buffer size to be greater than " + minSize + " but got " + bufferSize);

        if (bufferSize < minSize) {
            throw new IllegalArgumentException(msg);
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
