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

package org.flag4jv3.exceptions;

import java.io.Serial;

/// Thrown when the number of indices supplied for an [nD-array][org.flag4jv3.ndarrays.base.NDArrayBase] does
/// not match the array [rank][org.flag4jv3.ndarrays.Shape#rank()] or when an operation is attempted on an nD-array with an unsupported rank.
///
/// For example, this exception may be thrown when two indices are supplied
/// for an array with rank three.
public final class NDArrayRankException extends IllegalArgumentException {
    @Serial
    private static final long serialVersionUID = 1L;


    public NDArrayRankException(String message) {
        super(message);
    }


    public NDArrayRankException(String message, Throwable cause) {
        super(message, cause);
    }
}