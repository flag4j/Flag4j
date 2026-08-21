/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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


/// Base exception for runtime failures in linear algebra operations.
///
/// Examples include expected matrix properties (e.g., positive definite), failed matrix factorizations, and numerical
/// algorithm failures.
///
/// There are cases where a more specific exception may be appropriate:
/// - [SingularMatrixException] should be thrown when a matrix is expected to be singular but is not.
/// - [ConvergenceException] should be thrown when a numerical algorithm fails to converge within some criteria.
public class LinearAlgebraException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 1L;


    public LinearAlgebraException(String message) {
        super(message);
    }


    public LinearAlgebraException(String message, Throwable cause) {
        super(message, cause);
    }


    public LinearAlgebraException(Throwable cause) {
        super(cause);
    }
}
