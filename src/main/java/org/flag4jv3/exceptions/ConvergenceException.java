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


/**
 * Thrown when an iterative numerical algorithm fails to converge within its
 * allowed iteration limit or convergence criteria.
 *
 * <p>This exception may be thrown by algorithms such as QR iteration,
 * iterative linear-system solvers, eigenvalue algorithms, or optimization
 * routines.
 */
public final class ConvergenceException extends LinearAlgebraException {
    // TODO: We could include another constructor that takes convergence diagnostics: iterations, residuals, etc.


    /**
     * Creates a convergence exception with the specified detail message.
     *
     * @param message Description of the convergence failure.
     */
    public ConvergenceException(String message) {
        super(message);
    }


    /**
     * Creates a convergence exception with the specified detail message and cause.
     *
     * @param message Description of the convergence failure.
     * @param cause The underlying cause of the failure.
     */
    public ConvergenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
