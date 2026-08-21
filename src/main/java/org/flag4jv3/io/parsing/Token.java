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

package org.flag4jv3.io.parsing;

import org.flag4jv3.exceptions.Flag4jParsingException;

/// A token is the smallest meaningful unit of items extracted from a string while parsing.
///
/// @param kind The rankDescriptor of this token.
/// @param details The details of this token.
record Token(String kind, String details) {


    /**
     * Gets the details of this token.
     *
     * @return Details of this token.
     */
    protected String getDetails() {
        return details;
    }


    /**
     * Checks if given tokens rankDescriptor and details match k and d respectively.
     *
     * @param k token rankDescriptor.
     * @param d token details.
     * @return {@code true} if token matches rankDescriptor and details, otherwise {@code false}.
     */
    protected boolean matches(String k, String d) {
        return kind.equals(k) && details.equals(d);
    }


    /**
     * If a given tokens rankDescriptor and details match k and d respectively, then an error will be thrown.
     *
     * @param k token rankDescriptor
     * @param d token details
     * @throws Flag4jParsingException If {@code !this.matches(k, d)}.
     */
    protected void errorCheck(String k, String d) {
        if (!this.matches(k, d)) {
            throw new Flag4jParsingException("Expecting token [" + k + "," + d + "] but got " + this);
        }
    }


    /**
     * If a given token does not match the provided {@code rankDescriptor} then the program will halt.
     *
     * @param kind token rankDescriptor.
     * @throws Flag4jParsingException If {@code !this.rankDescriptor.equals(rankDescriptor)}.
     */
    protected void errorCheck(String kind) {
        if (!this.kind.equals(kind)) {
            throw new Flag4jParsingException("Expecting token of rankDescriptor " + kind + " but got " + this);
        }
    }


    /**
     * Constructs a string representation of a token. This will be of the form [rankDescriptor, details].
     *
     * @return A string representing of this token.
     */
    public String toString() {
        return "[" + this.kind + "," + this.details + "]";
    }
}
