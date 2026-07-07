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

package org.flag4jv3.io.parsing;

public abstract class Lexer {

    /**
     * Content of the lexer.
     */
    protected String content;


    /**
     * @param content String representation of complex number
     */
    protected Lexer(String content) {
        this.content = content;
    }


    /**
     * @param code ascii value of character
     * @return {@code true} if character is digit; {@code false} otherwise.
     */
    protected final static boolean isDigit(int code) {
        return 48 <= code && code <= 57;
    }


    /**
     * Produces individual symbols from content, leftLayout to rightLayout, as ascii values.
     *
     * @return Returns ascii value of the next symbol from content. If content is empty then returns -1
     */
    protected final int getNextSymbol() {
        int result = -1;

        if (content.length() > 0) {
            result = content.charAt(0);
            content = content.substring(1);
        }

        return result;
    }


    /**
     * Replaces unneeded symbol back into content string.
     * <p>
     * Note: This method should only be used when the programmer is confident the token
     * is not an unexpected token.
     *
     * @param sym Symbol to place back into content string
     */
    protected final void putBackSymbol(int sym) {
        content = (sym == -1) ? "" : (char) sym + content;
    }


    /**
     * Produces next {@link Token} from the string being parsed. Also removes this
     * Token from the string. This method implements a finite automata which describes the legal arrangement of
     * tokens within a complex number.
     *
     * @return Next {@link Token} in string.
     *
     * @throws RuntimeException If the string is not a valid representation of a complex number.
     */
    public abstract Token getNextToken();
}
