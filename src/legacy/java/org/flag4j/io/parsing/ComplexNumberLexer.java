/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.io.parsing;

import org.flag4j.util.exceptions.Flag4jParsingException;

/**
 * A lexer for producing the tokens of a complex number represented as a string.
 */
class ComplexNumberLexer extends Lexer {
    /**
     * Error message for unexpected symbol.
     */
    private static final String ERR_MSG = "Unexpected symbol while parsing: %s";

    /**
     * @param content - String representation of complex number
     */
    public ComplexNumberLexer(String content) {
        super(content);
    }


    /**
     * Produces next {@link Token} from complex number string. Also removes this
     * Token from the string. This method implements a finite automata which describes the legal arrangement of
     * tokens within a complex number.
     *
     * @return Next {@link Token} in string.
     * @throws Flag4jParsingException If the string is not a valid representation of a complex number.
     */
    public Token getNextToken() {
        int state = 1;  // State of Finite Automata
        boolean done = false;
        StringBuilder dataBuilder = new StringBuilder(); // specific info for the Token
        int sym;  // holds current symbol


        do {
            sym = getNextSymbol(); // Will return -1 if there is no symbol to get

            if(state == 1) {
                if(sym == 45) {
                    state = 2;
                    dataBuilder.append((char) sym);
                }
                else if(sym == 43) {
                    state = 4;
                    dataBuilder.append((char) sym);
                    done = true;
                }
                else if(isDigit(sym)) {
                    state = 3;
                    dataBuilder.append((char) sym);
                }
                else if(sym == 'i' || sym == 'j') {
                    state = 6;
                    dataBuilder.append((char) sym);
                    done = true;
                }
                else if(sym == -1) { // We have reached the end of the string
                    state = 5;
                    done = true;
                }
                else if(!(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32)){
                    // Otherwise, if it is not a whitespace character, we have encountered an unexpected symbol.
                    error(String.format(ERR_MSG, (char) sym));
                }
            }

            else if(state == 2) {
                if(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32) { // Whitespace
                    putBackSymbol(sym);
                    done = true;
                }
                else if(isDigit(sym)) {
                    state = 3;
                    dataBuilder.append((char) sym);
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }
            else if(state == 3) {
                if(isDigit(sym)) {
                    // State does not need to change here.
                    dataBuilder.append((char) sym);
                }
                else if(sym == '.') {
                    state = 7;
                    dataBuilder.append((char) sym);
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }
            else if(state == 7) {
                if(isDigit(sym)) {
                    state = 8;
                    dataBuilder.append((char) sym);
                }
                else {
                    error(String.format(ERR_MSG, (char) sym));
                }
            }
            else if(state == 8) {
                if(isDigit(sym)) {
                    // State does not need to change here.
                    dataBuilder.append((char) sym);
                }
                else {
                    putBackSymbol(sym);
                    done = true;
                }
            }

        } while(!done);

        if(state == 2 || state == 4) { // we have an operator
            return new Token("opp", dataBuilder.toString());
        }
        else if(state == 3 || state == 8) { // we have a number
            return new Token("num", dataBuilder.toString());
        }
        else if(state == 5) { // end of number
            return new Token("eof", dataBuilder.toString());
        }
        else if(state == 6) { // we have the imaginary unit
            return new Token("im","i");
        }
        else {
            error("Somehow Lexer FA halted in bad state " + state);
            return null;
        }
    }


    /**
     * Stops execution with an error message
     * @param message - error message to print
     */
    protected static void error( String message ) {
        throw new Flag4jParsingException(message);
    }
}
