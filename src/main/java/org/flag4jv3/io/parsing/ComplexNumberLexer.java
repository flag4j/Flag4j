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

/// A lexer for producing the tokens of a complex number represented as a string.
class ComplexNumberLexer extends Lexer {
    /// Error message for unexpected symbol.
    private static final String ERR_MSG = "Unexpected symbol while parsing: %s";


    /// Constructs a lexer for complex scalars.
    ///
    /// @param content String representation of a complex number
    public ComplexNumberLexer(String content) {
        super(content);
    }


    @Override
    public Token getNextToken() {
        int sym;

        do {
            sym = getNextSymbol();
        } while (isWhitespace(sym));

        if (sym == -1) {
            return new Token("eof", "");
        }

        // Always leave sign interpretation to the parser.
        if (sym == '+' || sym == '-') {
            return new Token("opp", Character.toString((char) sym));
        }

        if (sym == 'i' || sym == 'j') {
            return new Token("im", "i");
        }

        if (startsNumber(sym)) {
            StringBuilder data = new StringBuilder();
            scanNumber(data, sym);
            return new Token("num", data.toString());
        }

        error(String.format(ERR_MSG, (char) sym));
        return null; // Unreachable.
    }


    /// Checks if the symbol is a whitespace character.
    ///
    /// @param sym The symbol to check.
    /// @return `true` if `sym` is a whitespace character; `false` otherwise.
    private static boolean isWhitespace(int sym) {
        return sym != -1 && Character.isWhitespace((char) sym);
    }


    /// Checks if the symbol starts a numeric token (including `Infinity` and `NaN`).
    ///
    /// @param sym The symbol to check.
    /// @return `true` if `sym` starts a numeric token; `false` otherwise.
    private static boolean startsNumber(int sym) {
        return isDigit(sym)
                || sym == '.'
                || sym == 'N'
                || sym == 'I';
    }


    /// Reads a numeric token begining with `first`.
    ///
    /// Supports decimal floating-point literals, optional exponents, `NaN`, and `Infinity`.
    ///
    /// @param data Builder to append the number to.
    /// @param first The first symbol of the number.
    private void scanNumber(StringBuilder data, int first) {
        if (first == 'N') {
            scanNamedLiteral(data, first, "NaN");
            return;
        }

        if (first == 'I') {
            scanNamedLiteral(data, first, "Infinity");
            return;
        }

        int sym;

        if (first == '.') {
            data.append('.');

            sym = getNextSymbol();
            if (!isDigit(sym)) {
                error("Expected a digit after decimal point while parsing number.");
            }

            sym = appendDigits(data, sym);
        } else {
            // first is guaranteed to be a digit.
            sym = appendDigits(data, first);

            if (sym == '.') {
                data.append('.');

                sym = getNextSymbol();

                // "1." is valid, so digits after the decimal point are optional.
                if (isDigit(sym)) {
                    sym = appendDigits(data, sym);
                }
            }
        }

        if (sym == 'e' || sym == 'E') {
            data.append((char) sym);
            sym = getNextSymbol();

            if (sym == '+' || sym == '-') {
                data.append((char) sym);
                sym = getNextSymbol();
            }

            if (!isDigit(sym)) {
                error("Expected a digit in exponent while parsing number.");
            }

            sym = appendDigits(data, sym);
        }

        putBackSymbol(sym);
    }


    /**
     * Appends consecutive decimal digits beginning with {@code sym}.
     *
     * @return The first non-digit symbol after the scanned digits.
     */
    private int appendDigits(StringBuilder data, int sym) {
        while (isDigit(sym)) {
            data.append((char) sym);
            sym = getNextSymbol();
        }

        return sym;
    }


    /// Reads a named literal from this lexer (e.g., `NaN`, or `Infinity`).
    ///
    /// @param data Builder to append literal to.
    /// @param first The first symbol.
    /// @param literal The literal value to scan for.
    private void scanNamedLiteral(StringBuilder data, int first, String literal) {
        data.append((char) first);

        for (int i = 1; i < literal.length(); i++) {
            int sym = getNextSymbol();

            if (sym != literal.charAt(i)) {
                error("Invalid floating-point literal while parsing complex number.");
            }

            data.append((char) sym);
        }
    }


    /**
     * Stops execution with an error message
     *
     * @param message Error message to print
     */
    protected static void error(String message) {
        throw new Flag4jParsingException(message);
    }
}
