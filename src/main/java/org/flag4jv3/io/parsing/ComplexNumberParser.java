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
import org.flag4jv3.util.tuples.DoublePair;
import org.flag4jv3.util.tuples.FloatPair;

import java.util.Objects;


/// A parser for parsing complex elements represented as a string.
///
/// ### Grammar
/// The grammar for a complex number is given by:
/// ```text
///     COMPLEX := i | j
///          | - (i | j)
///          | SIGNED_DOUBLE
///          | SIGNED_DOUBLE (i | j)
///          | SIGNED_DOUBLE (+ | -) (i | j)
///          | SIGNED_DOUBLE (+ | -) DOUBLE (i | j)
///
///     SIGNED_DOUBLE := DOUBLE | - DOUBLE
/// ```
///
/// > **NOTE**: If using, [#getComponentsAsFloats], the same grammer is used during parsing,
/// > but the values are cast to `float`s before returning.
///
/// ### Examples of Valid Complex Numbers
/// - `"3.43"`
/// - `"1 - 2i"`
/// - `"i"`
/// - `"-i"`
/// - `"-0.234 + 3.24i"`
/// - `"0.234 - 3.24j"`
/// - `"Infinity - Infinityi"`
public final class ComplexNumberParser {

    private ComplexNumberParser() {
        // Hide default constructor for utility class.
    }


    /// Parses a complex number in the form of a string into its real and imaginary parts represented as 32-bit floats.
    /// For example, the string `"2+3i"` would be parsed into real and imaginary parts
    /// `2` and `3` respectively.
    ///
    /// WARNING: This method may result in loss of precision as values will be parsed as `double`s then cast to `float`s.
    ///
    /// @param num A complex number in one of three forms: `a + bi`, `a` or `bi` where `a` and `b` are
    ///                 real elements and `i` is the imaginary unit `sqrt(-1)`
    /// @return The complex number represented by the string num.
    ///
    /// @throws Flag4jParsingException If `num` cannot be parsed as a complex number.
    /// @throws NullPointerException   If `num` is null.
    public static FloatPair getComponentsAsFloats(String num) {
        DoublePair components = getComponents(num);
        return new FloatPair(
                (float) components.first(),
                (float) components.second()
        );
    }


    /// Parses a complex number in the form of a string into its real and imaginary parts.
    /// For example, the string `"2+3i"` would be parsed into real and imaginary parts
    /// `2` and `3` respectively.
    ///
    /// @param num A complex number in one of three forms: `a + bi`, `a` or `bi` where `a` and `b` are
    ///                 real elements and `i` is the imaginary unit `sqrt(-1)`
    /// @return The complex number represented by the string num.
    ///
    /// @throws Flag4jParsingException If `num` cannot be parsed as a complex number.
    /// @throws NullPointerException   If `num` is null.
    public static DoublePair getComponents(String num) {
        Objects.requireNonNull(num, "num");

        ComplexNumberLexer lex = new ComplexNumberLexer(num);
        Token token = lex.getNextToken();

        // Case 1: Just the imaginary unit i or j.
        if (token.matches("im", "i")) {
            lex.getNextToken().errorCheck("eof", "");
            return new DoublePair(0.0, 1.0);
        }

        Token real;
        double realSign;

        // Case 2: The negative imaginary unit -i or -j.
        if (token.matches("opp", "-")) {
            token = lex.getNextToken();

            if (token.matches("im", "i")) {
                lex.getNextToken().errorCheck("eof", "");
                return new DoublePair(0.0, -1.0);
            }

            token.errorCheck("num");
            real = token;
            realSign = -1.0;

        } else {
            token.errorCheck("num");
            real = token;
            realSign = 1.0;
        }

        double realValue = realSign*parseDouble(
                real.getDetails(),
                "real component"
        );

        token = lex.getNextToken();

        // Case 3: Real component only.
        if (token.matches("eof", "")) {
            return new DoublePair(realValue, 0.0);
        }

        // Case 4: Imaginary component only.
        if (token.matches("im", "i")) {
            lex.getNextToken().errorCheck("eof", "");
            return new DoublePair(0.0, realValue);
        }

        token.errorCheck("opp");
        double imaginarySign = token.matches("opp", "-") ? -1.0 : 1.0;

        token = lex.getNextToken();

        // Case 5: Real component with imaginary unit.
        if (token.matches("im", "i")) {
            lex.getNextToken().errorCheck("eof", "");
            return new DoublePair(realValue, imaginarySign);
        }

        // Case 6: Real and complex component.
        token.errorCheck("num");
        double imaginaryValue = parseDouble(
                token.getDetails(),
                "imaginary component"
        );

        lex.getNextToken().errorCheck("im", "i");
        lex.getNextToken().errorCheck("eof", "");

        return new DoublePair(
                realValue,
                imaginarySign*imaginaryValue
        );
    }


    /// Attempts to parse a string as a double.
    ///
    /// @param text The string to parse as a double.
    /// @param componentName The name of the component being parsed. Useful for error messages.
    private static double parseDouble(String text, String componentName) {
        try {
            return Double.parseDouble(text);
        } catch (NumberFormatException e) {
            throw new Flag4jParsingException(
                    "Invalid " + componentName + " in complex number: \"" + text + "\""
            );
        }
    }
}
