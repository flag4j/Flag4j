package org.flag4jv3.io.parsing;

import org.flag4jv3.exceptions.Flag4jParsingException;
import org.flag4jv3.util.tuples.DoublePair;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ComplexNumberParserTests {

    private static final double TOL = 0.0;


    @Test
    void parsesRealNumbers() {
        assertComponents("3.43", 3.43, 0.0);
        assertComponents("-3.43", -3.43, 0.0);
        assertComponents(".5", 0.5, 0.0);
        assertComponents("1.", 1.0, 0.0);
        assertComponents("1e3", 1000.0, 0.0);
        assertComponents("-2.5e-4", -2.5e-4, 0.0);
    }


    @Test
    void parsesUnitImaginaryNumbers() {
        assertComponents("i", 0.0, 1.0);
        assertComponents("j", 0.0, 1.0);
        assertComponents("-i", 0.0, -1.0);
        assertComponents("-j", 0.0, -1.0);
    }


    @Test
    void parsesPureImaginaryNumbers() {
        assertComponents("2i", 0.0, 2.0);
        assertComponents("-2i", 0.0, -2.0);
        assertComponents(".25j", 0.0, 0.25);
        assertComponents("-2.5e-4i", 0.0, -2.5e-4);
    }


    @Test
    void parsesComplexNumbers() {
        assertComponents("2+3i", 2.0, 3.0);
        assertComponents("2-3i", 2.0, -3.0);
        assertComponents("-2+3i", -2.0, 3.0);
        assertComponents("-2-3i", -2.0, -3.0);
        assertComponents("2+i", 2.0, 1.0);
        assertComponents("2-i", 2.0, -1.0);
        assertComponents("1 - 2i", 1.0, -2.0);
        assertComponents("-0.234 + 3.24i", -0.234, 3.24);
        assertComponents("1e3-2.5E-4i", 1000.0, -2.5e-4);
    }


    @Test
    void parsesSpecialDoubleValues() {
        DoublePair infinity = ComplexNumberParser.getComponents("Infinity-Infinityi");
        assertEquals(Double.POSITIVE_INFINITY, infinity.first());
        assertEquals(Double.NEGATIVE_INFINITY, infinity.second());

        DoublePair nan = ComplexNumberParser.getComponents("NaN+2i");
        assertTrue(Double.isNaN(nan.first()));
        assertEquals(2.0, nan.second());
    }


    @Test
    void rejectsInvalidComplexNumbers() {
        assertInvalid("");
        assertInvalid(" ");
        assertInvalid("+");
        assertInvalid("-");
        assertInvalid("+i");
        assertInvalid("+2");
        assertInvalid("2+");
        assertInvalid("2-");
        assertInvalid("2ii");
        assertInvalid("i2");
        assertInvalid("2 3i");
        assertInvalid("2+-3i");
        assertInvalid("2--3i");
        assertInvalid("2+3");
        assertInvalid("2+3ii");
        assertInvalid("1e");
        assertInvalid("1e+");
        assertInvalid(".");
        assertInvalid("-.i");
        assertInvalid("abv");
    }


    @Test
    void rejectsNullInput() {
        assertThrows(
                NullPointerException.class,
                () -> ComplexNumberParser.getComponents(null)
        );
    }


    private static void assertComponents(
            String input,
            double expectedReal,
            double expectedImaginary
    ) {
        DoublePair actual = ComplexNumberParser.getComponents(input);

        assertEquals(expectedReal, actual.first(), TOL,
                () -> "Unexpected real component for: " + input);

        assertEquals(expectedImaginary, actual.second(), TOL,
                () -> "Unexpected imaginary component for: " + input);
    }


    private static void assertInvalid(String input) {
        assertThrows(
                Flag4jParsingException.class,
                () -> ComplexNumberParser.getComponents(input),
                () -> "Expected parsing failure for: " + input
        );
    }
}