package org.flag4j.util;

import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class ValidateParametersTests {
    Shape shape1, shape2;

    @Test
    void matMultTestCase() {
        // ------------ sub-case 1 ------------
        shape1 = new Shape(5, 5);
        shape2 = new Shape(5, 5);

        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        // ------------ sub-case 2 ------------
        shape1 = new Shape(5, 1);
        shape2 = new Shape(1, 5);

        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        // ------------ sub-case 3 ------------
        shape1 = new Shape(1, 5);
        shape2 = new Shape(1, 5);

        assertThrows(LinearAlgebraException.class,
                ()-> ValidateParameters.ensureMatMultShapes(shape1, shape2));

        // ------------ sub-case 4 ------------
        shape1 = new Shape(6, 114);
        shape2 = new Shape(114, 6);

        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        // ------------ sub-case 5 ------------
        shape1 = new Shape(112, 1, 1);
        shape2 = new Shape(113);
        assertThrows(LinearAlgebraException.class,
                ()-> ValidateParameters.ensureMatMultShapes(shape1, shape2));
    }
}
