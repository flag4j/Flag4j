package org.flag4j.util;

import org.flag4j.arrays.Shape;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ErrorMessagesTests {
    Shape s1, s2;
    String expMsg;


    @Test
    void EqualShapeErrMsgTestCase() {
        // --------- sub-case 1 ---------
        s1 = new Shape(2);
        s2 = new Shape(5);
        expMsg = String.format("Expecting tensors to have the same shape but got %s and %s.",
                "(2)", "(5)");

        assertEquals(expMsg, ErrorMessages.equalShapeErrMsg(s1, s2));


        // --------- sub-case 2 ---------
        s1 = new Shape(1, 2, 3, 4);
        s2 = new Shape(4, 3, 2, 1);
        expMsg = String.format("Expecting tensors to have the same shape but got %s and %s.",
                "(1, 2, 3, 4)", "(4, 3, 2, 1)");

        assertEquals(expMsg, ErrorMessages.equalShapeErrMsg(s1, s2));
    }


    @Test
    void matMultShapeErrMsgTestCase() {
        // --------- sub-case 1 ---------
        s1 = new Shape(10, 5);
        s2 = new Shape(14, 4);
        expMsg = String.format("Cannot multiply matrices/vector with shapes (10, 5) and (14, 4).");

        assertEquals(expMsg, ErrorMessages.matMultShapeErrMsg(s1, s2));
    }


    @Test
    void negativeDimErrMsgTestCase() {
        int[] dims = {-1, 1};
        expMsg = String.format("Shape dimensions must be non-negative but got shape %s.",
                Arrays.toString(dims));

        assertEquals(expMsg, ErrorMessages.negativeDimErrMsg(dims));
    }


    @Test
    void utilityClassErrMsgTestCase() {
        expMsg = "Utility class cannot be instantiated.";

        assertEquals(expMsg, ErrorMessages.getUtilityClassErrMsg());
    }
}
