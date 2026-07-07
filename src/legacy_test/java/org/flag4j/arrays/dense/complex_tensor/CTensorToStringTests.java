package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CTensorToStringTests {
    static Shape aShape;
    static Complex128[] aEntries;
    static CTensor A;
    static String exp;

    @BeforeAll
    static void setup() {
        PrintOptions.resetAll();
        aShape = new Shape(2, 3, 1, 2);
        aEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(235.61, 1.45), new Complex128(0, -0.00024),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(1.345, -781.2),
                new Complex128(0.014, -2.45),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(51.0, 24.56),  new Complex128(6.1, -0.03),  new Complex128(-0.00014, 1.34),};
        A = new CTensor(aShape, aEntries);
    }


    @AfterEach
    void cleanup() {
        PrintOptions.resetAll();
    }


    @Test
    void toStringTestCase() {

        // ---------------------- sub-case 1 ----------------------
        exp = "shape: (2, 3, 1, 2)\n" +
                "[ 1.4415-0.0245i  235.61+1.45i  -2.4E-4i  1  -85.1+9.234i  1.345-781.2i  0.014-2.45i  -140  1.5i  ...  -1.4E-4+1.34i ]";
        assertEquals(exp, A.toString());

        // ---------------------- sub-case 2 ----------------------
        PrintOptions.setMaxColumns(15);
        PrintOptions.setPrecision(2);
        PrintOptions.setCentering(false);
        exp = "shape: (2, 3, 1, 2)\n" +
                "[1.44-0.02i  235.61+1.45i  0  1  -85.1+9.23i  1.35-781.2i  0.01-2.45i  -140  1.5i  51+24.56i  6.1-0.03i  1.34i  ]";
        assertEquals(exp, A.toString());
    }
}
