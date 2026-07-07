package org.flag4j.arrays.dense.vector;


import org.flag4j.arrays.dense.Vector;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorToStringTest {

    static double[] aEntries;
    static Vector A;
    static String exp;

    @BeforeAll
    static void setup() {
        PrintOptions.resetAll();
        aEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        A = new Vector(aEntries);
    }


    @AfterEach
    void cleanup() {
        PrintOptions.resetAll();
    }


    @Test
    void toStringTestCase() {
        PrintOptions.resetAll();
        // ---------------------- sub-case 1 ----------------------
        exp = "shape: (12)\n" +
                "[ 1  -1.4133  113.4  0.4  11.3  445  133.445  9.8  13384  ...  12 ]";
        assertEquals(exp, A.toString());

        // ---------------------- sub-case 2 ----------------------
        PrintOptions.setMaxColumns(15);
        PrintOptions.setPrecision(2);
        PrintOptions.setCentering(false);
        exp = "shape: (12)\n" +
                "[1  -1.41  113.4  0.4  11.3  445  133.45  9.8  13384  -993.44  11  12  ]";
        assertEquals(exp, A.toString());
    }
}
