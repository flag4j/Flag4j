package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorToStringTests {

    static Shape aShape;
    static double[] aEntries;
    static Tensor A;
    static String exp;

    @BeforeAll
    static void setup() {
        PrintOptions.resetAll();
        aShape = new Shape(2, 3, 1, 2);
        aEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        A = new Tensor(aShape, aEntries);
    }


    @AfterEach
    void cleanup() {
        PrintOptions.resetAll();
    }


    @Test
    void toStringTestCase() {
        PrintOptions.resetAll();
        // ---------------------- sub-case 1 ----------------------
        exp = "shape: (2, 3, 1, 2)\n" +
                "[ 1  -1.4133  113.4  0.4  11.3  445  133.445  9.8  13384  ...  12 ]";
        assertEquals(exp, A.toString());

        // ---------------------- sub-case 2 ----------------------
        PrintOptions.setMaxColumns(15);
        PrintOptions.setPrecision(2);
        PrintOptions.setCentering(false);
        exp = "shape: (2, 3, 1, 2)\n" +
                "[1  -1.41  113.4  0.4  11.3  445  133.45  9.8  13384  -993.44  11  12  ]";
        assertEquals(exp, A.toString());
    }
}
