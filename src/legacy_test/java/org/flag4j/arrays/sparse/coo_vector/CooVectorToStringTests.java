package org.flag4j.arrays.sparse.coo_vector;

import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooVectorToStringTests {

    String exp;
    static CooVector a;

    @BeforeAll
    static void setup() {
        double[] nze = {1.34, 525, 63.7, -0.0234};
        int size = 3056;
        int[] indices = {1, 567, 1567, 2506};

        a = new CooVector(size, nze, indices);
    }


    @AfterAll
    static void cleanup() {
        // Reset all print options.
        PrintOptions.resetAll();
    }


    @Test
    void toStringTestCase() {
        // --------------------- sub-case 1 ---------------------
        exp = """
                shape: (3056)
                nnz: 4
                Non-zero data: [ 1.34  525  63.7  -0.0234 ]
                Indices: [ 1  567  1567  2506 ]""";
        assertEquals(exp, a.toString());

        // --------------------- sub-case 2 ---------------------
        PrintOptions.setCentering(false);
        PrintOptions.setMaxColumns(2);
        PrintOptions.setPrecision(2);
        exp = """
                shape: (3056)
                nnz: 4
                Non-zero data: [1.34  ...  -0.02  ]
                Indices: [1  ...  2506  ]""";
        assertEquals(exp, a.toString());
    }
}
