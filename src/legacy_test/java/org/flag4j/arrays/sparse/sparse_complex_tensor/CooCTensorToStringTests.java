package org.flag4j.arrays.sparse.sparse_complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCTensorToStringTests {

    static CooCTensor A;
    static Shape aShape;
    static Complex128[] aEntries;
    static int[][] aIndices;

    String exp;

    @BeforeAll
    static void setUp() {
        // Ensure print options are set to default before any test is run.
        PrintOptions.resetAll();
    }


    @AfterEach
    void tearDown() {
        // Ensure print options are reset after each test is run.
        PrintOptions.resetAll();
    }


    @Test
    void cooCTensorToStringTests() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(5, 4, 2, 3, 15);
        aEntries = new Complex128[]{new Complex128(0.8103, 0.0203), new Complex128(0.5684, 0.4151), new Complex128(0.9044, 0.8734), new Complex128(0.201, 0.7032), new Complex128(0.9682, 0.2723), new Complex128(0.4699, 0.8203), new Complex128(0.3871, 0.3395), new Complex128(0.7851, 0.3768), new Complex128(0.2315, 0.7695), new Complex128(0.8333, 0.8837), new Complex128(0.0398, 0.559), new Complex128(0.0405, 0.9707), new Complex128(0.488, 0.8343), new Complex128(0.2441, 0.7806), new Complex128(0.3995, 0.6793), new Complex128(0.3689, 0.6126), new Complex128(0.0767, 0.9631), new Complex128(0.8007, 0.4023)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 1, 0, 0, 7},
                {0, 2, 1, 2, 4},
                {0, 3, 0, 0, 11},
                {0, 3, 1, 0, 10},
                {0, 3, 1, 2, 6},
                {1, 1, 1, 0, 7},
                {2, 0, 0, 0, 8},
                {2, 0, 0, 0, 10},
                {2, 2, 0, 2, 12},
                {2, 2, 1, 1, 6},
                {3, 0, 0, 0, 2},
                {3, 0, 1, 1, 1},
                {4, 0, 0, 2, 10},
                {4, 0, 1, 0, 7},
                {4, 2, 1, 1, 8},
                {4, 3, 0, 0, 6},
                {4, 3, 1, 2, 13}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        exp = "Shape: (5, 4, 2, 3, 15)\n" +
                "nnz: 18\n" +
                "Non-zero Entries: [ 0.8103+0.0203i  0.5684+0.4151i  0.9044+0.8734i  0.201+0.7032i  0.9682+0.2723i  0.4699+0.8203i  0.3871+0.3395i  0.7851+0.3768i  0.2315+0.7695i  ...  0.8007+0.4023i ]\n" +
                "Non-zero Indices: [ [ 0  0  0  0  0 ]  \n" +
                "                    [ 0  1  0  0  7 ]  \n" +
                "                    [ 0  2  1  2  4 ]  \n" +
                "                    [ 0  3  0  0  11 ]  \n" +
                "                    [ 0  3  1  0  10 ]  \n" +
                "                    [ 0  3  1  2  6 ]  \n" +
                "                    [ 1  1  1  0  7 ]  \n" +
                "                    [ 2  0  0  0  8 ]  \n" +
                "                    [ 2  0  0  0  10 ]  \n" +
                "                     ...  \n" +
                "                    [ 4  3  1  2  13 ]  ]";
        assertEquals(exp, A.toString());

        // ----------------------- sub-case 2 -----------------------
        PrintOptions.setMaxRows(15);
        PrintOptions.setMaxColumns(3);
        PrintOptions.setPrecision(3);

        exp = """
                Shape: (5, 4, 2, 3, 15)
                nnz: 18
                Non-zero Entries: [ 0.81+0.02i  0.568+0.415i  ...  0.801+0.402i ]
                Non-zero Indices: [ [ 0  0  ...  0 ] \s
                                    [ 0  1  ...  7 ] \s
                                    [ 0  2  ...  4 ] \s
                                    [ 0  3  ...  11 ] \s
                                    [ 0  3  ...  10 ] \s
                                    [ 0  3  ...  6 ] \s
                                    [ 1  1  ...  7 ] \s
                                    [ 2  0  ...  8 ] \s
                                    [ 2  0  ...  10 ] \s
                                    [ 2  2  ...  12 ] \s
                                    [ 2  2  ...  6 ] \s
                                    [ 3  0  ...  2 ] \s
                                    [ 3  0  ...  1 ] \s
                                    [ 4  0  ...  10 ] \s
                                     ... \s
                                    [ 4  3  ...  13 ]  ]""";
        assertEquals(exp, A.toString());
    }
}
