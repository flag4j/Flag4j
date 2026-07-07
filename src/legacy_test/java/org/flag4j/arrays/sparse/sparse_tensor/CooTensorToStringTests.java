package org.flag4j.arrays.sparse.sparse_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooTensorToStringTests {

    static CooTensor A;
    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;

    String exp;

    @AfterEach
    void reset() {
        PrintOptions.resetAll();
    }

    @Test
    void cooTensorToStringTests() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new double[]{0.3369876012099134, -1.2778944176165061, 0.3003526634585191, -2.871698370528301, -0.7476330328637798, 0.0005862743350540067, 0.27050784380653264, -0.8804057494085602, 0.3433759561302188, -0.6271337465977211, -0.535101554634142, -1.9109729935630098, 0.7218896402785697, 0.8404528180001959, 0.5203214826154133, 0.3371173022767966, -0.6726729311708143, 0.08353784066932919};
        aIndices = new int[][]{
                {0, 0, 1, 0, 1},
                {0, 1, 1, 0, 4},
                {0, 2, 0, 0, 0},
                {0, 2, 1, 0, 1},
                {0, 3, 0, 0, 2},
                {1, 1, 0, 0, 0},
                {1, 1, 1, 0, 0},
                {1, 1, 1, 0, 2},
                {1, 2, 1, 0, 0},
                {1, 2, 1, 0, 4},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 4},
                {2, 0, 1, 0, 4},
                {2, 1, 0, 0, 0},
                {2, 1, 1, 0, 1},
                {2, 1, 1, 0, 3},
                {2, 1, 1, 0, 4},
                {2, 3, 0, 0, 2}};
        A = new CooTensor(aShape, aEntries, aIndices);

        exp = """
                Shape: (3, 4, 2, 1, 5)
                nnz: 18
                Non-zero Entries: [ 0.3369876  -1.27789442  0.30035266  -2.87169837  -0.74763303  5.8627E-4  0.27050784  -0.88040575  0.34337596  ...  0.08353784 ]
                Non-zero Indices: [ [ 0  0  1  0  1 ] \s
                                    [ 0  1  1  0  4 ] \s
                                    [ 0  2  0  0  0 ] \s
                                    [ 0  2  1  0  1 ] \s
                                    [ 0  3  0  0  2 ] \s
                                    [ 1  1  0  0  0 ] \s
                                    [ 1  1  1  0  0 ] \s
                                    [ 1  1  1  0  2 ] \s
                                    [ 1  2  1  0  0 ] \s
                                     ... \s
                                    [ 2  3  0  0  2 ]  ]""";
        assertEquals(exp, A.toString());

        // ----------------------- sub-case 2 -----------------------
        PrintOptions.setMaxRows(15);
        PrintOptions.setMaxColumns(3);
        PrintOptions.setPrecision(3);

        exp = """
                Shape: (3, 4, 2, 1, 5)
                nnz: 18
                Non-zero Entries: [ 0.337  -1.278  ...  0.084 ]
                Non-zero Indices: [ [ 0  0  ...  1 ] \s
                                    [ 0  1  ...  4 ] \s
                                    [ 0  2  ...  0 ] \s
                                    [ 0  2  ...  1 ] \s
                                    [ 0  3  ...  2 ] \s
                                    [ 1  1  ...  0 ] \s
                                    [ 1  1  ...  0 ] \s
                                    [ 1  1  ...  2 ] \s
                                    [ 1  2  ...  0 ] \s
                                    [ 1  2  ...  4 ] \s
                                    [ 2  0  ...  0 ] \s
                                    [ 2  0  ...  4 ] \s
                                    [ 2  0  ...  4 ] \s
                                    [ 2  1  ...  0 ] \s
                                     ... \s
                                    [ 2  3  ...  2 ]  ]""";
        assertEquals(exp, A.toString());
    }
}
