package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrCMatrixToStringTests {
    static CsrCMatrix A;
    static Shape aShape;
    static Complex128[] nnz;
    static int[] colIndices;
    static int[] rowIndices;
    String exp = "";

    static void build() {
        A = new CooCMatrix(aShape, nnz, rowIndices, colIndices).toCsr();
    }

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
    void toStringTests() {
        // ------------------ sub-case 1 ------------------
        aShape = new Shape(150, 2256);
        nnz = new Complex128[]{new Complex128(1.325, 9.2), new Complex128(-6, 1), new Complex128(23),
                new Complex128(0, 34615), new Complex128(0, -1), new Complex128(25, 1)};
        rowIndices = new int[]{0, 0, 1, 121, 141, 149};
        colIndices = new int[]{150, 2200, 14, 15, 892, 156};
        exp = """
                shape: (150, 2256)
                nnz: 6
                Non-zero data: [ 1.325+9.2i  -6+i  23  34615i  -i  25+i ]
                Row Pointers: [ 0  2  3  3  3  3  3  3  3  ...  6 ]
                Col Indices: [ 150  2200  14  15  892  156 ]""";
        build();

        assertEquals(exp, A.toString());

        // ------------------ sub-case 2 ------------------
        aShape = new Shape(12, 12);
        nnz = new Complex128[]{
                new Complex128(1), new Complex128(2), new Complex128(3), new Complex128(4), new Complex128(5),
                new Complex128(6), new Complex128(7), new Complex128(8), new Complex128(9), new Complex128(10),
                new Complex128(11), new Complex128(12), new Complex128(13), new Complex128(14), new Complex128(15), 
                new Complex128(16), new Complex128(17), new Complex128(18), new Complex128(19), new Complex128(20), 
                new Complex128(21), new Complex128(22), new Complex128(23), new Complex128(24), new Complex128(25)};
        rowIndices = new int[]{0, 0, 0, 0, 1, 1, 1,  2, 2, 2, 2, 2, 2,  2,  3, 5, 5, 5,  8, 8, 8, 8, 9, 9, 9};
        colIndices = new int[]{0, 1, 2, 9, 5, 6, 11, 0, 2, 7, 8, 9, 10, 11, 6, 2, 9, 11, 0, 1, 5, 7, 8, 9, 11};
        exp = """
                shape: (12, 12)
                nnz: 25
                Non-zero data: [ 1  2  3  4  5  6  7  8  9  ...  25 ]
                Row Pointers: [ 0  4  7  14  15  15  18  18  18  ...  25 ]
                Col Indices: [ 0  1  2  9  5  6  11  0  2  ...  11 ]""";
        build();

        assertEquals(exp, A.toString());
    }
}
