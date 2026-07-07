package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CsrMatrixTransposeTests {

    static CsrMatrix A;
    static CsrMatrix exp;
    static Matrix denseA;
    static double[][] aDenseEntries;


    private static void makeMatrices() {
        denseA = new Matrix(aDenseEntries);
        exp = denseA.T().toCsr();
        A = denseA.toCsr();
    }


    @Test
    void transposeTest() {
        // --------------------- sub-case 1 ----------------------
        aDenseEntries = new double[][]{
                {0.245, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 9, 0, 0, 0},
                {0, 0, 1.14, 0, 0, 0, 0, 24.1},
                {0, 0, 0, 12, 0, 0, -93.51134, 0},
                {0, 0, 0, 15.14, 24.15, 0, 0, -0.114}
        };
        makeMatrices();
        assertEquals(exp, A.T());

        // --------------------- sub-case 2 ----------------------
        aDenseEntries = new double[][]{
                {0, 0, 1, 0},
                {0, -25, 0, 0},
                {0, 4, 0, 0},
                {0, 12, 0, 0},
                {0, 14.15, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {-1, 92, 15.2, 7},
                {0, 0, 0, 8.13}
        };
        makeMatrices();
        assertEquals(exp, A.T());
    }


    @Test
    void isSymmetricTests() {
        // -------------------- sub-case 1 ---------------------
        aDenseEntries = new double[5][5];
        aDenseEntries[0][0] = 1;
        aDenseEntries[2][1] = -491.3;
        aDenseEntries[1][2] = -491.3;
        aDenseEntries[3][4] = 5.3;
        aDenseEntries[4][3] = 5.3;
        aDenseEntries[3][3] = 9;
        makeMatrices();

        assertTrue(A.isSymmetric());

        // -------------------- sub-case 2 ---------------------
        aDenseEntries = new double[5][6];
        aDenseEntries[0][0] = 1;
        aDenseEntries[2][1] = -491.3;
        aDenseEntries[1][2] = -491.3;
        aDenseEntries[3][4] = 5.3;
        aDenseEntries[4][3] = 5.3;
        aDenseEntries[3][3] = 9;
        makeMatrices();

        assertFalse(A.isSymmetric());

        // -------------------- sub-case 1 ---------------------
        aDenseEntries = new double[415][415];
        aDenseEntries[0][0] = 1;
        aDenseEntries[2][1] = -491.3;
        aDenseEntries[1][2] = -491.3;
        aDenseEntries[3][4] = 5.3;
        aDenseEntries[4][3] = 5.3;
        aDenseEntries[3][3] = 9;
        aDenseEntries[45][2] = 10.3;
        aDenseEntries[2][45] = 10.3;
        aDenseEntries[13][44] = -94.2;
        aDenseEntries[44][13] = -94.2;
        aDenseEntries[255][255] = 1.3;
        aDenseEntries[304][34] = 849.1;
        aDenseEntries[34][304] = 849.1;
        aDenseEntries[304][55] = -8.4;
        aDenseEntries[55][304] = -8.4;
        aDenseEntries[400][400] = 84;
        aDenseEntries[402][0] = -29.32;
        aDenseEntries[0][402] = -29.32;

        makeMatrices();

        assertTrue(A.isSymmetric());
    }
}
