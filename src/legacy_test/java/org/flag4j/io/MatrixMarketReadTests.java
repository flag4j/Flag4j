package org.flag4j.io;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixMarketReadTests {

    @Test
    void loadRealDenseTestCase() throws IOException {
        String fileName;
        Matrix exp;
        Matrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/items/matrix_market/array/ReDe3x3.mtx";
        exp = new Matrix(MatrixMarketMatrices.ReDe3x3);
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 2 ----------------
        fileName = "./src/test/items/matrix_market/array/ReDe2x4.mtx";
        exp = new Matrix(MatrixMarketMatrices.ReDe2x4);
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 3 ----------------
        fileName = "./src/test/items/matrix_market/array/ReDe3x2.mtx";
        exp = new Matrix(MatrixMarketMatrices.ReDe3x2);
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 4 ----------------
        fileName = "./src/test/items/matrix_market/array/ReDe49x15.mtx";

        exp = new Matrix(MatrixMarketMatrices.ReDe49x15);
        act = (Matrix) mmParser.read(fileName);
        assertEquals(exp, act);
    }


    @Test
    void loadComplexDenseTestCase() throws IOException {
        String fileName;
        CMatrix exp;
        CMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/items/matrix_market/array/CmDe3x3.mtx";
        exp = new CMatrix(MatrixMarketMatrices.CmDe3x3);
        act = (CMatrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 2 ----------------
        fileName = "./src/test/items/matrix_market/array/CmDe5x2.mtx";
        exp = new CMatrix(MatrixMarketMatrices.CmDe5x2);
        act = (CMatrix) mmParser.read(fileName);

        assertEquals(exp, act);
    }


    @Test
    void loadRealCooTestCase() throws IOException {
        String fileName;
        CooMatrix exp;
        CooMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/items/matrix_market/coordinate/ReCoo5x5_8.mtx";
        exp = MatrixMarketMatrices.ReCoo5x5_8;
        act = (CooMatrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 2 ----------------
        fileName = "./src/test/items/matrix_market/coordinate/jgl009.mtx";
        exp = MatrixMarketMatrices.jgl009;
        act = (CooMatrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 3 ----------------
        fileName = "./src/test/items/matrix_market/coordinate/fidapm05.mtx";
        exp = MatrixMarketMatrices.fidapm05;
        act = (CooMatrix) mmParser.read(fileName);

        assertEquals(exp, act);
    }


    @Test
    void loadRealDenseSymmTestCase() throws IOException {
        String fileName;
        Matrix exp;
        Matrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/items/matrix_market/array/ReDeSymm5x5.mtx";
        exp = MatrixMarketMatrices.ReDeSymm5x5;
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);
    }


    @Test
    void loadComplexDenseHermTestCase() throws IOException {
        String fileName;
        CMatrix exp;
        CMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/items/matrix_market/array/CmDeHerm5x5.mtx";
        exp = MatrixMarketMatrices.CmDeHerm5x5;
        act = (CMatrix) mmParser.read(fileName);

        assertEquals(exp, act);
    }


    @Test
    void loadPatternCooSymmTestCase() throws IOException {
        String fileName;
        CooMatrix exp;
        CooMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/items/matrix_market/coordinate/can___24.mtx";
        exp = MatrixMarketMatrices.can___24;
        act = (CooMatrix) mmParser.read(fileName);

        assertEquals(exp, act);
    }
}
