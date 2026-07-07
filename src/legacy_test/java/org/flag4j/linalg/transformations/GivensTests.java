package org.flag4j.linalg.transformations;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GivensTests {

    double[] vEntriesReal;
    Complex128[] vEntriesComplex;

    Vector vReal;
    CVector vComplex;

    double[][] expEntriesReal;
    Matrix expReal;

    Complex128[][] expEntriesComplex;
    CMatrix expComplex;


    @Test
    void generalRotatorTestCase() {
        int size, i, j;
        double theta;

        // ------------------- sub-case 1 -------------------
        size = 105;
        j = 4;
        i = 91;
        theta = Math.PI/4.0;
        expReal = Matrix.I(size);
        expReal.set(Math.cos(theta), i, i);
        expReal.set(Math.cos(theta), j, j);
        expReal.set(Math.sin(theta), i, j);
        expReal.set(-Math.sin(theta), j, i);

        assertEquals(expReal, Givens.getGeneralRotator(size, i, j, theta));

        // ------------------- sub-case 2 -------------------
        size = 3;
        j = 1;
        i = 2;
        theta = 0.0;
        expReal = Matrix.I(size);
        expReal.set(Math.cos(theta), i, i);
        expReal.set(Math.cos(theta), j, j);
        expReal.set(Math.sin(theta), i, j);
        expReal.set(-Math.sin(theta), j, i);

        assertEquals(expReal, Givens.getGeneralRotator(size, i, j, theta));

        // ------------------- sub-case 3 -------------------
        assertThrows(IllegalArgumentException.class, ()-> Givens.getGeneralRotator(5, 3, 3, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getGeneralRotator(5, 1, 6, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getGeneralRotator(5, 6, 0, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getGeneralRotator(5, -1, 2, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getGeneralRotator(5, 1, -5, 2));
    }


    @Test
    void realRotatorTestCase() {
        int i;

        // ------------------- sub-case 1 -------------------
        i = 4;
        vEntriesReal = new double[]{1.56, 1.3567, -0.02456, 103.6, -992.255, 88.156};
        vReal = new Vector(vEntriesReal);

        expEntriesReal = new double[][]{
                {0.001572174564045712, 0.0, 0.0, 0.0, -0.9999987641328064, 0.0},
                {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                {0.9999987641328064, 0.0, 0.0, 0.0, 0.001572174564045712, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
        };
        expReal = new Matrix(expEntriesReal);

        assertEquals(expReal, Givens.getRotator(vReal, i));

        // ------------------- sub-case 2 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getRotator(vReal, -1));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getRotator(vReal, vReal.size));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getRotator(vReal, vReal.size + 3));
    }


    @Test
    void complexRotatorTestCase() {
        int i;

        // ------------------- sub-case 1 -------------------
        i = 1;
        vEntriesComplex = new Complex128[]{new Complex128(1.456), new Complex128(-2, 15.6),
                new Complex128(2.6, -0.2), Complex128.ZERO};
        vComplex = new CVector(vEntriesComplex);

        expEntriesComplex = new Complex128[][]{
                {new Complex128(0.09095028651755654, 0.0), new Complex128(-0.12493171224939086, 0.9744673555452487), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.12493171224939086, -0.9744673555452487), new Complex128(0.09095028651755654, -0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(1.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(1.0, 0.0)}
        };
        expComplex = new CMatrix(expEntriesComplex);

        assertEquals(expComplex, Givens.getRotator(vComplex, i));

        // ------------------- sub-case 2 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getRotator(vComplex, -1));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getRotator(vComplex, vComplex.size));
        assertThrows(IndexOutOfBoundsException.class, ()-> Givens.getRotator(vComplex, vComplex.size + 3));
    }


    @Test
    void real2x2RotatorTestCase() {
        // ------------------- sub-case 1 -------------------
        vEntriesReal = new double[]{1.56, 1.3567};
        vReal = new Vector(vEntriesReal);

        expEntriesReal = new double[][]{
                {0.7545628263597614, 0.6562278118732616},
                {-0.6562278118732616, 0.7545628263597614}
        };
        expReal = new Matrix(expEntriesReal);

        assertEquals(expReal, Givens.get2x2Rotator(vReal));

        // ------------------- sub-case 2 -------------------
        vEntriesReal = new double[]{1.56, 0};
        vReal = new Vector(vEntriesReal);

        expEntriesReal = new double[][]{
                {1.0, 0.0},
                {-0.0, 1.0}
        };
        expReal = new Matrix(expEntriesReal);
        
        assertEquals(expReal, Givens.get2x2Rotator(vReal));

        // ------------------- sub-case 3 -------------------
        vEntriesReal = new double[]{0, -9.3};
        vReal = new Vector(vEntriesReal);

        expEntriesReal = new double[][]{
                {0.0, -1.0},
                {1.0, 0.0}
        };
        expReal = new Matrix(expEntriesReal);

        assertEquals(expReal, Givens.get2x2Rotator(vReal));

        // ------------------- sub-case 4 -------------------
        vEntriesReal = new double[]{0, 0};
        vReal = new Vector(vEntriesReal);

        expEntriesReal = new double[][]{
                {1.0, 0.0},
                {-0.0, 1.0}
        };
        expReal = new Matrix(expEntriesReal);

        assertEquals(expReal, Givens.get2x2Rotator(vReal));

        // ------------------- sub-case 5 -------------------
        assertThrows(IllegalArgumentException.class, ()-> Givens.get2x2Rotator(new Vector(1)));
        assertThrows(IllegalArgumentException.class, ()-> Givens.get2x2Rotator(new Vector(5)));
    }


    @Test
    void complex2x2RotatorTestCase() {
        // ------------------- sub-case 1 -------------------
        vEntriesComplex = new Complex128[]{new Complex128(2.56, -9.53), new Complex128(3.6, 0.00134)};
        vComplex = new CVector(vEntriesComplex);

        expEntriesComplex = new Complex128[][]{
                {new Complex128(0.24371614282534812, 0.9072714223146748), new Complex128(0.3427258258481458, 1.2757016851014318E-4)},
                {new Complex128(-0.3427258258481458, -1.2757016851014318E-4), new Complex128(0.24371614282534812, -0.9072714223146748)}
        };
        expComplex = new CMatrix(expEntriesComplex);

        assertEquals(expComplex, Givens.get2x2Rotator(vComplex));

        // ------------------- sub-case 2 -------------------
        assertThrows(IllegalArgumentException.class, ()-> Givens.get2x2Rotator(new CVector(1)));
        assertThrows(IllegalArgumentException.class, ()-> Givens.get2x2Rotator(new CVector(5)));
    }
}
