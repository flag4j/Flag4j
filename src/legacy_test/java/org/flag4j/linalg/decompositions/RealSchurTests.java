package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.schur.RealSchur;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class RealSchurTests {
    static final long SEED = 0xC0DE;

    static Shape aShape;
    static double[] aData;
    static Matrix a;
    static Matrix U;
    static Matrix T;
    static CMatrix[] TUcm;

    static RealSchur schur;
    static RealSchur schurU;

    @BeforeAll
    static void setUp() {
        schurU = new RealSchur(true, SEED);
        schur = new RealSchur(false, SEED);
    }

    @Test
    void realSchurTests() {
//        // ----------------- sub-case 1 -----------------
//        aShape = new Shape(11, 11);
//        aData = new double[]{1e-08, 0.02, 0.0, 5e-05, 30000.0, 0.0, -1000.0, 0.0, 70.0, 0.0, 9000000.0, 5000.0, 100.0, -0.002, 0.0,
//                0.0, 0.0, 9e-09, 0.0, 0.0, -300000.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-09,
//                100000000.0, -10.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, -0.0003, 40000.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -400.0, 0.0, 10.0,
//                200.0, 0.0, 200000.0, 0.0, 1e-05, 0.0, 0.0, 0.0, 0.0, 5e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000000000.0, 0.0,
//                0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.005, 0.0002, 1e-07, 0.0, 0.0, 10.0, 0.0, -90000.0, 0.0, 3e-06, 0.0,
//                0.0, 0.0, 300000000.0, 0.0, 0.0, 0.0, 0.0005, 0.0, 0.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 1e-08, 200000.0, 7000.0, 0.0,
//                0.0, 0.2, 0.0, 0.0, 1000.0, 0.0, 0.0, -400.0, 0.01};
//        a = new Matrix(aShape, aData);
//
//        // TODO: TEMP FOR TESTING
//        PrintOptions.setMaxRowsCols(100);
//        TestHelpers.printAsJavaArray(a);
//        // TODO: END OF TEMP
//
//        schurU.decompose(a);
//        U = schurU.getU();
//        T = schurU.getT();
//        TUcm = schurU.real2ComplexSchur();
//
//        assertTrue(TUcm[0].isTriU());
//        assertTrue(TUcm[1].isOrthogonal());
//        CustomAssertions.assertEquals(a, TUcm[1].mult(TUcm[0]).mult(TUcm[1].H()).toReal(), 1.0e-12);
    }
}
