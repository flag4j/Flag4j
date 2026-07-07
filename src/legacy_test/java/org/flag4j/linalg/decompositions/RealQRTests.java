package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.qr.RealQR;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealQRTests {

    double[][] aEntries, expQEntries, expREntries;
    Matrix A, expQ, expR;
    RealQR QR;

    @Test
    void fullTestCase() {
        // Tests account for numerical loss of precision.
        QR = new RealQR();

        // --------------------------- sub-case 1 ---------------------------
        aEntries = new double[][]
                {{0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1},
                {1, 0, 0, 0}};
        A = new Matrix(aEntries);
        expQEntries = new double[][]
                {{0.0, -1.0, 0.0, 0.0},
                {0.0, 0.0, -1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
                {-1.0, 0.0, 0.0, 0.0}};
        expQ = new Matrix(expQEntries);
        expREntries = new double[][]
                {{-1.0, 0.0, 0.0, 0.0},
                {0.0, -1.0, 0.0, 0.0},
                {0.0, 0.0, -1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0}};
        expR = new Matrix(expREntries);

        QR.decompose(A);

        assertEquals(expQ, QR.getQ());
        assertEquals(expR, QR.getR());

        // --------------------------- sub-case 2 ---------------------------
        aEntries = new double[][]{{1.0, 5.6, -9.355, 215.0},
                {56.0, 1.0, 15.2, 14.0},
                {2.4, -0.00025, 1.0, 0.0},
                {1.0, 49.4, 106.2, -8.5}};
        A = new Matrix(aEntries);
        expQEntries = new double[][]{
                {-0.017835090116782304, 0.11199486596156988, 0.9933393762756556, 0.020394683107807277},
                {-0.9987650465398173, -0.019667899323835336, -0.014829165852483617, -0.043146858363307505},
                {-0.04280421628027788, -0.0017104833343519052, -0.021083705815527258, 0.9988595249906157},
                {-0.017835090116782453, 0.9935126561756528, -0.11230519457930435, -0.0014334747400640763}
        };
        expQ = new Matrix(expQEntries);
        expREntries = new double[][]{
                {-56.06924290553601, -1.9796843019087813, -16.951272226045297, -17.665656760672988},
                {0.0, 49.687028992759025, 104.16266956172718, 15.358688013710776},
                {0.0, 0.0, -21.465988556154155, 214.31495173125523},
                {0.0, 0.0, 0.0, 3.79298538638281}
        };
        expR = new Matrix(expREntries);

        QR.decompose(A);

        assertEquals(expQ, QR.getQ());
        assertEquals(expR, QR.getR());
    }
}
