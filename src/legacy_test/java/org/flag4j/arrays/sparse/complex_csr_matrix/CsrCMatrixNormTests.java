package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.linalg.MatrixNorms;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixNormTests {

    @Test
    void csrLpqNorms() {
        Shape aShape;
        Complex128[] aData;
        int[] aRowPointers, aColIndices;
        CsrCMatrix a;
        double exp, p, q;

        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(32, 32);
        aData = new Complex128[]{new Complex128(0.02477, 0.76398), new Complex128(0.40479, 0.85967), new Complex128(0.4548, 0.38816), new Complex128(0.91506, 0.39483), new Complex128(0.57948, 0.66513), new Complex128(0.66353, 0.70929), new Complex128(0.43548, 0.52384), new Complex128(0.34137, 0.33486), new Complex128(0.24652, 0.1701), new Complex128(0.95208, 0.50698)};
        aRowPointers = new int[]{0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 6, 6, 6, 7, 9, 9, 9, 10, 10};
        aColIndices = new int[]{2, 20, 11, 15, 22, 15, 21, 17, 24, 28};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 1;
        q = 1;
        exp = 7.700099282600347;

        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(32, 32);
        aData = new Complex128[]{new Complex128(0.49025, 0.29208), new Complex128(0.17466, 0.43675), new Complex128(0.51197, 0.38465), new Complex128(0.4191, 0.46123), new Complex128(0.43736, 0.70804), new Complex128(0.49982, 0.30753), new Complex128(0.11769, 0.52149), new Complex128(0.05334, 0.41568), new Complex128(0.07662, 0.35198), new Complex128(0.63588, 0.99672)};
        aRowPointers = new int[]{0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 5, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10};
        aColIndices = new int[]{13, 31, 7, 24, 16, 9, 12, 2, 12, 13};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 1;
        q = 2;
        exp = 2.4710068494660162;

        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(32, 32);
        aData = new Complex128[]{new Complex128(0.72585, 0.65), new Complex128(0.07972, 0.38903), new Complex128(0.20433, 0.35512), new Complex128(0.52162, 0.26212), new Complex128(0.97118, 0.80551), new Complex128(0.90915, 0.03623), new Complex128(0.81902, 0.84555), new Complex128(0.84593, 0.62298), new Complex128(0.876, 0.03699), new Complex128(0.97226, 0.77725)};
        aRowPointers = new int[]{0, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 7, 7, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10};
        aColIndices = new int[]{25, 31, 5, 31, 4, 7, 8, 26, 12, 9};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 2;
        q = 1;
        exp = 8.611013354258432;
        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 4 -----------------------
        aShape = new Shape(32, 32);
        aData = new Complex128[]{new Complex128(0.27792, 0.15589), new Complex128(0.66318, 0.34308), new Complex128(0.52266, 0.94122), new Complex128(0.03301, 0.23813), new Complex128(0.29464, 0.79951), new Complex128(0.89043, 0.34081), new Complex128(0.60507, 0.00875), new Complex128(0.86107, 0.62376), new Complex128(0.48382, 0.17035), new Complex128(0.74956, 0.71469)};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 4, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10};
        aColIndices = new int[]{24, 30, 5, 1, 0, 9, 4, 27, 30, 5};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 4.12;
        q = 9.3;
        exp = 1.2907615018426493;

        assertEquals(exp, MatrixNorms.norm(a, p, q));

        // ----------------------- sub-case 5 -----------------------
        aShape = new Shape(32, 32);
        aData = new Complex128[]{new Complex128(0.86738, 0.01318), new Complex128(0.83588, 0.48586), new Complex128(0.02535, 0.51349), new Complex128(0.00979, 0.97632), new Complex128(0.02841, 0.76271), new Complex128(0.02338, 0.49899), new Complex128(0.84757, 0.03883), new Complex128(0.7871, 0.17426), new Complex128(0.24826, 0.88086), new Complex128(0.34889, 0.24449)};
        aRowPointers = new int[]{0, 0, 0, 1, 3, 3, 3, 3, 5, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
        aColIndices = new int[]{0, 15, 31, 1, 25, 20, 10, 4, 4, 31};
        a = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        p = 0;
        q = 0;

        CsrCMatrix finalA = a;
        double finalP = p;
        double finalQ = q;
        assertThrows(IllegalArgumentException.class, () -> MatrixNorms.norm(finalA, finalP, finalQ));
    }
}
