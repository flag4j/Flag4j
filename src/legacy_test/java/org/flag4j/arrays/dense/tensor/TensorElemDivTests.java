package org.flag4j.arrays.dense.tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorElemDivTests {
    static double[] aEntries;
    static Tensor A;
    static Shape aShape, bShape, expShape;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries, expEntries;
        Tensor B, exp;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new double[]{
                aEntries[0]/bEntries[0], aEntries[1]/bEntries[1], aEntries[2]/bEntries[2],
                aEntries[3]/bEntries[3], aEntries[4]/bEntries[4], aEntries[5]/bEntries[5],
                aEntries[6]/bEntries[6], aEntries[7]/bEntries[7], aEntries[8]/bEntries[8],
                aEntries[9]/bEntries[9], aEntries[10]/bEntries[10], aEntries[11]/bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.div(B));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.div(finalB));

        // ----------------------- sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.div(finalB1));
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries, expEntries;
        CTensor B, exp;

        // ----------------------- sub-case 1 -----------------------
        bEntries = new Complex128[]{
                new Complex128(-0.00234, 2.452), new Complex128(15.6), new Complex128(99.2442, 9.1),
                new Complex128(100.252, 1235), new Complex128(-78.2556, -99.1441), new Complex128(0.111134, -772.4),
                new Complex128(671.455, 15.56), new Complex128(-0.00024), new Complex128(515.667, 895.52),
                new Complex128(14.515), new Complex128(100.135), new Complex128(0, 1)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);
        expEntries = new Complex128[]{
                new Complex128(-4.7871787233852756E-4, -0.5016308645188331), new Complex128(0.16384615384615386, 0.0),
                new Complex128(-1.21404568186213, 0.1113195099053182),
                new Complex128(0.0010193178730469059, -0.012556932262826965),
                new Complex128(-0.06940905320103984, 0.087936174682313),
                new Complex128(-1.8485193391306292E-5, -0.12847520448688052),
                new Complex128(2.0020375218720557E-6, -4.639432849607075E-8),
                new Complex128(-11154.166666666668, -0.0), new Complex128(0.003930754845372011, -0.006826245579274111),
                new Complex128(-1.3365483982087493E-5, -0.0), new Complex128(0.00998651820042942, 0.0),
                new Complex128(0.0, -234.0)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);
        assertEquals(exp, A.div(B));

        // ----------------------- sub-case 2 -----------------------
        bEntries = new Complex128[]{
                new Complex128(-0.00234, 2.452), new Complex128(15.6), new Complex128(99.2442, 9.1),
                new Complex128(100.252, 1235), new Complex128(-78.2556, -99.1441), new Complex128(0.111134, -772.4),
                new Complex128(671.455, 15.56), new Complex128(-0.00024), new Complex128(515.667, 895.52),
                new Complex128(14.515), new Complex128(100.135), new Complex128(0, 1)
        };
        bShape = new Shape(12);
        B = new CTensor(bShape, bEntries);

        CTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.div(finalB));
    }
}
