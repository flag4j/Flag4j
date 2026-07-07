package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixNormTests {

    @Test
    void testCMatrixNorm() {
        Shape aShape;
        Complex128[] aData;
        CMatrix a;
        double exp, p, q;

        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(5, 5);
        aData = new Complex128[]{new Complex128(0.967, 0.682), new Complex128(0.575, 0.711), new Complex128(0.969, 0.409), new Complex128(0.76, 0.337), new Complex128(0.29, 0.104), new Complex128(0.647, 0.085), new Complex128(0.008, 0.563), new Complex128(0.662, 0.6), new Complex128(0.337, 0.836), new Complex128(0.783, 0.89), new Complex128(0.9, 0.439), new Complex128(0.633, 0.53), new Complex128(0.002, 0.758), new Complex128(0.033, 0.509), new Complex128(0.438, 0.833), new Complex128(0.72, 0.67), new Complex128(0.043, 0.722), new Complex128(0.642, 0.79), new Complex128(0.602, 0.604), new Complex128(0.398, 0.693), new Complex128(0.599, 0.455), new Complex128(0.981, 0.875), new Complex128(0.15, 0.586), new Complex128(0.151, 0.755), new Complex128(0.663, 0.114)};
        a = new CMatrix(aShape, aData);

        p = 1;
        q = 1;
        exp = 21.011943301767335;

        assertEquals(exp, a.norm(p, q));
    }
}
