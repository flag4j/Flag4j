package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCMatrixNormTests {


    @Test
    void cooCMatrixNormTests() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aData;
        CooCMatrix a;
        double exp;

        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(12, 12);
        aData = new Complex128[]{new Complex128(0.808, 0.929), new Complex128(0.231, 0.157), new Complex128(0.509, 0.895), new Complex128(0.25, 0.055), new Complex128(0.602, 0.428), new Complex128(0.39, 0.012), new Complex128(0.609, 0.324), new Complex128(0.248, 0.281), new Complex128(0.25, 0.694), new Complex128(0.176, 0.411), new Complex128(0.938, 0.241), new Complex128(0.831, 0.321), new Complex128(0.541, 0.092), new Complex128(0.406, 0.908)};
        aRowIndices = new int[]{1, 1, 2, 2, 4, 5, 5, 6, 6, 7, 7, 7, 9, 10};
        aColIndices = new int[]{1, 5, 1, 9, 0, 6, 7, 1, 2, 2, 9, 11, 7, 8};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        exp = 2.7927951947824603;

        assertEquals(exp, a.norm(), 1e-12);

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(300, 300);
        aData = new Complex128[]{new Complex128(0.754, 0.419), new Complex128(0.936, 0.196), new Complex128(0.661, 0.887), new Complex128(0.476, 0.081), new Complex128(0.129, 0.368), new Complex128(0.029, 0.25), new Complex128(0.482, 0.922), new Complex128(0.581, 0.33), new Complex128(0.627, 0.832)};
        aRowIndices = new int[]{5, 6, 46, 81, 98, 111, 186, 271, 294};
        aColIndices = new int[]{69, 212, 265, 42, 246, 186, 95, 39, 109};
        a = new CooCMatrix(aShape, aData, aRowIndices, aColIndices);

        exp = 2.438246090943242;

        assertEquals(exp, a.norm(), 1e-12);
    }
}
