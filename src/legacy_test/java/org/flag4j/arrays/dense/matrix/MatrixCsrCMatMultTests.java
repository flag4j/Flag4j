package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixCsrCMatMultTests {

    Matrix A;
    double[][] aEntries;

    CMatrix exp;
    Complex128[][] expEntries;

    CsrCMatrix B;
    Shape bShape;
    Complex128[] bEntries;
    int[] bRowPointers;
    int[] bColIndices;


    @Test
    void standardTests() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new double[][]{
                {0.7604, 0.3083, 0.62095, 0.80536, 0.99013},
                {0.79418, 0.0794, 0.01267, 0.19388, 0.25461},
                {0.98692, 0.34513, 0.82222, 0.05736, 0.3754},
                {0.36058, 0.74475, 0.05811, 0.62647, 0.37478},
                {0.54116, 0.91009, 0.26198, 0.77406, 0.89967}};
        A = new Matrix(aEntries);

        bShape = new Shape(5, 4);
        bEntries = new Complex128[]{new Complex128(0.21723, 0.346), new Complex128(0.41679, 0.05828)};
        bRowPointers = new int[]{0, 0, 0, 1, 2, 2};
        bColIndices = new int[]{2, 2};
        B = new CsrCMatrix(bShape, bEntries, bRowPointers, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.4705549629+0.26178508079999996i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.08355954930000001+0.0156831464i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.202517925+0.2878310608i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.2737296666+0.0566167316i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.3795303828+0.13575729679999998i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 2 ------------------------
        bShape = new Shape(3, 10);
        bEntries = new Complex128[]{new Complex128(0.8369, 0.15486), new Complex128(0.62217, 0.32219), new Complex128(0.66049, 0.96046)};
        bRowPointers = new int[]{0, 1, 3, 3};
        bColIndices = new int[]{8, 2, 5};
        B = new CsrCMatrix(bShape, bEntries, bRowPointers, bColIndices);

        aEntries = new double[][]{
                {0.02829, 0.12932, 0.0628},
                {0.41774, 0.34779, 0.53389},
                {0.94484, 0.98662, 0.61331},
                {0.99492, 0.90161, 0.53586},
                {0.09276, 0.7781, 0.38807},
                {0.60931, 0.48504, 0.75447},
                {0.83833, 0.11039, 0.64302},
                {0.98093, 0.8886, 0.95656},
                {0.29491, 0.50597, 0.51248},
                {0.59388, 0.09991, 0.27436},
                {0.93154, 0.0315, 0.94464}};
        A = new Matrix(aEntries);

        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0804590244+0.041665610799999996i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0854145668+0.12420668719999998i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.023675901+0.0043809894i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.2163845043+0.11205446009999999i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.2297118171+0.33403838339999997i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.349606606+0.0646912164i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.6138453654+0.3178790978i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.6516526438000001+0.9476090452i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.790736596+0.1463179224i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.5609546937000001+0.2904897259i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.5955043889+0.8659603406i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.832648548+0.1540733112i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.484110477+0.250696039i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.5139272690000001+0.747333926i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.07763084399999999+0.014364813599999998i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.3017773368+0.1562750376i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.3203640696+0.4658615184i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.509931539+0.0943577466i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0686813463+0.035566554099999995i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0729114911+0.1060251794i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.701598377+0.1298237838i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.5528602619999999+0.286298034i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.586911414+0.8534647559999999i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.8209403169999999+0.15190681979999998i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.3147993549+0.1630184743i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.3341881253+0.4859639462i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.246810179+0.0456697626i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0621610047+0.0321900029i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0659895559+0.0959595586i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.497018172+0.0919682568i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.019598355+0.010148985i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.020805435+0.03025449i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.7796058260000001+0.1442582844i"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 3 ------------------------
        A = new Matrix(24, 516);
        B = new CsrCMatrix(15, 12);
        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
