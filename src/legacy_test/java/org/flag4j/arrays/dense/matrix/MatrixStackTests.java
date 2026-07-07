package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixStackTests {

    double[][] expEntries;
    Complex128[][] expCEntries;

    double[][] entries;
    double[][] entries2;
    Complex128[][] CEntries;

    double[] sparseEntries;
    Complex128[] sparseCEntries;

    int[] indices;
    int[] rowIndices;
    int[] colIndices;

    Shape sparseShape;

    Matrix realDense;
    Matrix realDense2;
    CooMatrix realSparse;
    CMatrix complexDense;
    CooCMatrix complexSparse;
    Vector realDenseVector;
    CooVector realCooVector;
    CVector complexDenseVector;
    CooCVector complexSparseVector;

    Matrix exp;
    CMatrix expC;

    @Test
    void realDenseTestCase() {
        // -----------------------  sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}, {99.2134, -0.23}};

        expEntries = new double[][]{{1, 2, 3, 4, 5}, {4, 14, -5.12, 5, 4}, {0, 9.34, -0.13, 99.2134, -0.23}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertEquals(exp, realDense.stack(realDense2, 0));

        // -----------------------  sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 0));


        // -----------------------  sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5, -0.53}, {5, 4, 1.3}};

        expEntries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}, {4, 5, -0.53}, {5, 4, 1.3}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertEquals(exp, realDense.stack(realDense2, 1));

        // -----------------------  sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 1));

        // -----------------------  sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 7));
    }
}
