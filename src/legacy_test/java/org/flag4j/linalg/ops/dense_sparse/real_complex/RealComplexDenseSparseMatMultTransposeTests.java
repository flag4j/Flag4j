package org.flag4j.linalg.ops.dense_sparse.real_complex;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatMultTranspose;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealComplexDenseSparseMatMultTransposeTests {
    static double[][] rdEntries = {};
    static Complex128[][] cdEntries = {};
    static Complex128[] actData;

    static double[] rspEntries = {};
    static Complex128[] cspEntries ={};

    static int[][] rindices;
    static int[][] cindices;

    static Shape rspShape;
    static Shape cspShape;

    static Matrix realDense;
    static CMatrix complexDense;
    static CooMatrix realSp;
    static CooCMatrix complexSp;

    static CMatrix exp;

    @BeforeAll
    static void setup() {
        rdEntries = new double[][]{
                {1.234, -00.024, 0, 1},
                {100.4, 5.14, -1.444, 0.041},
                {1.45, 985.1, -75.1, 4}
        };
        cdEntries = new Complex128[][]{
                {new Complex128(-0.24, 14.5), new Complex128(0.425)},
                {new Complex128(8.33, -84.4), Complex128.ZERO},
                {new Complex128(4.5, -9.24), new Complex128(0, -85.2)},
                {new Complex128("1.345"), new Complex128("-85.445+15.5i")}
        };

        rspShape = new Shape(2, 3).swapAxes(0, 1);
        cspShape = new Shape(4, 2).swapAxes(0, 1);

        rindices = new int[][]{{0, 2, 1}, {0, 0, 1}};
        cindices = new int[][]{{0, 1, 1}, {0, 2, 3}};

        rspEntries = new double[]{1.445, -9.25, 1.5};
        cspEntries = new Complex128[]{new Complex128(51.5, -0.42),
                new Complex128(-5.25, 15), new Complex128(3.45)};

        realDense = new Matrix(rdEntries);
        complexDense = new CMatrix(cdEntries);
        realSp = new CooMatrix(rspShape, rspEntries, rindices[0], rindices[1]);
        complexSp = new CooCMatrix(cspShape, cspEntries, cindices[0], cindices[1]);
    }


    @Test
    void realDenseComplexSpTestCase() {
        // ---------------------- sub-case 1 ----------------------
        exp = realDense.mult(new CooCMatrix(complexSp.shape.swapAxes(0, 1),
                complexSp.data, complexSp.colIndices, complexSp.rowIndices));

        actData = new Complex128[realDense.numRows*complexSp.numRows];
        RealFieldDenseCooMatMultTranspose.multTranspose(
                realDense.data, realDense.shape, complexSp.data,
                complexSp.rowIndices, complexSp.colIndices, complexSp.shape, actData);
        CMatrix act = new CMatrix(realDense.numRows, complexSp.numRows, actData);
        assertEquals(exp, act);
    }


    @Test
    void complexDenseRealSpTestCase() {
        // ---------------------- sub-case 1 ----------------------
        exp = complexDense.mult(new CooMatrix(realSp.shape.swapAxes(0, 1),
                realSp.data, realSp.colIndices, realSp.rowIndices));

        actData = new Complex128[realSp.numRows*complexDense.numRows];
        RealFieldDenseCooMatMultTranspose.multTranspose(
                complexDense.data, complexDense.shape, realSp.data,
                realSp.rowIndices, realSp.colIndices, realSp.shape, actData);
        CMatrix act = new CMatrix(complexDense.numRows, realSp.numRows, actData);
        assertEquals(exp, act);
    }
}
