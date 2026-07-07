package org.flag4j.arrays.dense.matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixElemMultTests {

    Matrix A, B, result, expResult;
    CMatrix BC, resultC, expResultC;
    double[][] entriesA, entriesB, expEntries;
    Complex128[][] entriesBC;

    CooMatrix BSparse, expSparse, sparseResult;
    CooCMatrix BSparseComplex, expSparseComplex, sparseComplexResult;
    Complex128[] bEntriesSparseComplex, expEntriesSparseComplex;
    double[] bEntriesSparse, expEntriesSparse;
    Shape shape;

    int[] rowIndices;
    int[] colIndices;

    private double[] getExp(double[] src1, double[] src2) {
        double[] result = new double[src1.length];

        for(int i=0; i<result.length; i++) {
            result[i] = src1[i]*src2[i];
        }

        return result;
    }

    private Complex128[] getExp(double[] src1, Complex128[] src2) {
        Complex128[] result = new Complex128[src1.length];

        for(int i=0; i<result.length; i++) {
            result[i] = src2[i].mult(src1[i]);
        }

        return result;
    }


    @Test
    void elemMultTestCase() {
        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94, -0.4442}, {0.0000234, 1333.4, 44.5, 134.3}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);
        expResult = new Matrix(A.shape, getExp(A.data, B.data));

        result = A.elemMult(B);

        assertArrayEquals(expResult.data, result.data);
        assertEquals(expResult.shape, result.shape);

        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94}, {0.0000234, 1333.4, 44.5}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));
    }


    @Test
    void elemMultComplexTestCase() {
        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new Complex128[][]{{new Complex128(1.4, 5), new Complex128(0, -1), new Complex128(1.3), Complex128.ZERO},
                {new Complex128(4.55, -93.2), new Complex128(-2, -13), new Complex128(8.9), new Complex128(0, 13)}};
        A = new Matrix(entriesA);
        BC = new CMatrix(entriesBC);
        expResultC = new CMatrix(A.shape, getExp(A.data, BC.data));

        resultC = A.elemMult(BC);

        assertArrayEquals(expResultC.data, resultC.data);
        assertEquals(expResultC.shape, resultC.shape);

        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new Complex128[][]{{new Complex128(1.4, 5), new Complex128(0, -1), new Complex128(1.3)},
                {new Complex128(4.55, -93.2), new Complex128(-2, -13), new Complex128(8.9)}};
        A = new Matrix(entriesA);
        BC = new CMatrix(entriesBC);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(BC));
    }


    @Test
    void elemMultSparseTestCase() {
        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new Matrix(entriesA);
        bEntriesSparse = new double[]{1.45, 31.13, 7.1};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(entriesA.length, entriesA[0].length);
        BSparse = new CooMatrix(shape, bEntriesSparse, rowIndices, colIndices);

        expEntriesSparse = new double[]{2*1.45, -6*31.13, 0};
        expSparse = new CooMatrix(shape, expEntriesSparse, rowIndices.clone(), colIndices.clone());

        sparseResult = A.elemMult(BSparse);

        assertArrayEquals(expSparse.data, sparseResult.data);
        assertEquals(expSparse.shape, sparseResult.shape);

        // ----------------- sub-case 2 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new Matrix(entriesA);
        bEntriesSparse = new double[]{1.45, 31.13, 7.1};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(3, 4);
        BSparse = new CooMatrix(shape, bEntriesSparse, rowIndices, colIndices);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(BSparse));
    }


    @Test
    void elemMultSparseComplexTestCase() {
        // ----------------- sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new Matrix(entriesA);
        bEntriesSparseComplex = new Complex128[]{new Complex128(1000234, -8.312), new Complex128(19.334, -96.23), new Complex128(0, 1.56)};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(entriesA.length, entriesA[0].length);
        BSparseComplex = new CooCMatrix(shape, bEntriesSparseComplex, rowIndices, colIndices);

        expEntriesSparseComplex = new Complex128[]{bEntriesSparseComplex[0].mult(2), bEntriesSparseComplex[1].mult(-6), Complex128.ZERO};
        expSparseComplex = new CooCMatrix(shape, expEntriesSparseComplex, rowIndices.clone(), colIndices.clone());

        sparseComplexResult = A.elemMult(BSparseComplex);

        assertArrayEquals(expSparseComplex.data, sparseComplexResult.data);
        assertEquals(expSparseComplex.shape, sparseComplexResult.shape);

        // ----------------- sub-case 2 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new Matrix(entriesA);
        bEntriesSparseComplex = new Complex128[]{new Complex128(1000234, -8.312), new Complex128(19.334, -96.23), new Complex128(0, 1.56)};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(15, 500);
        BSparseComplex = new CooCMatrix(shape, bEntriesSparseComplex, rowIndices, colIndices);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(BSparseComplex));
    }
}
