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

class MatrixAddTests {

    double[][] aEntries, bEntries;
    Complex128[][] bCEntries;

    Matrix A, B;
    CMatrix BC, expC;
    double b;
    Complex128 bC;
    Matrix sum, exp;
    CMatrix sumC;
    Shape expShape;
    double[] expEntries;
    Complex128[] expEntriesC;

    @Test
    void matrixMatrixTestCase() {
        // --------------- sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}, {1.343, 6.7, -88.4}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        expShape = A.shape;
        expEntries = new double[]{1+0.333, 2+56.4, 3+13.4, 4-1.44, 5+5, 6+85.1, 7+1.343, 8+6.7, 9-88.4};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(B);

        assertEquals(exp, sum);

        // --------------- sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        expShape = A.shape;
        expEntries = new double[]{1+0.333, 2+56.4, 3+13.4, 4 - 1.44, 5+5, 6+85.1};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(B);

        assertEquals(exp, sum);

        // --------------- sub-case 3 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}, {1, 2, 3}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));

        // --------------- sub-case 4 ---------------
        aEntries = new double[][]{{1, 2}, {4, 5}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));
    }


    @Test
    void matrixDoubleTestCase() {
        // --------------- sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        b = 2.133;
        A = new Matrix(aEntries);
        expShape = A.shape;
        expEntries = new double[]{1+2.133, 2+2.133, 3+2.133, 4+2.133, 5+2.133, 6+2.133, 7+2.133, 8+2.133, 9+2.133};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(b);

        assertEquals(exp, sum);

        // --------------- sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        b = 2.133;
        A = new Matrix(aEntries);
        expShape = A.shape;
        expEntries = new double[]{1+2.133, 2+2.133, 3+2.133, 4+2.133, 5+2.133, 6+2.133};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(b);

        assertEquals(exp, sum);
    }


    @Test
    void matrixComplex128TestCase() {
        // --------------- sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bC = new Complex128(33.444, -9.3545);
        A = new Matrix(aEntries);
        expShape = A.shape;
        expEntriesC = new Complex128[]{
                new Complex128(1).add(bC), new Complex128(2).add(bC), new Complex128(3).add(bC),
                new Complex128(4).add(bC), new Complex128(5).add(bC), new Complex128(6).add(bC),
                new Complex128(7).add(bC), new Complex128(8).add(bC), new Complex128(9).add(bC)
        };

        sumC = A.add(bC);

        assertArrayEquals(expEntriesC, sumC.data);
        assertEquals(expShape, sumC.shape);

        // --------------- sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bC = new Complex128(33.444, -9.3545);
        A = new Matrix(aEntries);
        expShape = A.shape;
        expEntriesC = new Complex128[]{
                new Complex128(1).add(bC), new Complex128(2).add(bC), new Complex128(3).add(bC),
                new Complex128(4).add(bC), new Complex128(5).add(bC), new Complex128(6).add(bC)
        };

        sumC = A.add(bC);

        assertArrayEquals(expEntriesC, sumC.data);
        assertEquals(expShape, sumC.shape);
    }


    @Test
    void matrixCMatrixTestCase() {
        // --------------- sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bCEntries = new Complex128[][]{
                {new Complex128(1.23, -344.5), new Complex128(2.33, 5.6), new Complex128(3.13, -34)},
                {new Complex128(0, 66.45), new Complex128(33.1334, 5513.5), new Complex128(99.3)},
                {new Complex128(1.23), new Complex128(8, 3), new Complex128(9, -0.000000001)}
        };
        A = new Matrix(aEntries);
        BC = new CMatrix(bCEntries);
        expShape = A.shape;
        expEntriesC = new Complex128[]{
                new Complex128(1.23+1, -344.5), new Complex128(2.33+2, 5.6), new Complex128(3.13+3, -34),
                new Complex128(4, 66.45), new Complex128(33.1334+5, 5513.5), new Complex128(99.3+6),
                new Complex128(1.23+7), new Complex128(8+8, 3), new Complex128(9+9, -0.000000001)
        };

        sumC = A.add(BC);

        assertArrayEquals(expEntriesC, sumC.data);
        assertEquals(expShape, sumC.shape);


        // --------------- sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bCEntries = new Complex128[][]{
                {new Complex128(1.23, -344.5), new Complex128(2.33, 5.6), new Complex128(3.13, -34)},
                {new Complex128(0, 66.45), new Complex128(33.1334, 5513.5), new Complex128(99.3)}
        };
        A = new Matrix(aEntries);
        BC = new CMatrix(bCEntries);
        expShape = A.shape;
        expEntriesC = new Complex128[]{
                new Complex128(1.23+1, -344.5), new Complex128(2.33+2, 5.6), new Complex128(3.13+3, -34),
                new Complex128(4, 66.45), new Complex128(33.1334+5, 5513.5), new Complex128(99.3+6)
        };

        sumC = A.add(BC);

        assertArrayEquals(expEntriesC, sumC.data);
        assertEquals(expShape, sumC.shape);

        // --------------- sub-case 3 ---------------
        aEntries = new double[][]{{1, 2}, {4, 5}};
        bCEntries = new Complex128[][]{
                {new Complex128(1.23, -344.5), new Complex128(2.33, 5.6), new Complex128(3.13, -34)},
                {new Complex128(0, 66.45), new Complex128(33.1334, 5513.5), new Complex128(99.3)}
        };
        A = new Matrix(aEntries);
        BC = new CMatrix(bCEntries);

        assertThrows(LinearAlgebraException.class, ()->A.add(BC));
    }

    @Test
    void realSparseAddTestCase() {
        double[] bEntries;
        int[] bRowIndices;
        int[] bColIndices;
        Shape bShape;
        CooMatrix B;

        // --------------- sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);

        bEntries = new double[]{-0.99, 1, 14.2, 8.3};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = A.shape;
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[]{1, 2-0.99, 3, 4+1, 5, 6+14.2, 7, 8, 9, 10+8.3, 11, 12};
        expShape = A.shape;
        exp = new Matrix(expShape, expEntries);

        assertEquals(exp, A.add(B));

        // --------------- sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);

        bEntries = new double[]{-0.99, 1, 14.2, 8.3};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = new Shape(4, 3);
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }


    @Test
    void complexSparseAddTestCase() {
        Complex128[] bEntries;
        int[] bRowIndices;
        int[] bColIndices;
        Shape bShape;
        CooCMatrix B;

        // --------------- sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new Matrix(aEntries);

        bEntries = new Complex128[]{new Complex128(-0.123, 1), new Complex128(0, 1),
            new Complex128(8, 3.3), new Complex128(100.23, -1000.2)};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = A.shape;
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntriesC = new Complex128[]{
                new Complex128(1), new Complex128(2-0.123, 1), new Complex128(3),
                new Complex128(4, 1), new Complex128(5), new Complex128(6+8, 3.3),
                new Complex128(7), new Complex128(8), new Complex128(9),
                new Complex128(10+100.23, -1000.2), new Complex128(11), new Complex128(12)};
        expShape = A.shape;
        expC = new CMatrix(expShape, expEntriesC);

        assertEquals(expC, A.add(B));

        // --------------- sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);

        bEntries = new Complex128[]{new Complex128(-0.123, 1), new Complex128(0, 1),
                new Complex128(8, 3.3), new Complex128(100.23, -1000.2)};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = new Shape(4, 3);
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }
}
