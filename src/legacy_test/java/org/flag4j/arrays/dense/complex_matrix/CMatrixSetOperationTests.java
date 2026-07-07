package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixSetOperationTests {

    Complex128[][] entriesA, entriesExp;
    CMatrix A, exp;
    int sparseSize;
    int[] sparseIndices;
    int[] rowIndices, colIndices;
    Shape sparseShape;

    @Test
    void setValuesComplex128TestCase() {
        Complex128[][] values;

        // -------------- sub-case 1 --------------
        values = new Complex128[][]{{new Complex128(23.4, -9.433), new Complex128(-9431, 0.23)},
                {new Complex128(9.23, 55.6), new Complex128(0, -78)},
                {new Complex128(5.1114, -5821.23), new Complex128(754.1, -823.1)}};
        exp = new CMatrix(values);
        entriesA = new Complex128[][]{{new Complex128(23.4, -9.433), new Complex128(-9431, 0.23)},
                {new Complex128(9.23, 55.6), new Complex128(0, -78)},
                {new Complex128(5.1114, -5821.23), new Complex128(754.1, -823.1)}};
        A = new CMatrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- sub-case 2 --------------
        values = new Complex128[][]{{new Complex128(23.4, -9.433), new Complex128(-9431, 0.23)},
                {new Complex128(9.23, 55.6), new Complex128(0, -78)},
                {new Complex128(5.1114, -5821.23), new Complex128(754.1, -823.1)}};
        entriesA = new Complex128[][]{{new Complex128(23.4, -9.433), new Complex128(-9431, 0.23), new Complex128(9.23, 55.6)},
                {new Complex128(0, -78), new Complex128(5.1114, -5821.23), new Complex128(754.1, -823.1)}};
        A = new CMatrix(entriesA);

        Complex128[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setColumnComplex128TestCase() {
        Complex128[] values;
        int col;

        // -------------- sub-case 1 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32), new Complex128(94.1)};
        col = 0;
        entriesExp = new Complex128[][]
                {{new Complex128(2.345, 5.15), new Complex128(0)},
                {new Complex128(-445, 0.32), new Complex128(4)},
                {new Complex128(94.1), new Complex128(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}, {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- sub-case 2 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32), new Complex128(94.1)};
        col = 0;
        entriesA = new Complex128[][]{{new Complex128(2.345, 5.15), new Complex128(0), new Complex128(1)},
                {new Complex128(-445, 0.32), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- sub-case 3 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32), new Complex128(94.1)};
        col = 1;
        entriesExp = new Complex128[][]{{new Complex128(0), new Complex128(2.345, 5.15)},
                {new Complex128(1), new Complex128(-445, 0.32)},
                {new Complex128(1331.14), new Complex128(94.1)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}, {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32)};
        col = -1;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32)};
        col = 3;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }


    @Test
    void setColumnCVectorTestCase() {
        Complex128[] values;
        CVector valuesVec;
        int col;

        // -------------- sub-case 1 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32), new Complex128(94.1)};
        valuesVec = new CVector(values);
        col = 0;
        entriesExp = new Complex128[][]
                {{new Complex128(2.345, 5.15), new Complex128(0)},
                        {new Complex128(-445, 0.32), new Complex128(4)},
                        {new Complex128(94.1), new Complex128(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{
            new Complex128(0), new Complex128(0)},
                {new Complex128(1), new Complex128(4)},
                {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(valuesVec, col);

        assertEquals(exp, A);

        // -------------- sub-case 2 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32), new Complex128(94.1)};
        valuesVec = new CVector(values);
        col = 0;
        entriesA = new Complex128[][]{{new Complex128(2.345, 5.15), new Complex128(0), new Complex128(1)},
                {new Complex128(-445, 0.32), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        CVector finalValuesVec = valuesVec;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValuesVec, finalCol));

        // -------------- sub-case 3 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32), new Complex128(94.1)};
        valuesVec = new CVector(values);
        col = 1;
        entriesExp = new Complex128[][]{{new Complex128(0), new Complex128(2.345, 5.15)},
                {new Complex128(1), new Complex128(-445, 0.32)},
                {new Complex128(1331.14), new Complex128(94.1)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}, {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(valuesVec, col);

        assertEquals(exp, A);

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32)};
        valuesVec = new CVector(values);
        col = -1;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        CVector finalValuesVec1 = valuesVec;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValuesVec1, finalCol1));

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(2.345, 5.15), new Complex128(-445, 0.32)};
        valuesVec = new CVector(values);
        col = 3;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        CVector finalValuesVec2 = valuesVec;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValuesVec2, finalCol2));
    }


    @Test
    void setRowComplex128TestCase() {
        Complex128[] values;
        int row;

        // -------------- sub-case 1 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        row = 0;
        entriesExp = new Complex128[][]{
                {new Complex128(34, -55.6), new Complex128(0.44, -0.23)},
                {new Complex128(1), new Complex128(4)},
                {new Complex128(1331.14), new Complex128(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}, {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- sub-case 2 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        row = 0;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- sub-case 3 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        row = 1;
        entriesExp = new Complex128[][]{
                {new Complex128(0), new Complex128(0)},
                {new Complex128(34, -55.6), new Complex128(0.44, -0.23)},
                {new Complex128(1331.14), new Complex128(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}, {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        row = -1;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(IndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23), new Complex128(9.234, -0.2334)};
        row = 3;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(IndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setRowCVectorTestCase() {
        Complex128[] values;
        CVector valuesVec;
        int row;

        // -------------- sub-case 1 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        valuesVec = new CVector(values);
        row = 0;
        entriesExp = new Complex128[][]{
                {new Complex128(34, -55.6), new Complex128(0.44, -0.23)},
                {new Complex128(1), new Complex128(4)},
                {new Complex128(1331.14), new Complex128(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}, {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(valuesVec, row);

        assertEquals(exp, A);

        // -------------- sub-case 2 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        valuesVec = new CVector(values);
        row = 0;
        entriesA = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(1)},
                {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        Complex128[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- sub-case 3 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        valuesVec = new CVector(values);
        row = 1;
        entriesExp = new Complex128[][]{
                {new Complex128(0), new Complex128(0)},
                {new Complex128(34, -55.6), new Complex128(0.44, -0.23)},
                {new Complex128(1331.14), new Complex128(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new Complex128[][]{
                {new Complex128(0), new Complex128(0)},
                {new Complex128(1), new Complex128(4)},
                {new Complex128(1331.14), new Complex128(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23)};
        valuesVec = new CVector(values);
        row = -1;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0)}, {new Complex128(1), new Complex128(4)}};
        A = new CMatrix(entriesA);

        CVector finalValues1 = valuesVec;
        int finalRow1 = row;
        assertThrows(IndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- sub-case 4 --------------
        values = new Complex128[]{new Complex128(34, -55.6), new Complex128(0.44, -0.23), new Complex128(9.234, -0.2334)};
        valuesVec = new CVector(values);
        row = 3;
        entriesA = new Complex128[][]{{new Complex128(0), new Complex128(0), new Complex128(1)}, {new Complex128(1), new Complex128(4), new Complex128(2)}};
        A = new CMatrix(entriesA);

        CVector finalValues2 = valuesVec;
        int finalRow2 = row;
        assertThrows(IndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setSliceCopyCMatrixTestCase() {
        Complex128[][] valueEntries;
        CMatrix values;
        int row, col;

        // -------------- sub-case 1 --------------
        valueEntries = new Complex128[][]{{new Complex128(-71.33), new Complex128(34.61)}, {new Complex128(-99.24), new Complex128(-13.4)}};
        values = new CMatrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(2.2), new Complex128(83.1)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-7.13), new Complex128(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new Complex128[][]{{new Complex128(-71.33), new Complex128(34.61), new Complex128(2.2), new Complex128(83.1)}, {new Complex128(-99.24), new Complex128(-13.4), new Complex128(-7.13), new Complex128(0.00013)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- sub-case 2 --------------
        valueEntries = new Complex128[][]{{new Complex128(-71.33), new Complex128(34.61)}, {new Complex128(-99.24), new Complex128(-13.4)}};
        values = new CMatrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(2.2), new Complex128(83.1)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-7.13), new Complex128(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(-71.33), new Complex128(34.61)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-99.24), new Complex128(-13.4)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- sub-case 3 --------------
        valueEntries = new Complex128[][]{{new Complex128(-71.33), new Complex128(34.61)}, {new Complex128(-99.24), new Complex128(-13.4)}};
        values = new CMatrix(valueEntries);
        entriesA = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(2.2), new Complex128(83.1)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-7.13), new Complex128(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(-71.33), new Complex128(34.61)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-99.24), new Complex128(-13.4)}};
        exp = new CMatrix(entriesExp);

        CMatrix finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceCMatrixTestCase() {
        Complex128[][] values;
        CMatrix mat;
        int row, col;

        // -------------- sub-case 1 --------------
        values = new Complex128[][]{
                {new Complex128(0.234, -84.12), new Complex128(33, 441.435)},
                {new Complex128(0, 442.4), new Complex128(24.88)}};
        mat = new CMatrix(values);
        row = 0;
        col = 0;
        entriesA = new Complex128[][]{
                {new Complex128(-99.234), new Complex128(132), new Complex128(2.2), new Complex128(83.1)},
                {new Complex128(11.346), new Complex128(124.6), new Complex128(-7.13), new Complex128(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new Complex128[][]{
                {new Complex128(0.234, -84.12), new Complex128(33, 441.435), new Complex128(2.2), new Complex128(83.1)},
                {new Complex128(0, 442.4), new Complex128(24.88), new Complex128(-7.13), new Complex128(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(mat, row, col);
        assertEquals(exp, A);

        // -------------- sub-case 2 --------------
        values = new Complex128[][]{
                {new Complex128(0.234, -84.12), new Complex128(33, 441.435)},
                {new Complex128(0, 442.4), new Complex128(24.88)}};
        mat = new CMatrix(values);
        row = 0;
        col = 2;
        entriesA = new Complex128[][]{
                {new Complex128(-99.234), new Complex128(132), new Complex128(2.2), new Complex128(83.1)},
                {new Complex128(11.346), new Complex128(124.6), new Complex128(-7.13), new Complex128(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new Complex128[][]{
                {new Complex128(-99.234), new Complex128(132), new Complex128(0.234, -84.12), new Complex128(33, 441.435)},
                {new Complex128(11.346), new Complex128(124.6), new Complex128(0, 442.4), new Complex128(24.88)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(mat, row, col);

        assertEquals(exp, A);

        // -------------- sub-case 3 --------------
        values = new Complex128[][]{
                {new Complex128(0.234, -84.12), new Complex128(33, 441.435)},
                {new Complex128(0, 442.4), new Complex128(24.88)}};
        mat = new CMatrix(values);
        entriesA = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(2.2), new Complex128(83.1)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-7.13), new Complex128(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new Complex128[][]{{new Complex128(-99.234), new Complex128(132), new Complex128(-71), new Complex128(34)}, {new Complex128(11.346), new Complex128(124.6), new Complex128(-99), new Complex128(-13)}};
        exp = new CMatrix(entriesExp);

        CMatrix finalValues = mat;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }
}
