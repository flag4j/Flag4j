package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixAugmentTests {

    @Test
    void realSparseAugmentTest() {
        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        Complex128[] bEntries;
        CooCMatrix b;

        Shape cShape;
        int[] cRowIndices;
        int[] cColIndices;
        double[] cEntries;
        CooMatrix c;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        bShape = new Shape(3, 2);
        bEntries = new Complex128[]{new Complex128("0.84785+0.65961i")};
        bRowIndices = new int[]{2};
        bColIndices = new int[]{0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cShape = new Shape(3, 4);
        cEntries = new double[]{0.54711, 0.0956};
        cRowIndices = new int[]{1, 2};
        cColIndices = new int[]{1, 2};
        c = new CooMatrix(cShape, cEntries, cRowIndices, cColIndices);

        expShape = new Shape(3, 6);
        expEntries = new Complex128[]{new Complex128("0.54711"), new Complex128("0.84785+0.65961i"), new Complex128("0.0956")};
        expRowIndices = new int[]{1, 2, 2};
        expColIndices = new int[]{3, 0, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(c));

        // ---------------------  sub-case 2 ---------------------
        bShape = new Shape(2, 1);
        bEntries = new Complex128[]{};
        bRowIndices = new int[]{};
        bColIndices = new int[]{};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cShape = new Shape(2, 5);
        cEntries = new double[]{0.11012, 0.05635};
        cRowIndices = new int[]{0, 1};
        cColIndices = new int[]{1, 4};
        c = new CooMatrix(cShape, cEntries, cRowIndices, cColIndices);

        expShape = new Shape(2, 6);
        expEntries = new Complex128[]{new Complex128("0.11012"), new Complex128("0.05635")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{2, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(c));

        // ---------------------  sub-case 3 ---------------------
        bShape = new Shape(5, 14);
        bEntries = new Complex128[]{new Complex128("0.20701+0.87362i"), new Complex128("0.99981+0.30156i"), new Complex128("0.59447+0.9851i"), new Complex128("0.59934+0.03263i"), new Complex128("0.8151+0.29466i"), new Complex128("0.99307+0.28989i"), new Complex128("0.53224+0.11927i"), new Complex128("0.94235+0.30785i"), new Complex128("0.40525+0.58865i"), new Complex128("0.97904+0.92044i"), new Complex128("0.21766+0.52526i"), new Complex128("0.94594+0.3945i"), new Complex128("0.64977+0.97097i"), new Complex128("0.70112+0.12694i")};
        bRowIndices = new int[]{0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4};
        bColIndices = new int[]{0, 1, 1, 4, 13, 1, 3, 9, 12, 6, 7, 4, 5, 6};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cShape = new Shape(6, 14);
        cEntries = new double[]{0.2779, 0.48408, 0.54389, 0.09072, 0.90854, 0.18127, 0.57206, 0.80702, 0.42993, 0.84323, 0.96082, 0.38449, 0.42693, 0.55268, 0.03662, 0.21952, 0.36737};
        cRowIndices = new int[]{0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5};
        cColIndices = new int[]{9, 12, 8, 9, 10, 11, 1, 5, 8, 10, 0, 6, 7, 13, 13, 0, 9};
        c = new CooMatrix(cShape, cEntries, cRowIndices, cColIndices);

        CooCMatrix finalb = b;
        CooMatrix finalc = c;
        assertThrows(Exception.class, ()->finalb.augment(finalc));
    }


    @Test
    void complexSparseAugmentTest() {
        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        Complex128[] bEntries;
        CooCMatrix b;

        Shape dShape;
        int[] dRowIndices;
        int[] dColIndices;
        Complex128[] dEntries;
        CooCMatrix d;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        bShape = new Shape(3, 2);
        bEntries = new Complex128[]{new Complex128("0.31432+0.12231i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        dShape = new Shape(3, 4);
        dEntries = new Complex128[]{new Complex128("0.7633+0.64041i"), new Complex128("0.31419+0.11711i")};
        dRowIndices = new int[]{0, 2};
        dColIndices = new int[]{2, 3};
        d = new CooCMatrix(dShape, dEntries, dRowIndices, dColIndices);

        expShape = new Shape(3, 6);
        expEntries = new Complex128[]{new Complex128("0.31432+0.12231i"), new Complex128("0.7633+0.64041i"), new Complex128("0.31419+0.11711i")};
        expRowIndices = new int[]{0, 0, 2};
        expColIndices = new int[]{1, 4, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(d));

        // ---------------------  sub-case 2 ---------------------
        bShape = new Shape(2, 1);
        bEntries = new Complex128[]{};
        bRowIndices = new int[]{};
        bColIndices = new int[]{};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        dShape = new Shape(2, 5);
        dEntries = new Complex128[]{new Complex128("0.64596+0.34447i"), new Complex128("0.19735+0.00648i")};
        dRowIndices = new int[]{0, 0};
        dColIndices = new int[]{1, 3};
        d = new CooCMatrix(dShape, dEntries, dRowIndices, dColIndices);

        expShape = new Shape(2, 6);
        expEntries = new Complex128[]{new Complex128("0.64596+0.34447i"), new Complex128("0.19735+0.00648i")};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{2, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(d));

        // ---------------------  sub-case 3 ---------------------
        bShape = new Shape(5, 14);
        bEntries = new Complex128[]{new Complex128("0.61976+0.88453i"), new Complex128("0.79514+0.80535i"), new Complex128("0.81806+0.71575i"), new Complex128("0.4667+0.1409i"), new Complex128("0.42131+0.5932i"), new Complex128("0.74726+0.137i"), new Complex128("0.33884+0.75794i"), new Complex128("0.8802+0.65175i"), new Complex128("0.81513+0.70436i"), new Complex128("0.86364+0.37206i"), new Complex128("0.54062+0.81757i"), new Complex128("0.66025+0.76792i"), new Complex128("0.88691+0.59128i"), new Complex128("0.82567+0.90133i")};
        bRowIndices = new int[]{0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4};
        bColIndices = new int[]{5, 8, 9, 5, 8, 2, 7, 9, 12, 13, 2, 5, 8, 11};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        dShape = new Shape(6, 14);
        dEntries = new Complex128[]{new Complex128("0.18121+0.09923i"), new Complex128("0.40046+0.45834i"), new Complex128("0.70669+0.49123i"), new Complex128("0.14628+0.52565i"), new Complex128("0.07834+0.83063i"), new Complex128("0.76763+0.26101i"), new Complex128("0.61068+0.5401i"), new Complex128("0.68096+0.10431i"), new Complex128("0.7411+0.29505i"), new Complex128("0.78902+0.40017i"), new Complex128("0.29019+0.90398i"), new Complex128("0.08451+0.94349i"), new Complex128("0.15869+0.17056i"), new Complex128("0.79893+0.59939i"), new Complex128("0.71202+0.00529i"), new Complex128("0.89399+0.9254i"), new Complex128("0.71421+0.37517i")};
        dRowIndices = new int[]{0, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5};
        dColIndices = new int[]{10, 3, 0, 2, 5, 7, 8, 0, 1, 3, 4, 10, 5, 7, 8, 12, 13};
        d = new CooCMatrix(dShape, dEntries, dRowIndices, dColIndices);

        CooCMatrix finalb = b;
        CooCMatrix finald = d;
        assertThrows(Exception.class, ()->finalb.augment(finald));
    }
}
