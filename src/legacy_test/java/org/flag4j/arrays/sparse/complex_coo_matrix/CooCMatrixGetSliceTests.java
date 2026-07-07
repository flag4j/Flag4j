package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixGetSliceTests {

    @Test
    void getSliceTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.33773+0.88426i"), new Complex128("0.02816+0.65465i"), new Complex128("0.27934+0.52608i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{3, 4, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(2, 5);
        expEntries = new Complex128[]{new Complex128("0.27934+0.52608i")};
        expRowIndices = new int[]{0};
        expColIndices = new int[]{3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.getSlice(1, 3, 0, 5));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new Complex128[]{new Complex128("0.43994+0.07352i"), new Complex128("0.82499+0.10899i"), new Complex128("0.74053+0.53794i"), new Complex128("0.12124+0.95986i"), new Complex128("0.87873+0.48001i")};
        aRowIndices = new int[]{0, 1, 2, 7, 9};
        aColIndices = new int[]{1, 1, 15, 2, 16};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(6, 2);
        expEntries = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.getSlice(2, 8, 9, 11));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new Complex128[]{new Complex128("0.46464+0.79351i"), new Complex128("0.98213+0.25416i"), new Complex128("0.77128+0.84765i"), new Complex128("0.64964+0.31303i"), new Complex128("0.50435+0.23437i"), new Complex128("0.12206+0.02502i"), new Complex128("0.46309+0.77862i"), new Complex128("0.41095+0.28293i"), new Complex128("0.44977+0.64556i"), new Complex128("0.05415+0.31609i"), new Complex128("0.36894+0.45128i"), new Complex128("0.37491+0.80106i"), new Complex128("0.70345+0.23871i"), new Complex128("0.49137+0.07681i"), new Complex128("0.36961+0.09297i"), new Complex128("0.53207+0.59729i"), new Complex128("0.57691+0.90888i"), new Complex128("0.89953+0.79917i"), new Complex128("0.99015+0.86375i"), new Complex128("0.67917+0.2598i"), new Complex128("0.13047+0.19129i"), new Complex128("0.61695+0.49238i"), new Complex128("0.7335+0.60326i"), new Complex128("0.61287+0.38448i"), new Complex128("0.40295+0.48659i"), new Complex128("0.78436+0.03043i"), new Complex128("0.39923+0.33141i"), new Complex128("0.30404+0.45666i"), new Complex128("0.97371+0.37412i"), new Complex128("0.11395+0.27795i"), new Complex128("0.39418+0.86901i"), new Complex128("0.54765+0.61628i"), new Complex128("0.9016+0.78159i"), new Complex128("0.18087+0.38666i"), new Complex128("0.15718+0.0762i"), new Complex128("0.99891+0.22251i"), new Complex128("0.86968+0.71794i"), new Complex128("0.53375+0.56806i"), new Complex128("0.43581+0.38262i"), new Complex128("0.76399+0.406i"), new Complex128("0.40403+0.40351i"), new Complex128("0.41428+0.26309i"), new Complex128("0.4167+0.64441i"), new Complex128("0.17623+0.61847i"), new Complex128("0.32269+0.69253i"), new Complex128("0.36361+0.20824i"), new Complex128("0.04418+0.77549i"), new Complex128("0.19683+0.76965i"), new Complex128("0.0318+0.76005i"), new Complex128("0.03521+0.34343i"), new Complex128("0.4995+0.55016i"), new Complex128("0.16115+0.45417i"), new Complex128("0.39745+0.74596i"), new Complex128("0.08716+0.47403i"), new Complex128("0.23352+0.42345i"), new Complex128("0.27351+0.70656i"), new Complex128("0.57461+0.78692i"), new Complex128("0.74864+0.70278i"), new Complex128("0.07117+0.52322i"), new Complex128("0.04518+0.08786i"), new Complex128("0.61565+0.95192i"), new Complex128("0.7337+0.45525i"), new Complex128("0.49011+0.53482i"), new Complex128("0.41676+0.82087i"), new Complex128("0.55944+0.75964i"), new Complex128("0.89415+0.16034i"), new Complex128("0.96627+0.13485i"), new Complex128("0.54719+0.78764i"), new Complex128("0.05208+0.09381i"), new Complex128("0.63664+0.62852i"), new Complex128("0.42961+0.72674i"), new Complex128("0.1263+0.12728i"), new Complex128("0.20177+0.90459i"), new Complex128("0.33223+0.18979i"), new Complex128("0.95159+0.12382i"), new Complex128("0.35476+0.70323i"), new Complex128("0.00417+0.52988i"), new Complex128("0.05027+0.77044i"), new Complex128("0.09176+0.28579i"), new Complex128("0.31417+0.70709i"), new Complex128("0.12943+0.70069i"), new Complex128("0.7408+0.93089i"), new Complex128("0.15254+0.00615i"), new Complex128("0.25411+0.3934i"), new Complex128("0.74607+0.47682i"), new Complex128("0.37316+0.30878i"), new Complex128("0.90373+0.87499i"), new Complex128("0.5778+0.62946i"), new Complex128("0.75169+0.31657i"), new Complex128("0.4191+0.84715i"), new Complex128("0.24835+0.24237i"), new Complex128("0.52994+0.55078i"), new Complex128("0.8066+0.60089i"), new Complex128("0.52349+0.86194i"), new Complex128("0.8645+0.97998i"), new Complex128("0.38479+0.26912i"), new Complex128("0.1653+0.18391i"), new Complex128("0.07626+0.46427i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{5, 198, 203, 212, 275, 280, 435, 465, 537, 590, 592, 686, 690, 708, 719, 721, 775, 66, 79, 82, 131, 187, 241, 244, 261, 305, 320, 346, 388, 418, 477, 575, 680, 728, 774, 957, 988, 15, 74, 183, 218, 325, 326, 396, 443, 472, 506, 519, 605, 610, 719, 793, 847, 854, 871, 956, 2, 58, 72, 164, 167, 189, 308, 339, 381, 404, 422, 475, 489, 524, 525, 529, 662, 756, 783, 834, 956, 128, 169, 175, 220, 243, 338, 362, 409, 453, 534, 540, 547, 594, 641, 657, 732, 752, 841, 932, 957, 973};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 37);
        expEntries = new Complex128[]{new Complex128("0.99015+0.86375i"), new Complex128("0.67917+0.2598i"), new Complex128("0.43581+0.38262i")};
        expRowIndices = new int[]{1, 1, 2};
        expColIndices = new int[]{7, 10, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.getSlice(0, 3, 72, 109));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.49203+0.27876i"), new Complex128("0.13602+0.10076i"), new Complex128("0.73285+0.89175i"), new Complex128("0.53981+0.59873i")};
        aRowIndices = new int[]{1, 1, 3, 4};
        aColIndices = new int[]{0, 2, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getSlice(-1, 2, 1, 2));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.67135+0.88834i"), new Complex128("0.90098+0.52257i"), new Complex128("0.47715+0.29783i"), new Complex128("0.20484+0.18195i")};
        aRowIndices = new int[]{1, 1, 3, 4};
        aColIndices = new int[]{1, 2, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getSlice(0, 1, -1, 2));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.81497+0.55398i"), new Complex128("0.7668+0.2618i"), new Complex128("0.79522+0.83472i"), new Complex128("0.95094+0.82128i")};
        aRowIndices = new int[]{2, 3, 4, 4};
        aColIndices = new int[]{1, 0, 0, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.getSlice(0, 6, 0, 1));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.45463+0.77134i"), new Complex128("0.892+0.12117i"), new Complex128("0.2369+0.76664i"), new Complex128("0.87665+0.68521i")};
        aRowIndices = new int[]{0, 0, 2, 3};
        aColIndices = new int[]{0, 2, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.getSlice(0, 2, 0, 40));
    }
}
