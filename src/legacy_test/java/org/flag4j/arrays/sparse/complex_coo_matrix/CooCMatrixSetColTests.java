package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixSetColTests {

    @Test
    void setColTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bIndices;
        Complex128[] bEntries;
        CooCVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.8969+0.65632i"), new Complex128("0.44955+0.52831i"), new Complex128("0.61332+0.4159i")};
        aRowIndices = new int[]{0, 0, 0};
        aColIndices = new int[]{0, 1, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new Complex128[]{new Complex128("0.7937+0.18442i")};
        bIndices = new int[]{1};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.44955+0.52831i"), new Complex128("0.61332+0.4159i"), new Complex128("0.7937+0.18442i")};
        expRowIndices = new int[]{0, 0, 1};
        expColIndices = new int[]{1, 4, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new Complex128[]{new Complex128("0.14783+0.979i"), new Complex128("0.24607+0.9379i"), new Complex128("0.59171+0.21038i"), new Complex128("0.08942+0.19028i"), new Complex128("0.3154+0.99695i")};
        aRowIndices = new int[]{1, 3, 7, 9, 10};
        aColIndices = new int[]{8, 11, 9, 3, 17};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11);
        bEntries = new Complex128[]{new Complex128("0.27382+0.15016i"), new Complex128("0.71287+0.96394i"), new Complex128("0.02+0.81414i"), new Complex128("0.34607+0.36712i")};
        bIndices = new int[]{0, 2, 7, 10};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(11, 23);
        expEntries = new Complex128[]{new Complex128("0.27382+0.15016i"), new Complex128("0.14783+0.979i"), new Complex128("0.71287+0.96394i"), new Complex128("0.24607+0.9379i"), new Complex128("0.02+0.81414i"), new Complex128("0.59171+0.21038i"), new Complex128("0.08942+0.19028i"), new Complex128("0.34607+0.36712i"), new Complex128("0.3154+0.99695i")};
        expRowIndices = new int[]{0, 1, 2, 3, 7, 7, 9, 10, 10};
        expColIndices = new int[]{6, 8, 6, 11, 6, 9, 3, 6, 17};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 6));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new Complex128[]{new Complex128("0.40108+0.20157i"), new Complex128("0.06329+0.35963i"), new Complex128("0.30048+0.19493i"), new Complex128("0.77913+0.49546i"), new Complex128("0.44026+0.50983i"), new Complex128("0.8898+0.04999i"), new Complex128("0.76201+0.8372i"), new Complex128("0.4491+0.55776i"), new Complex128("0.33349+0.84622i"), new Complex128("0.53084+0.75426i"), new Complex128("0.8051+0.60392i"), new Complex128("0.24584+0.2645i"), new Complex128("0.00087+0.6845i"), new Complex128("0.84881+0.82459i"), new Complex128("0.74433+0.5689i"), new Complex128("0.50837+0.7612i"), new Complex128("0.21847+0.62454i"), new Complex128("0.46747+0.08047i"), new Complex128("0.99323+0.17234i"), new Complex128("0.09636+0.99651i"), new Complex128("0.73467+0.27681i"), new Complex128("0.59151+0.41083i"), new Complex128("0.64217+0.61058i"), new Complex128("0.18733+0.53352i"), new Complex128("0.24633+0.43535i"), new Complex128("0.39783+0.69325i"), new Complex128("0.18177+0.05664i"), new Complex128("0.56967+0.42902i"), new Complex128("0.43933+0.16761i"), new Complex128("0.51236+0.47152i"), new Complex128("0.15577+0.02981i"), new Complex128("0.04133+0.00795i"), new Complex128("0.45155+0.55251i"), new Complex128("0.91551+0.41024i"), new Complex128("0.92948+0.36421i"), new Complex128("0.85228+0.0383i"), new Complex128("0.87495+0.62375i"), new Complex128("0.96061+0.89504i"), new Complex128("0.44373+0.21953i"), new Complex128("0.80214+0.71303i"), new Complex128("0.66831+0.20498i"), new Complex128("0.52174+0.18296i"), new Complex128("0.54523+0.20421i"), new Complex128("0.23072+0.94682i"), new Complex128("0.20653+0.2705i"), new Complex128("0.78021+0.97566i"), new Complex128("0.48083+0.21072i"), new Complex128("0.24483+0.23042i"), new Complex128("0.34011+0.34358i"), new Complex128("0.19875+0.75521i"), new Complex128("0.28207+0.69059i"), new Complex128("0.47211+0.12439i"), new Complex128("0.87843+0.54793i"), new Complex128("0.22343+0.3172i"), new Complex128("0.65505+0.6357i"), new Complex128("0.91364+0.00219i"), new Complex128("0.96748+0.72511i"), new Complex128("0.36055+0.92089i"), new Complex128("0.57674+0.12894i"), new Complex128("0.78033+0.60651i"), new Complex128("0.00551+0.2861i"), new Complex128("0.98859+0.32837i"), new Complex128("0.53356+0.61318i"), new Complex128("0.92697+0.67302i"), new Complex128("0.27655+0.26212i"), new Complex128("0.39756+0.94816i"), new Complex128("0.17353+0.80238i"), new Complex128("0.55777+0.74899i"), new Complex128("0.06685+0.24266i"), new Complex128("0.05017+0.32321i"), new Complex128("0.8217+0.15646i"), new Complex128("0.7867+0.71882i"), new Complex128("0.19771+0.67689i"), new Complex128("0.39419+0.54151i"), new Complex128("0.25364+0.28836i"), new Complex128("0.51256+0.28559i"), new Complex128("0.90397+0.5678i"), new Complex128("0.6858+0.47496i"), new Complex128("0.45589+0.43801i"), new Complex128("0.64739+0.01515i"), new Complex128("0.36916+0.57014i"), new Complex128("0.35399+0.27049i"), new Complex128("0.25753+0.73246i"), new Complex128("0.24983+0.05406i"), new Complex128("0.51121+0.45062i"), new Complex128("0.39576+0.64992i"), new Complex128("0.19051+0.97617i"), new Complex128("0.41709+0.90475i"), new Complex128("0.47649+0.10058i"), new Complex128("0.55394+0.99386i"), new Complex128("0.04258+0.47874i"), new Complex128("0.48511+0.53381i"), new Complex128("0.5789+0.8816i"), new Complex128("0.1921+0.32532i"), new Complex128("0.48501+0.36391i"), new Complex128("0.43698+0.97725i"), new Complex128("0.75557+0.5387i"), new Complex128("0.56365+0.7452i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{107, 118, 156, 259, 261, 306, 447, 455, 525, 599, 623, 724, 797, 903, 929, 967, 974, 989, 43, 59, 100, 104, 126, 129, 184, 267, 290, 312, 482, 486, 487, 620, 676, 683, 895, 896, 912, 932, 941, 952, 39, 71, 103, 177, 226, 256, 367, 381, 392, 422, 569, 588, 590, 608, 619, 669, 672, 687, 707, 763, 824, 891, 921, 67, 155, 190, 250, 258, 290, 301, 343, 417, 455, 488, 504, 578, 617, 792, 818, 842, 974, 68, 87, 144, 280, 329, 392, 445, 475, 478, 479, 497, 534, 550, 628, 718, 868, 982};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new Complex128[]{new Complex128("0.73879+0.06946i"), new Complex128("0.45811+0.14543i")};
        bIndices = new int[]{0, 4};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        expShape = new Shape(5, 1000);
        expEntries = new Complex128[]{new Complex128("0.40108+0.20157i"), new Complex128("0.06329+0.35963i"), new Complex128("0.30048+0.19493i"), new Complex128("0.77913+0.49546i"), new Complex128("0.44026+0.50983i"), new Complex128("0.8898+0.04999i"), new Complex128("0.76201+0.8372i"), new Complex128("0.4491+0.55776i"), new Complex128("0.33349+0.84622i"), new Complex128("0.53084+0.75426i"), new Complex128("0.8051+0.60392i"), new Complex128("0.24584+0.2645i"), new Complex128("0.00087+0.6845i"), new Complex128("0.84881+0.82459i"), new Complex128("0.74433+0.5689i"), new Complex128("0.50837+0.7612i"), new Complex128("0.21847+0.62454i"), new Complex128("0.46747+0.08047i"), new Complex128("0.73879+0.06946i"), new Complex128("0.99323+0.17234i"), new Complex128("0.09636+0.99651i"), new Complex128("0.73467+0.27681i"), new Complex128("0.59151+0.41083i"), new Complex128("0.64217+0.61058i"), new Complex128("0.18733+0.53352i"), new Complex128("0.24633+0.43535i"), new Complex128("0.39783+0.69325i"), new Complex128("0.18177+0.05664i"), new Complex128("0.56967+0.42902i"), new Complex128("0.43933+0.16761i"), new Complex128("0.51236+0.47152i"), new Complex128("0.15577+0.02981i"), new Complex128("0.04133+0.00795i"), new Complex128("0.45155+0.55251i"), new Complex128("0.91551+0.41024i"), new Complex128("0.92948+0.36421i"), new Complex128("0.85228+0.0383i"), new Complex128("0.87495+0.62375i"), new Complex128("0.96061+0.89504i"), new Complex128("0.44373+0.21953i"), new Complex128("0.80214+0.71303i"), new Complex128("0.66831+0.20498i"), new Complex128("0.52174+0.18296i"), new Complex128("0.54523+0.20421i"), new Complex128("0.23072+0.94682i"), new Complex128("0.20653+0.2705i"), new Complex128("0.78021+0.97566i"), new Complex128("0.48083+0.21072i"), new Complex128("0.24483+0.23042i"), new Complex128("0.34011+0.34358i"), new Complex128("0.19875+0.75521i"), new Complex128("0.28207+0.69059i"), new Complex128("0.47211+0.12439i"), new Complex128("0.87843+0.54793i"), new Complex128("0.22343+0.3172i"), new Complex128("0.65505+0.6357i"), new Complex128("0.91364+0.00219i"), new Complex128("0.96748+0.72511i"), new Complex128("0.36055+0.92089i"), new Complex128("0.57674+0.12894i"), new Complex128("0.78033+0.60651i"), new Complex128("0.00551+0.2861i"), new Complex128("0.98859+0.32837i"), new Complex128("0.53356+0.61318i"), new Complex128("0.92697+0.67302i"), new Complex128("0.27655+0.26212i"), new Complex128("0.39756+0.94816i"), new Complex128("0.17353+0.80238i"), new Complex128("0.55777+0.74899i"), new Complex128("0.06685+0.24266i"), new Complex128("0.05017+0.32321i"), new Complex128("0.8217+0.15646i"), new Complex128("0.7867+0.71882i"), new Complex128("0.19771+0.67689i"), new Complex128("0.39419+0.54151i"), new Complex128("0.25364+0.28836i"), new Complex128("0.51256+0.28559i"), new Complex128("0.90397+0.5678i"), new Complex128("0.6858+0.47496i"), new Complex128("0.45589+0.43801i"), new Complex128("0.64739+0.01515i"), new Complex128("0.36916+0.57014i"), new Complex128("0.35399+0.27049i"), new Complex128("0.25753+0.73246i"), new Complex128("0.24983+0.05406i"), new Complex128("0.51121+0.45062i"), new Complex128("0.39576+0.64992i"), new Complex128("0.19051+0.97617i"), new Complex128("0.41709+0.90475i"), new Complex128("0.47649+0.10058i"), new Complex128("0.55394+0.99386i"), new Complex128("0.04258+0.47874i"), new Complex128("0.48511+0.53381i"), new Complex128("0.5789+0.8816i"), new Complex128("0.1921+0.32532i"), new Complex128("0.48501+0.36391i"), new Complex128("0.43698+0.97725i"), new Complex128("0.75557+0.5387i"), new Complex128("0.56365+0.7452i"), new Complex128("0.45811+0.14543i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        expColIndices = new int[]{107, 118, 156, 259, 261, 306, 447, 455, 525, 599, 623, 724, 797, 903, 929, 967, 974, 989, 999, 43, 59, 100, 104, 126, 129, 184, 267, 290, 312, 482, 486, 487, 620, 676, 683, 895, 896, 912, 932, 941, 952, 39, 71, 103, 177, 226, 256, 367, 381, 392, 422, 569, 588, 590, 608, 619, 669, 672, 687, 707, 763, 824, 891, 921, 67, 155, 190, 250, 258, 290, 301, 343, 417, 455, 488, 504, 578, 617, 792, 818, 842, 974, 68, 87, 144, 280, 329, 392, 445, 475, 478, 479, 497, 534, 550, 628, 718, 868, 982, 999};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.70609+0.77623i"), new Complex128("0.61771+0.37191i"), new Complex128("0.93683+0.4622i"), new Complex128("0.22006+0.06678i")};
        aRowIndices = new int[]{1, 1, 2, 4};
        aColIndices = new int[]{0, 1, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new Complex128[]{new Complex128("0.26721+0.56832i"), new Complex128("0.09961+0.57491i")};
        bIndices = new int[]{2, 3};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        CooCMatrix final0a = a;
        CooCVector final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, -1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.3543+0.92325i"), new Complex128("0.01246+0.59718i"), new Complex128("0.65951+0.75695i"), new Complex128("0.5218+0.61097i")};
        aRowIndices = new int[]{1, 2, 2, 4};
        aColIndices = new int[]{1, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new Complex128[]{new Complex128("0.26676+0.40491i"), new Complex128("0.41776+0.60048i")};
        bIndices = new int[]{1, 2};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        CooCMatrix final1a = a;
        CooCVector final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 4));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.44339+0.35944i"), new Complex128("0.9204+0.18754i"), new Complex128("0.17881+0.76557i"), new Complex128("0.60444+0.86329i")};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(7);
        bEntries = new Complex128[]{new Complex128("0.29086+0.80883i"), new Complex128("0.05468+0.08505i")};
        bIndices = new int[]{0, 2};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        CooCMatrix final2a = a;
        CooCVector final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 0));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.5686+0.2999i"), new Complex128("0.86547+0.85179i"), new Complex128("0.66343+0.4857i"), new Complex128("0.84483+0.58998i")};
        aRowIndices = new int[]{0, 1, 3, 3};
        aColIndices = new int[]{2, 2, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new Complex128[]{new Complex128("0.56015+0.91847i")};
        bIndices = new int[]{1};
        b = new CooCVector(bShape.getSize(0), bEntries, bIndices);

        CooCMatrix final3a = a;
        CooCVector final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 0));
    }
}
