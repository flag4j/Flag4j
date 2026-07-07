package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixSetTests {

    @Test
    void setTest() {
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
        aEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("0.96427+0.49577i"), new Complex128("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("445.0"), new Complex128("0.96427+0.49577i"), new Complex128("0.053+0.9829i")};
        expRowIndices = new int[]{0, 1, 1, 1, 2};
        expColIndices = new int[]{1, 1, 2, 4, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(445, 1, 2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new Complex128[]{new Complex128("0.4921+0.29279i"), new Complex128("0.13959+0.33724i"), new Complex128("0.60995+0.257i"), new Complex128("0.74365+0.27311i"), new Complex128("0.91068+0.75977i"), new Complex128("0.25598+0.80711i"), new Complex128("0.69927+0.9189i"), new Complex128("0.88488+0.94923i"), new Complex128("0.18563+0.32057i"), new Complex128("0.14302+0.76377i"), new Complex128("0.12337+0.50506i"), new Complex128("0.5966+0.54101i"), new Complex128("0.28354+0.38644i"), new Complex128("0.40568+0.14458i"), new Complex128("0.3541+0.37016i"), new Complex128("0.62362+0.73473i"), new Complex128("0.54154+0.11671i"), new Complex128("0.26052+0.17949i"), new Complex128("0.58631+0.78973i"), new Complex128("0.87975+0.01091i"), new Complex128("0.95361+0.07703i"), new Complex128("0.39905+0.00734i"), new Complex128("0.35175+0.76207i"), new Complex128("0.97743+0.43951i"), new Complex128("0.71768+0.2805i")};
        aRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        aColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(23, 11);
        expEntries = new Complex128[]{new Complex128("0.4921+0.29279i"), new Complex128("0.13959+0.33724i"), new Complex128("0.60995+0.257i"), new Complex128("0.74365+0.27311i"), new Complex128("0.91068+0.75977i"), new Complex128("0.25598+0.80711i"), new Complex128("0.69927+0.9189i"), new Complex128("0.88488+0.94923i"), new Complex128("0.18563+0.32057i"), new Complex128("0.14302+0.76377i"), new Complex128("0.12337+0.50506i"), new Complex128("0.5966+0.54101i"), new Complex128("0.28354+0.38644i"), new Complex128("0.40568+0.14458i"), new Complex128("0.3541+0.37016i"), new Complex128("0.62362+0.73473i"), new Complex128("-5.2"), new Complex128("0.54154+0.11671i"), new Complex128("0.26052+0.17949i"), new Complex128("0.58631+0.78973i"), new Complex128("0.87975+0.01091i"), new Complex128("0.95361+0.07703i"), new Complex128("0.39905+0.00734i"), new Complex128("0.35175+0.76207i"), new Complex128("0.97743+0.43951i"), new Complex128("0.71768+0.2805i")};
        expRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 15, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        expColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(-5.2, 15, 9));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 32156);
        aEntries = new Complex128[]{new Complex128("0.76483+0.33276i"), new Complex128("0.98077+0.54744i"), new Complex128("0.39142+0.33343i"), new Complex128("0.31495+0.65704i"), new Complex128("0.18212+0.0667i"), new Complex128("0.77918+0.62381i"), new Complex128("0.21766+0.26696i"), new Complex128("0.20254+0.50045i"), new Complex128("0.38428+0.85978i")};
        aRowIndices = new int[]{107, 245, 282, 363, 619, 779, 794, 887, 988};
        aColIndices = new int[]{3196, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1000, 32156);
        expEntries = new Complex128[]{new Complex128("0.76483+0.33276i"), new Complex128("7.2"), new Complex128("0.98077+0.54744i"), new Complex128("0.39142+0.33343i"), new Complex128("0.31495+0.65704i"), new Complex128("0.18212+0.0667i"), new Complex128("0.77918+0.62381i"), new Complex128("0.21766+0.26696i"), new Complex128("0.20254+0.50045i"), new Complex128("0.38428+0.85978i")};
        expRowIndices = new int[]{107, 234, 245, 282, 363, 619, 779, 794, 887, 988};
        expColIndices = new int[]{3196, 11002, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(7.2, 234, 11002));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.38871+0.73253i"), new Complex128("0.56447+0.3675i")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.set(1, 6, 1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.46549+0.31437i"), new Complex128("0.22736+0.59527i")};
        aRowIndices = new int[]{3, 4};
        aColIndices = new int[]{0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.set(1, 1, 9));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.26673+0.90314i"), new Complex128("0.87254+0.61073i")};
        aRowIndices = new int[]{1, 4};
        aColIndices = new int[]{1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.set(1, -1, 1));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.38003+0.55457i"), new Complex128("0.31182+0.59758i")};
        aRowIndices = new int[]{2, 4};
        aColIndices = new int[]{1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.set(1, 1, -1));

        // ---------------------  sub-case 8 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("0.96427+0.49577i"), new Complex128("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("0.96427+0.49577i"), new Complex128(34.8)};
        expRowIndices = new int[]{0, 1, 1, 2};
        expColIndices = new int[]{1, 1, 4, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(34.8, 2, 1));
    }


    @Test
    void setComplexTest() {
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
        aEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("0.96427+0.49577i"), new Complex128("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128(13, 4.99), new Complex128("0.96427+0.49577i"), new Complex128("0.053+0.9829i")};
        expRowIndices = new int[]{0, 1, 1, 1, 2};
        expColIndices = new int[]{1, 1, 2, 4, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new Complex128(13, 4.99), 1, 2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new Complex128[]{new Complex128("0.4921+0.29279i"), new Complex128("0.13959+0.33724i"), new Complex128("0.60995+0.257i"), new Complex128("0.74365+0.27311i"), new Complex128("0.91068+0.75977i"), new Complex128("0.25598+0.80711i"), new Complex128("0.69927+0.9189i"), new Complex128("0.88488+0.94923i"), new Complex128("0.18563+0.32057i"), new Complex128("0.14302+0.76377i"), new Complex128("0.12337+0.50506i"), new Complex128("0.5966+0.54101i"), new Complex128("0.28354+0.38644i"), new Complex128("0.40568+0.14458i"), new Complex128("0.3541+0.37016i"), new Complex128("0.62362+0.73473i"), new Complex128("0.54154+0.11671i"), new Complex128("0.26052+0.17949i"), new Complex128("0.58631+0.78973i"), new Complex128("0.87975+0.01091i"), new Complex128("0.95361+0.07703i"), new Complex128("0.39905+0.00734i"), new Complex128("0.35175+0.76207i"), new Complex128("0.97743+0.43951i"), new Complex128("0.71768+0.2805i")};
        aRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        aColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(23, 11);
        expEntries = new Complex128[]{new Complex128("0.4921+0.29279i"), new Complex128("0.13959+0.33724i"), new Complex128("0.60995+0.257i"), new Complex128("0.74365+0.27311i"), new Complex128("0.91068+0.75977i"), new Complex128("0.25598+0.80711i"), new Complex128("0.69927+0.9189i"), new Complex128("0.88488+0.94923i"), new Complex128("0.18563+0.32057i"), new Complex128("0.14302+0.76377i"), new Complex128("0.12337+0.50506i"), new Complex128("0.5966+0.54101i"), new Complex128("0.28354+0.38644i"), new Complex128("0.40568+0.14458i"), new Complex128("0.3541+0.37016i"), new Complex128("0.62362+0.73473i"), new Complex128(0, 24.1), new Complex128("0.54154+0.11671i"), new Complex128("0.26052+0.17949i"), new Complex128("0.58631+0.78973i"), new Complex128("0.87975+0.01091i"), new Complex128("0.95361+0.07703i"), new Complex128("0.39905+0.00734i"), new Complex128("0.35175+0.76207i"), new Complex128("0.97743+0.43951i"), new Complex128("0.71768+0.2805i")};
        expRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 15, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        expColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new Complex128(0, 24.1), 15, 9));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 32156);
        aEntries = new Complex128[]{new Complex128("0.76483+0.33276i"), new Complex128("0.98077+0.54744i"), new Complex128("0.39142+0.33343i"), new Complex128("0.31495+0.65704i"), new Complex128("0.18212+0.0667i"), new Complex128("0.77918+0.62381i"), new Complex128("0.21766+0.26696i"), new Complex128("0.20254+0.50045i"), new Complex128("0.38428+0.85978i")};
        aRowIndices = new int[]{107, 245, 282, 363, 619, 779, 794, 887, 988};
        aColIndices = new int[]{3196, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1000, 32156);
        expEntries = new Complex128[]{new Complex128("0.76483+0.33276i"), new Complex128(-1.4, -99.2), new Complex128("0.98077+0.54744i"), new Complex128("0.39142+0.33343i"), new Complex128("0.31495+0.65704i"), new Complex128("0.18212+0.0667i"), new Complex128("0.77918+0.62381i"), new Complex128("0.21766+0.26696i"), new Complex128("0.20254+0.50045i"), new Complex128("0.38428+0.85978i")};
        expRowIndices = new int[]{107, 234, 245, 282, 363, 619, 779, 794, 887, 988};
        expColIndices = new int[]{3196, 11002, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new Complex128(-1.4, -99.2), 234, 11002));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.38871+0.73253i"), new Complex128("0.56447+0.3675i")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.set(1, 6, 1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.46549+0.31437i"), new Complex128("0.22736+0.59527i")};
        aRowIndices = new int[]{3, 4};
        aColIndices = new int[]{0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.set(new Complex128(1, 1), 1, 9));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.26673+0.90314i"), new Complex128("0.87254+0.61073i")};
        aRowIndices = new int[]{1, 4};
        aColIndices = new int[]{1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.set(new Complex128(1, 1), -1, 1));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new Complex128[]{new Complex128("0.38003+0.55457i"), new Complex128("0.31182+0.59758i")};
        aRowIndices = new int[]{2, 4};
        aColIndices = new int[]{1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.set(new Complex128(1, 1), 1, -1));

        // ---------------------  sub-case 8 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("0.96427+0.49577i"), new Complex128("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.12119+0.74369i"), new Complex128("0.92925+0.0336i"), new Complex128("0.96427+0.49577i"), new Complex128(13, 4.99)};
        expRowIndices = new int[]{0, 1, 1, 2};
        expColIndices = new int[]{1, 1, 4, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new Complex128(13, 4.99), 2, 1));
    }
}
