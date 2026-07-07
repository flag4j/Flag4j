package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CsrMatrixTriDiagTests {

    static CsrMatrix A;
    static Shape aShape;
    static double[] aEntries;
    static int[] aRowPointers;
    static int[] aColIndices;

    @Test
    void traceTest() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.550651808914819, 0.25248902903617765, 0.7347711759989852, 0.6355561756397741, 0.4229489382733195};
        aRowPointers = new int[]{0, 1, 3, 3, 4, 4, 5};
        aColIndices = new int[]{1, 0, 3, 4, 2};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(0.0, A.trace());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.3947350230967598, 0.057362465663433504, 0.9347345786066202, 0.43387856849809536, 0.644454913626892};
        aRowPointers = new int[]{0, 2, 2, 2, 3, 3, 5};
        aColIndices = new int[]{2, 5, 3, 0, 3};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(0.9347345786066202, A.trace());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.8086884822224377, 0.8263266272760342, 0.32525649651881317, 0.9608789667749497, 0.4395049603468911,
                0.07305897091345792, 0.7678491080726771, 0.2294487507606635, 0.1382399153004038, 0.920903220380491, 0.6895253484430522,
                0.9256640817501549, 0.8520473820197568, 0.6642547057142691, 0.9992544305209986, 0.6144764840301362,
                0.38299169954163803, 0.04890094352081187, 0.07595056410539092, 0.6778873127860436, 0.5725538366855386,
                0.633943368653018};
        aRowPointers = new int[]{0, 2, 5, 5, 6, 9, 10, 13, 14, 15, 15, 16, 17, 20, 21, 22};
        aColIndices = new int[]{8, 9, 1, 2, 13, 14, 0, 11, 13, 7, 8, 12, 14, 8, 14, 5, 4, 1, 4, 12, 9, 9};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(1.0031438093048568, A.trace());

        // ----------------------- sub-case 4 -----------------------
        assertThrows(LinearAlgebraException.class, ()->new CsrMatrix(12, 4).tr());
    }


    @Test
    void getDiagTests() {
        CooVector exp;

        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.2742696420912182, 0.07751690909235676, 0.921830928702064, 0.1935807720906103, 0.7540620527220376};
        aRowPointers = new int[]{0, 0, 3, 3, 4, 5, 5};
        aColIndices = new int[]{0, 2, 4, 5, 4};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.0, 0.0, 0.0, 0.7540620527220376, 0.0).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.5117934899539107, 0.5232442605382139, 0.6241020346691114, 0.5149866370978797, 0.9899310737230431, 0.18946387228386474, 0.2824144399493015, 0.25137905573116825, 0.9609185162323945, 0.6746428794940532, 0.47997193451501086};
        aRowPointers = new int[]{0, 2, 5, 7, 9, 10, 11};
        aColIndices = new int[]{2, 5, 0, 1, 3, 2, 5, 0, 4, 1, 5};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.5149866370978797, 0.18946387228386474, 0.0, 0.0, 0.47997193451501086).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(12, 12);
        aEntries = new double[]{0.989336440083382, 0.5784226184530201, 0.4128615264146236, 0.8720518156451399, 0.015332897118263578, 0.30602701869807314, 0.5445047402785209, 0.6799626535731805, 0.8160519253929756, 0.44554133866387846, 0.5507111109163054, 0.5127437616463539, 0.12630665534801888, 0.8348974142473102};
        aRowPointers = new int[]{0, 1, 3, 5, 8, 9, 9, 10, 12, 12, 13, 14, 14};
        aColIndices = new int[]{6, 6, 8, 7, 10, 3, 6, 8, 4, 5, 3, 5, 3, 4};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.0, 0.0, 0.30602701869807314, 0.8160519253929756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 4 -----------------------
        aShape = new Shape(3, 7);
        aEntries = new double[]{0.46451579858863223, 0.761818146777147, 0.4513656938478876, 0.8045385713145264};
        aRowPointers = new int[]{0, 0, 2, 4};
        aColIndices = new int[]{1, 4, 1, 2};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.46451579858863223, 0.8045385713145264).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 5 -----------------------
        aShape = new Shape(7, 3);
        aEntries = new double[]{0.11343649359086139, 0.6082353444109962, 0.8703305006626817, 0.9546036038577662};
        aRowPointers = new int[]{0, 0, 2, 3, 4, 4, 4, 4};
        aColIndices = new int[]{1, 2, 2, 0};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.11343649359086139, 0.8703305006626817).toCoo();
        assertEquals(exp, A.getDiag());
    }


    @Test
    void isTril() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new double[]{0.03536046897095657, 0.5401385386889258, 0.7777684051892619, 0.24675745878877275, 0.025876533684179748, 0.16436028377484557, 0.6986274933624915, 0.9582940000149774, 0.5783087709774752, 0.020358148422358502};
        aRowPointers = new int[]{0, 5, 5, 5, 6, 7, 8, 10};
        aColIndices = new int[]{0, 2, 3, 4, 6, 3, 6, 0, 3, 6};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertFalse(A.isTriL());
        assertFalse(A.isTri());
        assertFalse(A.isDiag());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new double[]{0.9448565131258886, 0.6803882440455985, 0.3181498543027419, 0.653448286895677, 0.2363592026109237, 0.9248502624213077};
        aRowPointers = new int[]{0, 1, 1, 2, 3, 3, 5, 6};
        aColIndices = new int[]{0, 0, 2, 3, 5, 6};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriL());
        assertTrue(A.isTri());
        assertFalse(A.isDiag());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.8603981477375947, 0.3046683606558588, 0.11007027331099051, 0.4746507283151713, 0.5667955808993239, 0.5046332521751881, 0.2932302578424171, 0.7544118059856361, 0.6761113255197329, 0.7148365185969104, 0.8096087554064826, 0.5648390522079458, 0.39403660727620915, 0.6793611400411638, 0.13042393139511033, 0.4201967094552399, 0.7219889285611188, 0.9091132773462233, 0.22640551340497106, 0.6306376367101162, 0.10724657397258852, 0.7358566619041913, 0.7897474246945047, 0.339673026544825};
        aRowPointers = new int[]{0, 0, 0, 1, 2, 2, 5, 8, 8, 11, 14, 16, 19, 20, 23, 24};
        aColIndices = new int[]{0, 3, 0, 2, 5, 3, 4, 5, 0, 6, 8, 0, 1, 9, 3, 7, 0, 6, 11, 2, 3, 4, 7, 7};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriL());
        assertTrue(A.isTri());
        assertFalse(A.isDiag());
    }


    @Test
    void isTriu() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new double[]{0.9686346841636623, 0.21877442651375112, 0.9763371061919456, 0.06324864925948748, 0.7610961776714811, 0.17623031131300626, 0.8633163244725658, 0.3809902603585521, 0.13481705095409913, 0.594263171632082};
        aRowPointers = new int[]{0, 1, 2, 3, 4, 6, 7, 10};
        aColIndices = new int[]{2, 5, 6, 6, 0, 5, 2, 1, 3, 5};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertFalse(A.isTriU());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new double[]{0.987498054693959, 0.6886396714454422, 0.6524880599582122, 0.4483694388056334};
        aRowPointers = new int[]{0, 1, 1, 3, 4, 4, 4, 4};
        aColIndices = new int[]{4, 5, 6, 5};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriU());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.09392690937841508, 0.21659312610254544, 0.4640714558267387, 0.9901076587629565, 0.2346497714607133, 0.5023928929582768, 0.8875779242688255, 0.7027977268176925, 0.880291816596573, 0.7876547304826765, 0.738558875148028, 0.16881160559126562, 0.6373264799198883, 0.34413716558933816, 0.703597158746133, 0.87658904659461, 0.5731522410829861, 0.6999013255587964, 0.044857605428713865, 0.38105445036449503, 0.5722480748744379, 0.244200831204898, 0.9875219217112591, 0.7197306512009739, 0.13121023238087792};
        aRowPointers = new int[]{0, 2, 3, 5, 9, 11, 14, 15, 15, 19, 22, 23, 23, 24, 24, 25};
        aColIndices = new int[]{2, 7, 11, 5, 6, 3, 5, 9, 11, 9, 11, 5, 9, 12, 10, 8, 9, 10, 14, 9, 10, 12, 12, 12, 14};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriU());
    }
}
