package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CsrCMatrixTriDiagTests {

    static CsrCMatrix A;
    static Shape aShape;
    static Complex128[] aEntries;
    static int[] aRowPointers;
    static int[] aColIndices;

    @Test
    void traceTest() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new Complex128[]{new Complex128(0.13392, 0.67581), new Complex128(0.08953, 0.43612),
                new Complex128(0.68598, 0.57467), new Complex128(0.19996, 0.4443), new Complex128(0.1453, 0.59643)};
        aRowPointers = new int[]{0, 1, 1, 2, 2, 5, 5};
        aColIndices = new int[]{5, 0, 0, 1, 3};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(new Complex128(0), A.trace());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new Complex128[]{new Complex128(0.90327, 0.45253), new Complex128(0.21721, 0.28695),
                new Complex128(0.65185, 0.93707), new Complex128(0.48592, 0.63105), new Complex128(0.96722, 0.76818)};
        aRowPointers = new int[]{0, 1, 1, 2, 4, 5, 5};
        aColIndices = new int[]{3, 5, 3, 4, 5};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(new Complex128(0.65185, 0.93707), A.trace());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(12, 12);
        aEntries = new Complex128[]{new Complex128(0.52333, 0.67155), new Complex128(0.79849, 0.05489),
                new Complex128(0.2229, 0.67036), new Complex128(0.65271, 0.51699), new Complex128(0.63722, 0.55373),
                new Complex128(0.21806, 0.5938), new Complex128(0.06624, 0.41699), new Complex128(0.32211, 0.1279),
                new Complex128(0.11324, 0.21277), new Complex128(0.45704, 0.75931), new Complex128(0.13948, 0.53299),
                new Complex128(0.13934, 0.59231), new Complex128(0.30193, 0.98664), new Complex128(0.11591, 0.98686),
                new Complex128(0.29993, 0.8055), new Complex128(0.43436, 0.64936), new Complex128(0.9495, 0.32514),
                new Complex128(0.22636, 0.4559), new Complex128(0.58931, 0.3885), new Complex128(0.332, 0.82381),
                new Complex128(0.11975, 0.11127), new Complex128(0.46906, 0.80406)};
        aRowPointers = new int[]{0, 4, 5, 7, 8, 9, 10, 12, 13, 15, 19, 21, 22};
        aColIndices = new int[]{2, 3, 5, 11, 10, 2, 11, 4, 8, 11, 8, 9, 0, 3, 10, 3, 6, 7, 11, 1, 4, 2};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(new Complex128(0.21806, 0.5938), A.trace());

        // ----------------------- sub-case 4 -----------------------
        assertThrows(IllegalArgumentException.class, ()->new CsrCMatrix(12, 4).tr());
    }


    @Test
    void getDiagTests() {
        CooCVector exp;

        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new Complex128[]{new Complex128(0.16488, 0.46447), new Complex128(0.3774, 0.35246),
                new Complex128(0.48798, 0.43226), new Complex128(0.48544, 0.0083), new Complex128(0.77148, 0.92185)};
        aRowPointers = new int[]{0, 0, 0, 1, 3, 5, 5};
        aColIndices = new int[]{2, 0, 5, 2, 4};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.16488, 0.46447),
                new Complex128(0.0, 0.0), new Complex128(0.77148, 0.92185), new Complex128(0.0, 0.0)).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new Complex128[]{new Complex128(0.3583, 0.11616), new Complex128(0.38055, 0.12452),
                new Complex128(0.97075, 0.0994), new Complex128(0.43841, 0.0319), new Complex128(0.78954, 0.72668)};
        aRowPointers = new int[]{0, 1, 3, 4, 4, 4, 5};
        aColIndices = new int[]{4, 0, 3, 5, 2};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(0, 0, 0, 0, 0, 0).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(12, 12);
        aEntries = new Complex128[]{new Complex128(0.13994, 0.01355), new Complex128(0.5696, 0.69936),
                new Complex128(0.1473, 0.92664), new Complex128(0.40113, 0.88561), new Complex128(0.45177, 0.86077),
                new Complex128(0.91248, 0.28304), new Complex128(0.71058, 0.73788), new Complex128(0.04337, 0.79422),
                new Complex128(0.26867, 0.54323), new Complex128(0.28884, 0.58676), new Complex128(0.62439, 0.17585),
                new Complex128(0.76891, 0.03482), new Complex128(0.02111, 0.58864), new Complex128(0.36376, 0.44496)};
        aRowPointers = new int[]{0, 1, 4, 5, 6, 8, 10, 11, 12, 13, 13, 14, 14};
        aColIndices = new int[]{7, 3, 5, 8, 6, 3, 6, 7, 3, 5, 11, 10, 8, 11};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0),
                new Complex128(0.91248, 0.28304), new Complex128(0.0, 0.0), new Complex128(0.28884, 0.58676),
                new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.02111, 0.58864),
                new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 4 -----------------------
        aShape = new Shape(3, 7);
        aEntries = new Complex128[]{new Complex128(0.85218, 0.08775), new Complex128(0.16499, 0.26153), new Complex128(0.36642, 0.63926), new Complex128(0.63494, 0.79019)};
        aRowPointers = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{0, 3, 5, 1};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new Complex128(0.85218, 0.08775), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 5 -----------------------
        aShape = new Shape(7, 3);
        aEntries = new Complex128[]{new Complex128(0.38881, 0.35982), new Complex128(0.82873, 0.85744),
                new Complex128(0.22975, 0.7268), new Complex128(0.9345, 0.92117)};
        aRowPointers = new int[]{0, 1, 1, 2, 2, 2, 3, 4};
        aColIndices = new int[]{0, 2, 2, 0};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new Complex128(0.38881, 0.35982), new Complex128(0.0, 0.0),
                new Complex128(0.82873, 0.85744)).toCoo();
        assertEquals(exp, A.getDiag());
    }


    @Test
    void isTril() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new Complex128[]{new Complex128(0.51678, 0.43084), new Complex128(0.03154, 0.8453), new Complex128(0.24399, 0.9955), new Complex128(0.29959, 0.05705), new Complex128(0.59463, 0.42417), new Complex128(0.96114, 0.42932), new Complex128(0.88379, 0.28427), new Complex128(0.71297, 0.26953), new Complex128(0.92992, 0.8928), new Complex128(0.92364, 0.45705)};
        aRowPointers = new int[]{0, 1, 4, 5, 8, 8, 8, 10};
        aColIndices = new int[]{4, 0, 3, 5, 6, 0, 2, 4, 2, 3};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertFalse(A.isTriL());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new Complex128[]{new Complex128(0.45131, 0.97724), new Complex128(0.44317, 0.67049), new Complex128(0.88543, 0.96506), new Complex128(0.75028, 0.11579)};
        aRowPointers = new int[]{0, 1, 1, 1, 2, 3, 3, 4};
        aColIndices = new int[]{0, 2, 0, 2};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriL());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(15, 15);
        aEntries = new Complex128[]{new Complex128(0.13761, 0.98915), new Complex128(0.90174, 0.96046), new Complex128(0.30444, 0.60086), new Complex128(0.03844, 0.04768), new Complex128(0.18296, 0.45909), new Complex128(0.8832, 0.52968), new Complex128(0.42703, 0.27357), new Complex128(0.63788, 0.67968), new Complex128(0.90985, 0.42655), new Complex128(0.90188, 0.85166), new Complex128(0.85113, 0.55835), new Complex128(0.20384, 0.24774), new Complex128(0.01717, 0.75604), new Complex128(0.22755, 0.18921), new Complex128(0.81342, 0.09213), new Complex128(0.03044, 0.32735), new Complex128(0.03832, 0.69054), new Complex128(0.93671, 0.28034), new Complex128(0.03274, 0.07978), new Complex128(0.6285, 0.47879), new Complex128(0.43616, 0.68511), new Complex128(0.65648, 0.5702)};
        aRowPointers = new int[]{0, 1, 1, 1, 2, 2, 4, 5, 6, 8, 10, 11, 12, 16, 21, 22};
        aColIndices = new int[]{0, 1, 2, 5, 5, 6, 2, 4, 1, 2, 4, 9, 0, 1, 3, 9, 0, 2, 3, 11, 13, 12};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriL());
    }


    @Test
    void isTriu() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new Complex128[]{new Complex128(0.48308, 0.47911), new Complex128(0.96946, 0.90557), new Complex128(0.32221, 0.25055), new Complex128(0.8551, 0.48406), new Complex128(0.86014, 0.35776), new Complex128(0.98441, 0.7564), new Complex128(0.62677, 0.99078), new Complex128(0.45817, 0.6006), new Complex128(0.92659, 0.21957), new Complex128(0.40475, 0.19932)};
        aRowPointers = new int[]{0, 2, 5, 8, 8, 10, 10, 10};
        aColIndices = new int[]{5, 6, 0, 1, 6, 3, 5, 6, 1, 2};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertFalse(A.isTriU());
        assertFalse(A.isTri());
        assertFalse(A.isDiag());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(7, 7);
        aEntries = new Complex128[]{new Complex128(0.35653, 0.19435), new Complex128(0.73519, 0.75719), new Complex128(0.67925, 0.9127), new Complex128(0.45195, 0.85737)};
        aRowPointers = new int[]{0, 2, 3, 3, 3, 4, 4, 4};
        aColIndices = new int[]{3, 4, 1, 6};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriU());
        assertTrue(A.isTri());
        assertFalse(A.isDiag());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(15, 15);
        aEntries = new Complex128[]{new Complex128(0.35213, 0.48809), new Complex128(0.9931, 0.30221), new Complex128(0.08806, 0.7989), new Complex128(0.51773, 0.35794), new Complex128(0.96909, 0.73786), new Complex128(0.38499, 0.53154), new Complex128(0.32305, 0.18345), new Complex128(0.08315, 0.69159), new Complex128(0.81898, 0.16558), new Complex128(0.12321, 0.44276), new Complex128(0.81688, 0.84221), new Complex128(0.32693, 0.53394), new Complex128(0.73328, 0.79251), new Complex128(0.95499, 0.88291), new Complex128(0.92087, 0.24149), new Complex128(0.29432, 0.31341), new Complex128(0.11148, 0.36242), new Complex128(0.04913, 0.56587), new Complex128(0.06358, 0.33869), new Complex128(0.56164, 0.51624)};
        aRowPointers = new int[]{0, 2, 6, 6, 10, 11, 11, 14, 15, 15, 16, 16, 18, 19, 20, 20};
        aColIndices = new int[]{4, 9, 1, 2, 8, 14, 4, 5, 7, 11, 14, 8, 12, 13, 7, 13, 12, 13, 14, 13};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertTrue(A.isTriU());
        assertTrue(A.isTri());
        assertFalse(A.isDiag());
    }
}
