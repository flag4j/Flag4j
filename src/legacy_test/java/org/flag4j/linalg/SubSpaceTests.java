package org.flag4j.linalg;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.BeforeAll;


class SubSpaceTests {

    static double[][] aEntries, expAEntries;
    static Complex128[][] bEntries, expBEntries;

    static Matrix A, expA;
    static CMatrix B, expB;


    @BeforeAll
    static void setup() {
        aEntries = new double[][]{
                {2.4, 5.6, -0.35, 1.5},
                {34.6, 0, 2.4, 2},
                {-912.5, 15, 25.2, -0.3}};

        bEntries = new Complex128[][]{
                {new Complex128(44.5, -9.43), new Complex128(0, 2.45)},
                {Complex128.ZERO, new Complex128(-0.03, 2.5)}};

        A = new Matrix(aEntries);
        B = new CMatrix(bEntries);
    }


    // TODO: There are some issues with SVD. Bring back these tests once the SVD has been fixed.
//    @Test
//    void colSpaceTestCase() {
//        // -------------------- sub-case 1 --------------------
//        expAEntries = new double[][]{
//                {-0.002536286923471442, 0.2528788760188392},
//                {-0.03778044483704439, -0.9668312284180349},
//                {0.9992828454629634, -0.03591169474032637}
//        };
//        expA = new Matrix(expAEntries);
//
//        assertEquals(expA, SubSpace.getColSpace(A));
//
//        // -------------------- sub-case 2 --------------------
//        expBEntries = new Complex128[][]{
//                {new Complex128(-0.005034449960744143, 0.9999829442968625)},
//                {new Complex128(-5.042897264281646E-5, 0.0029602172030675465)}
//        };
//        expB = new CMatrix(expBEntries);
//
//        assertEquals(expB, SubSpace.getColSpace(B));
//    }
//
//
//    @Test
//    void rowSpaceTestCase() {
//        // -------------------- sub-case 1 --------------------
//        expAEntries = new double[][]{
//                {-0.9994883094236742, -0.020218901157440943},
//                {0.016390769560750296, 0.2333428640994128},
//                {0.027464336473866702, -0.8812728776030876}
//        };
//        expA = new Matrix(expAEntries);
//
//        assertEquals(expA, SubSpace.getRowSpace(A));
//
//        // -------------------- sub-case 2 --------------------
//        expBEntries = new Complex128[][]{
//                {new Complex128(-0.21192004255855398, 0.975797068015846)},
//                {new Complex128(0.05394352470531291, 2.7158060783330235E-4)}
//        };
//        expB = new CMatrix(expBEntries);
//
//        assertEquals(expB, SubSpace.getRowSpace(B));
//    }
//
//
//    @Test
//    void nullSpaceTestCase() {
//        // -------------------- sub-case 1 --------------------
//        expAEntries = new double[][]{
//                {0.006867348168070286},
//                {0.7309485843076149},
//                {-0.17161305691162324},
//                {0.66046647554988}
//        };
//        expA = new Matrix(expAEntries);
//
//        assertEquals(expA, SubSpace.getNullSpace(A));
//
//        // -------------------- sub-case 2 --------------------
//        expBEntries = new Complex128[][]{
//                {new Complex128(0.021880100969376936, -0.04930759368727217)},
//                {new Complex128(0.976851607231742, 0.2070047338470861)}
//        };
//        expB = new CMatrix(expBEntries);
//
//        assertEquals(expB, SubSpace.getNullSpace(B));
//    }
//
//
//    @Test
//    void leftNullSpaceTestCase() {
//        // -------------------- sub-case 1 --------------------
//        expAEntries = new double[][]{
//                {-0.9674946208181681},
//                {-0.2526064404238114},
//                {-0.012006037827724012}
//        };
//        expA = new Matrix(expAEntries);
//
//        assertEquals(expA, SubSpace.getLeftNullSpace(A));
//
//        // -------------------- sub-case 2 --------------------
//        expBEntries = new Complex128[][]{
//                {new Complex128(6.830867186744245E-4, 0.0028807675201914145)},
//                {new Complex128(-0.21902916510606682, -0.9757138204432532)}
//        };
//        expB = new CMatrix(expBEntries);
//
//        assertEquals(expB, SubSpace.getLeftNullSpace(B));
//    }
//
//
//    @Test
//    void hasEqualSpan() {
//        // -------------------- sub-case 1 --------------------
//        assertTrue(
//                SubSpace.hasEqualSpan(A.getSlice(0, A.numRows, 0, A.numRows),
//                        SubSpace.getColSpace(A))
//        );
//
//        // -------------------- sub-case 2 --------------------
//        assertTrue(
//                SubSpace.hasEqualSpan(B.getSlice(0, B.numRows, 0, B.numRows),
//                        SubSpace.getColSpace(B))
//        );
//    }
}
