package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.util.ArrayBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CsrCMatrixTransposeTests {

    static CsrCMatrix A;
    static Complex128[][] aEntries;
    static CsrCMatrix exp;
    static Complex128[][] expEntries;


    private static void buildFromDense() {
        A = new CMatrix(aEntries).toCsr();
        exp = new CMatrix(expEntries).toCsr();
    }

    private static void buildFromDense(boolean buildExp) {
        A = new CMatrix(aEntries).toCsr();
        if(buildExp) exp = new CMatrix(expEntries).toCsr();
    }


    @Test
    void transposeTests() {
        // -------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.89855+0.38209i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.86937+0.81929i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16951+0.68135i"), new Complex128("0.0")},
                {new Complex128("0.65696+0.92685i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.27084+0.48366i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16383+0.02238i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16657+0.97044i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.86937+0.81929i"), new Complex128("0.65696+0.92685i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.89855+0.38209i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16657+0.97044i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16383+0.02238i"), new Complex128("0.0"),
                        new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.16951+0.68135i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.27084+0.48366i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        buildFromDense();

        assertEquals(exp, A.T());

        // -------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.98031+0.659i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.55568+0.08748i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.93371+0.94355i"), new Complex128("0.0"), new Complex128("0.46316+0.37268i"), new Complex128("0.0"), new Complex128("0.0")}};
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.93371+0.94355i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.55568+0.08748i"), new Complex128("0.46316+0.37268i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.98031+0.659i"), new Complex128("0.0"), new Complex128("0.0")}};
        buildFromDense();

        assertEquals(exp, A.T());

        // -------------------- sub-case 3 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.35993+0.77312i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.90417+0.16239i")},
                {new Complex128("0.27902+0.20495i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.58847+0.90372i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        expEntries = new Complex128[][]{
                {new Complex128("0.35993+0.77312i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.27902+0.20495i"), new Complex128("0.58847+0.90372i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.90417+0.16239i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        buildFromDense();

        assertEquals(exp, A.T());

        // -------------------- sub-case 4 ---------------------
        aEntries = new Complex128[140][23];
        ArrayBuilder.fill(aEntries, Complex128.ZERO);
        expEntries = new Complex128[23][140];
        ArrayBuilder.fill(expEntries, Complex128.ZERO);
        buildFromDense();

        assertEquals(exp, A.T());
    }


    @Test
    void hermitianTransposeTests() {
        // -------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.89855+0.38209i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.86937+0.81929i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16951+0.68135i"), new Complex128("0.0")},
                {new Complex128("0.65696+0.92685i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.27084-0.48366i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16383+0.02238i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16657+0.97044i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.86937-0.81929i"), new Complex128("0.65696-0.92685i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.89855-0.38209i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16657-0.97044i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.16383-0.02238i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.16951-0.68135i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.27084+0.48366i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.0")}};
        buildFromDense();

        assertEquals(exp, A.H());

        // -------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.98031+0.659i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.55568+0.08748i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.93371+0.94355i"), new Complex128("0.0"), new Complex128("0.46316+0.37268i"), new Complex128("0.0"), new Complex128("0.0")}};
        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.93371-0.94355i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.55568-0.08748i"), new Complex128("0.46316-0.37268i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.98031-0.659i"), new Complex128("0.0"), new Complex128("0.0")}};
        buildFromDense();

        assertEquals(exp, A.H());

        // -------------------- sub-case 3 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.35993+0.77312i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.90417+0.16239i")},
                {new Complex128("0.27902-0.20495i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.58847+0.90372i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        expEntries = new Complex128[][]{
                {new Complex128("0.35993-0.77312i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.27902+0.20495i"),
                        new Complex128("0.58847-0.90372i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.90417-0.16239i"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.0")}};
        buildFromDense();

        assertEquals(exp, A.H());

        // -------------------- sub-case 4 ---------------------
        aEntries = new Complex128[140][23];
        ArrayBuilder.fill(aEntries, Complex128.ZERO);
        expEntries = new Complex128[23][140];
        ArrayBuilder.fill(expEntries, Complex128.ZERO);
        buildFromDense();

        assertEquals(exp, A.H());
    }


    @Test
    void isHermitianTests() {
        // -------------------- sub-case 1 ---------------------
        aEntries = new Complex128[5][5];
        ArrayBuilder.fill(aEntries, Complex128.ZERO);
        aEntries[0][0] = new Complex128(1.5);
        aEntries[2][1] = new Complex128(2.45, 85.12);
        aEntries[1][2] = new Complex128(2.45, -85.12);
        aEntries[3][4] = new Complex128(0, -1);
        aEntries[4][3] = new Complex128(0, 1);
        aEntries[3][3] = new Complex128(9.24);
        buildFromDense(false);

        assertTrue(A.isHermitian());

        // -------------------- sub-case 2 ---------------------
        aEntries = new Complex128[5][6];
        ArrayBuilder.fill(aEntries, Complex128.ZERO);
        aEntries[0][0] = new Complex128(1.5);
        aEntries[2][1] = new Complex128(2.45, 85.12);
        aEntries[1][2] = new Complex128(2.45, -85.12);
        aEntries[3][4] = new Complex128(0, -1);
        aEntries[4][3] = new Complex128(0, 1);
        aEntries[3][3] = new Complex128(9.24);
        buildFromDense(false);

        assertFalse(A.isHermitian());

        // -------------------- sub-case 1 ---------------------
        aEntries = new Complex128[415][415];
        ArrayBuilder.fill(aEntries, Complex128.ZERO);
        aEntries[0][0] = new Complex128(87.35);
        aEntries[2][1] = new Complex128(9671.4, -774.1);
        aEntries[1][2] = new Complex128(9671.4, 774.1);
        aEntries[3][4] = new Complex128(-87834.2, 12.5);
        aEntries[4][3] = new Complex128(-87834.2, -12.5);
        aEntries[3][3] = new Complex128(400);
        aEntries[45][2] = new Complex128(0, 124.34346);
        aEntries[2][45] = new Complex128(0, -124.34346);
        aEntries[13][44] = new Complex128(-93.15, 15.12355);
        aEntries[44][13] = new Complex128(-93.15, -15.12355);
        aEntries[255][255] = new Complex128(40.1);
        aEntries[304][34] = new Complex128(8);
        aEntries[34][304] = new Complex128(8);
        aEntries[304][55] = new Complex128(345.4, -88.3);
        aEntries[55][304] = new Complex128(345.4, 88.3);
        aEntries[400][400] = new Complex128(0);
        aEntries[402][0] = new Complex128(9.3, 15);
        aEntries[0][402] = new Complex128(9.3, -15);

        buildFromDense(false);

        assertTrue(A.isHermitian());
    }
}
