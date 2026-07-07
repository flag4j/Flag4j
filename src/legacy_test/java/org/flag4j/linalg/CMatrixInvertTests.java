package org.flag4j.linalg;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixInvertTests {

    static CMatrix A;
    static CMatrix exp;
    static Complex128[][] entries;
    static Complex128[][] expEntries;


    @Test
    void invTriUTests() {
        // --------------------- sub-case 1 ---------------------
        entries = new Complex128[][]{
                {new Complex128("0.72964+0.77161i"), new Complex128("0.04017+0.35861i")},
                {new Complex128("0.0"), new Complex128("0.02885+0.61375i")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[][]{
                {new Complex128(0.6469836227593948, -0.6841991025127141), new Complex128(-0.35324900268232806, 0.4255132196349395)},
                {new Complex128(0.0, 0.0), new Complex128(0.07641951197016296, -1.6257357182560666)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, Invert.invTriU(A));

        // --------------------- sub-case 2 ---------------------
        entries = new Complex128[][]{
                {new Complex128("0.88984+0.74576i"), new Complex128("0.35899+0.67567i"), new Complex128("0.67057+0.03486i"), new Complex128("0.27239+0.14667i")},
                {new Complex128("0.0"), new Complex128("0.29455+0.74263i"), new Complex128("0.99331+0.19522i"), new Complex128("0.89592+0.76741i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.70384+0.74177i"), new Complex128("0.59081+0.73518i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.97421+0.61448i")}};
        A = new CMatrix(entries);
        expEntries = new Complex128[][]{
                {new Complex128(0.6601318170773559, -0.5532454192929167), new Complex128(-0.569755474276714, 0.5964889359681658), new Complex128(0.0538261672936495, -0.24610680231262014), new Complex128(0.39210142039761287, -0.18312414720644973)},
                {new Complex128(0.0, 0.0), new Complex128(0.46148975736667824, -1.1635244899447168), new Complex128(0.29451765428159893, 1.2036600935687118), new Complex128(-0.5481358478041402, 0.10001294703720921)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.6731359287847971, -0.709411283664894), new Complex128(-0.7101078401778858, 0.3701442953886289)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.7343268609204316, -0.4631744382611417)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, Invert.invTriU(A));
    }


    @Test
    void invTriLTests() {
        // --------------------- sub-case 1 ---------------------
        entries = new Complex128[][]{
                {new Complex128("0.72964+0.77161i"), new Complex128("0.04017+0.35861i")},
                {new Complex128("0.0"), new Complex128("0.02885+0.61375i")}};
        A = new CMatrix(entries).T();
        expEntries = new Complex128[][]{
                {new Complex128(0.6469836227593948, -0.6841991025127141), new Complex128(-0.35324900268232806, 0.4255132196349395)},
                {new Complex128(0.0, 0.0), new Complex128(0.07641951197016296, -1.6257357182560666)}};
        exp = new CMatrix(expEntries).T();

        assertEquals(exp, Invert.invTriL(A));

        // --------------------- sub-case 2 ---------------------
        entries = new Complex128[][]{
                {new Complex128("0.88984+0.74576i"), new Complex128("0.35899+0.67567i"), new Complex128("0.67057+0.03486i"), new Complex128("0.27239+0.14667i")},
                {new Complex128("0.0"), new Complex128("0.29455+0.74263i"), new Complex128("0.99331+0.19522i"), new Complex128("0.89592+0.76741i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.70384+0.74177i"), new Complex128("0.59081+0.73518i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.97421+0.61448i")}};
        A = new CMatrix(entries).T();
        expEntries = new Complex128[][]{
                {new Complex128(0.6601318170773559, -0.5532454192929167), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(-0.5697554742767139, 0.5964889359681657), new Complex128(0.46148975736667824, -1.1635244899447168), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.05382616729364944, -0.2461068023126199), new Complex128(0.29451765428159893, 1.203660093568712), new Complex128(0.6731359287847971, -0.709411283664894), new Complex128(0.0, 0.0)},
                {new Complex128(0.39210142039761287, -0.18312414720644984), new Complex128(-0.5481358478041398, 0.10001294703720916), new Complex128(-0.7101078401778858, 0.3701442953886289), new Complex128(0.7343268609204316, -0.4631744382611417)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, Invert.invTriL(A));
    }


    @Test
    void invDiagTests() {
        // --------------------- sub-case 1 ---------------------
        entries = new Complex128[][]{
                {new Complex128(-14.43, 95.1), Complex128.ZERO},
                {Complex128.ZERO, new Complex128(0, 1.45)}};
        A = new CMatrix(entries);
        expEntries = new Complex128[][]{
                {new Complex128(-14.43, 95.1).invert(), Complex128.ZERO},
                {Complex128.ZERO, new Complex128(0, 1.45).invert()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, Invert.invDiag(A));

        // --------------------- sub-case 2 ---------------------
        entries = new Complex128[][]{
                {new Complex128(-14.43, 95.1), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128(0, 1.45), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, new Complex128(234.156)}};
        A = new CMatrix(entries);
        expEntries = new Complex128[][]{
                {new Complex128(-14.43, 95.1).invert(), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128(0, 1.45).invert(), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, new Complex128(234.156).invert()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, Invert.invDiag(A));
    }
}
