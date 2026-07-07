package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrCMatrixToDenseTests {
    static CsrCMatrix A;
    static Complex128[][] aEntries;

    static CMatrix exp;

    static void build() {
        A = new CMatrix(aEntries).toCsr();
        exp = new CMatrix(aEntries);
    }


    @Test
    void toDenseTests() {
        // --------------------- sub-case 1 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.62854+0.22343i"), new Complex128("0.44136+0.15692i"), new Complex128("0.69915+0.87818i"), new Complex128("0.74834+0.32125i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.27069+0.73057i"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.7623+0.12432i"), new Complex128("0.0")}};
        build();

        assertEquals(exp, A.toDense());

        // --------------------- sub-case 2 ---------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.3778+0.2475i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.60458+0.65914i"), new Complex128("0.0"), new Complex128("0.08514+0.96172i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.35664+0.90759i"), new Complex128("0.18187+0.30447i"), new Complex128("0.0"),
                        new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.67137+0.11747i")}};
        build();

        assertEquals(exp, A.toDense());
    }
}
