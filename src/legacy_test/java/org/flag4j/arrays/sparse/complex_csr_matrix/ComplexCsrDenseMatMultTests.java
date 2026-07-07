package org.flag4j.arrays.sparse.complex_csr_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexCsrDenseMatMultTests {
    static CsrCMatrix A;
    static CMatrix aDense;
    static Complex128[][] aEntries;
    static Matrix Breal;
    static CMatrix B;
    static double[][] bRealEntries;
    static Complex128[][] bComplexEntries;
    static CMatrix exp;

    private static void buildReal(boolean... args) {
        aDense = new CMatrix(aEntries);
        A = aDense.toCsr();
        Breal = new Matrix(bRealEntries);
        if(args.length != 1 || args[0]) exp = aDense.mult(Breal);
    }

    private static void buildComplex(boolean... args) {
        aDense = new CMatrix(aEntries);
        A = aDense.toCsr();
        B = new CMatrix(bComplexEntries);
        if(args.length != 1 || args[0]) exp = aDense.mult(B);
    }


    @Test
    void multRealDenseTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80.1, 2.5)},
                {new Complex128(0), new Complex128(1.41, -92.2), new Complex128(0), new Complex128(0, 15.5), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.25, 23.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(-999.1155, 2.25), new Complex128(-1, 1)}};
        bRealEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541},
                {0.6462, 0.36597},
                {0.18312, 0.77178},
                {0.40715, 0.35642}};
        buildReal();

        assertEquals(exp, A.mult(Breal));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bRealEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541},
                {0.6462, 0.36597}};
        buildReal();

        assertEquals(exp, A.mult(Breal));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bRealEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541}};
        buildReal(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(Breal));
    }


    @Test
    void multComplexDenseTests() {
        // ---------------------- sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80.1, 2.5)},
                {new Complex128(0), new Complex128(1.41, -92.2), new Complex128(0), new Complex128(0, 15.5), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.25, 23.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(-999.1155, 2.25), new Complex128(-1, 1)}};
        bComplexEntries = new Complex128[][]{
                {new Complex128(0.60886, 0.33378), new Complex128(0.00204, 0.66152)},
                {new Complex128(0.11395, 0.22798), new Complex128(0.85626, 0.48514)},
                {new Complex128(0.63642, 0.52434), new Complex128(0.95994, 0.9354)},
                {new Complex128(0.19401, 0.93407), new Complex128(0.64822, 0.24427)},
                {new Complex128(0.49749, 0.11432), new Complex128(0.06738, 0.73179)},
                {new Complex128(0.08942, 0.10066), new Complex128(0.02026, 0.06551)}};
        buildComplex();
        assertEquals(exp, A.mult(B));

        // ---------------------- sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bComplexEntries = new Complex128[][]{
                {new Complex128(0.66751, 0.11856), new Complex128(0.98271, 0.49906)},
                {new Complex128(0.14152, 0.98128), new Complex128(0.30904, 0.21053)},
                {new Complex128(0.28185, 0.28402), new Complex128(0.76892, 0.97375)},
                {new Complex128(0.44435, 0.06128), new Complex128(0.57068, 0.89705)}};
        buildComplex();
        assertEquals(exp, A.mult(B));

        // ---------------------- sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bComplexEntries = new Complex128[][]{
                {new Complex128(0.57033, 0.74092), new Complex128(0.62504, 0.25253)},
                {new Complex128(0.69264, 0.37406), new Complex128(0.29895, 0.17085)},
                {new Complex128(0.95162, 0.22682), new Complex128(0.30524, 0.91462)}};
        buildComplex(false);
        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
