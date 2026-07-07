package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorTransposeTest {
    static int[] aAxes;

    static Complex128[] aEntries;
    static Complex128[] expEntries;

    static Shape aShape;
    static Shape expShape;

    static CTensor A;
    static CTensor exp;

    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{
                new Complex128("1.4415+9.14i"), new Complex128("235.61-9.865i"), new Complex128("-0.00024+5.15i"),
                new Complex128("1.0"), new Complex128("-0.0-85.1i"), new Complex128("1.345+3.5i"),
                new Complex128("0.014+0.01i"), new Complex128("-140.0-0.0235i"), new Complex128("1.5+9.24i"),
                new Complex128("51.0"), new Complex128("6.1-265.55i"), new Complex128("-0.00014+4.14i")};
        aShape = new Shape(3, 2, 1, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void transposeTestCase() {
        // -------------------- sub-case 1 --------------------
        expEntries = new Complex128[]{
                new Complex128("1.4415+9.14i"), new Complex128("-0.0-85.1i"), new Complex128("1.5+9.24i"),
                new Complex128("-0.00024+5.15i"), new Complex128("0.014+0.01i"), new Complex128("6.1-265.55i"),
                new Complex128("235.61-9.865i"), new Complex128("1.345+3.5i"), new Complex128("51.0"),
                new Complex128("1.0"), new Complex128("-140.0-0.0235i"), new Complex128("-0.00014+4.14i")};
        expShape = new Shape(2, 2, 1, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T());

        // -------------------- sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new Complex128[]{
                new Complex128("1.4415+9.14i"), new Complex128("-0.00024+5.15i"), new Complex128("235.61-9.865i"),
                new Complex128("1.0"), new Complex128("-0.0-85.1i"), new Complex128("0.014+0.01i"),
                new Complex128("1.345+3.5i"), new Complex128("-140.0-0.0235i"), new Complex128("1.5+9.24i"),
                new Complex128("6.1-265.55i"), new Complex128("51.0"), new Complex128("-0.00014+4.14i")};
        expShape = new Shape(3, 1, 2, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T(aAxes));

        // -------------------- sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new Complex128[]{
                new Complex128("1.4415+9.14i"), new Complex128("-0.0-85.1i"), new Complex128("1.5+9.24i"),
                new Complex128("-0.00024+5.15i"), new Complex128("0.014+0.01i"), new Complex128("6.1-265.55i"),
                new Complex128("235.61-9.865i"), new Complex128("1.345+3.5i"), new Complex128("51.0"),
                new Complex128("1.0"), new Complex128("-140.0-0.0235i"), new Complex128("-0.00014+4.14i")};
        expShape = new Shape(2, 1, 2, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T(aAxes));

        // -------------------- sub-case 4 --------------------
        expEntries = new Complex128[]{
                new Complex128("1.4415+9.14i"), new Complex128("-0.00024+5.15i"), new Complex128("235.61-9.865i"),
                new Complex128("1.0"), new Complex128("-0.0-85.1i"), new Complex128("0.014+0.01i"),
                new Complex128("1.345+3.5i"), new Complex128("-140.0-0.0235i"), new Complex128("1.5+9.24i"),
                new Complex128("6.1-265.55i"), new Complex128("51.0"), new Complex128("-0.00014+4.14i")};
        expShape = new Shape(3, 2, 1, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T(3, 1));

        // -------------------- sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.T(aAxes));

        // -------------------- sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.T(aAxes));

        // -------------------- sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(LinearAlgebraException.class, ()->A.T(aAxes));

        // -------------------- sub-case 8 --------------------
        assertThrows(LinearAlgebraException.class, ()->A.T(-1, 0));

        // -------------------- sub-case 9 --------------------
        assertThrows(LinearAlgebraException.class, ()->A.T(1, 6));
    }
}
