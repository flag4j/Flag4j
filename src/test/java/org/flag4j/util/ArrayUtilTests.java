package org.flag4j.util;

import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ArrayUtilTests {
    Integer[] srcI;
    Double[] srcD;
    String[] srcS;
    Complex128[] expArr, actArr;
    double[] expRange;
    int[] expRangeI;

    @Test
    void copy2Complex128TestCase() {
        // -------------- sub-case 1 --------------
        srcI = new Integer[]{1, 2, 3, 4, 5, -1334, 14};
        expArr = new Complex128[]{
                new Complex128(1), new Complex128(2), new Complex128(3),
                new Complex128(4), new Complex128(5), new Complex128(-1334),
                new Complex128(14)};
        actArr = new Complex128[srcI.length];
        ArrayConversions.toComplex128(srcI, actArr);
        assertArrayEquals(expArr, actArr);

        // -------------- sub-case 2 --------------
        srcD = new Double[]{1.133, 2.445, 3.133, 4.1, 5223.334, -1334.0001, 14.};
        expArr = new Complex128[]{
                new Complex128(1.133), new Complex128(2.445), new Complex128(3.133),
                new Complex128(4.1), new Complex128(5223.334), new Complex128(-1334.0001),
                new Complex128(14.)};
        actArr = new Complex128[srcD.length];
        ArrayConversions.toComplex128(srcD, actArr);
        assertArrayEquals(expArr, actArr);

        // -------------- sub-case 3 --------------
        srcS = new String[]{"3+1i", "2.01334i", "-44.1-4i"};
        expArr = new Complex128[]{new Complex128("3+1i"), new Complex128("2.01334i"), new Complex128("-44.1-4i")};
        actArr = new Complex128[srcS.length];
        ArrayConversions.toComplex128(srcS, actArr);
        assertArrayEquals(expArr, actArr);
    }

    @Test
    void swapTestCase() {
        // -------------- sub-case 1 --------------
        int i = 0;
        int j = 2;
        double[] srcd = {1.0000001, 2, -3, 4.133};
        double[] expd = {-3, 2, 1.0000001, 4.133};
        ArrayUtils.swap(srcd, i, j);
        assertArrayEquals(expd, srcd);

        // -------------- sub-case 2 --------------
        i = 3;
        j = 1;
        srcd = new double[]{1.0000001, 2, -3, 4.133};
        expd = new double[]{1.0000001, 4.133, -3, 2};
        ArrayUtils.swap(srcd, i, j);
        assertArrayEquals(expd, srcd);

        // -------------- sub-case 3 --------------
        i = 3;
        j = 1;
        Complex128[] srcC = new Complex128[]{
                new Complex128(1.0000001), new Complex128(2),
                new Complex128(-3), new Complex128(4.133)};
        Complex128[] expC = new Complex128[]{
                new Complex128(1.0000001), new Complex128(4.133),
                new Complex128(-3), new Complex128(2)};
        ArrayUtils.swap(srcC, i, j);
        assertArrayEquals(expC, srcC);
    }


    @Test
    void rangeTestCase() {
        // -------------- sub-case 1 --------------
        expRange = new double[]{-1, 0, 1, 2, 3, 4, 5, 6, 7};
        assertArrayEquals(expRange, ArrayBuilder.range(-1, 8));

        // -------------- sub-case 2 --------------
        expRange = new double[]{99, 100, 101, 102};
        assertArrayEquals(expRange, ArrayBuilder.range(99, 103));

        // -------------- sub-case 3 --------------
        assertThrows(IllegalArgumentException.class, ()-> ArrayBuilder.range(5, 1));
    }


    @Test
    void copyStridedRangeTestCase() {
        int[] src1 = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        int[] src2 = {0, 1};
        int[] src3 = {};
        int[] src4 = {1};

        // -------------- sub-cases 1 --------------
        expRangeI = new int[]{2, 9};
        assertArrayEquals(expRangeI, ArrayUtils.copyOfStridedRange(src1, 1, 13, 6));

        expRangeI = new int[]{};
        assertArrayEquals(expRangeI, ArrayUtils.copyOfStridedRange(src1, 1, 1, 0));

        expRangeI = new int[]{13};
        assertArrayEquals(expRangeI, ArrayUtils.copyOfStridedRange(src1, 11, src1.length, 100));
    }
}
