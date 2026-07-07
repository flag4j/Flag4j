/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class CVectorEqualsTests {

    Complex128[] aEntries;
    CVector a;
    int sparseSize;
    int[] sparseIndices;


    @Test
    void complexDenseTestCase(){
        Complex128[] bEntries;
        CVector b;

        // ----------------- sub-case 1 -----------------
        aEntries = new Complex128[]{new Complex128(1, -9.234), new Complex128(0, 8.245),
                new Complex128(1.3), Complex128.ZERO};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{new Complex128(1, -9.234), new Complex128(0, 8.245),
                new Complex128(1.3), Complex128.ZERO};
        b = new CVector(bEntries);
        assertEquals(a, b);

        // ----------------- sub-case 2 -----------------
        aEntries = new Complex128[]{new Complex128(8.124, 9.4), new Complex128(1.55),
                new Complex128(0, -85.215), new Complex128(0.000013, 14.5),
                new Complex128(1.335676, -89345)};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{new Complex128(8.124, 9.4), new Complex128(1.55),
                new Complex128(0, -85.215), new Complex128(0.000013, 14.5),
                new Complex128(1.335676, -89345)};
        b = new CVector(bEntries);
        assertEquals(a, b);

        // ----------------- sub-case 3 -----------------
        aEntries = new Complex128[]{new Complex128(8.124, 9.4), new Complex128(1.55),
                new Complex128(0, -85.215), new Complex128(0.000013, 14.5),
                new Complex128(1.335676, -89345)};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{new Complex128(8.124, 9.4), new Complex128(1.55),
                new Complex128(0, -85.215), new Complex128(0.000013, 14.5)};
        b = new CVector(bEntries);
        assertNotEquals(a, b);

        // ----------------- sub-case 4 -----------------
        a = new CVector(45, -92341.566);
        b = new CVector(45, -92341.566);
        assertEquals(a, b);

        // ----------------- sub-case 5 -----------------
        a = new CVector(45, -92341.566);
        b = new CVector(41, -92341.566);
        assertNotEquals(a, b);

        // ----------------- sub-case 6 -----------------
        a = new CVector(45, new Complex128("92.1465+879234.9999324i"));
        b = new CVector(45, new Complex128("92.1465+879234.9999324i"));
        assertEquals(a, b);

        // ----------------- sub-case 7 -----------------
        a = new CVector(45, new Complex128("92.1465+879234.9999324i"));
        b = new CVector(41, new Complex128("92.1465+879234.9999324i"));
        assertNotEquals(a, b);

        // ----------------- sub-case 8 -----------------
        aEntries = new Complex128[]{new Complex128(1, -9.234), new Complex128(0, 8.245),
                new Complex128(1.3), Complex128.ZERO};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{new Complex128(1, -9.2341), new Complex128(0, 8.245),
                new Complex128(1.3), Complex128.ZERO};
        b = new CVector(bEntries);
        assertNotEquals(a, b);

        // ----------------- sub-case 9 -----------------
        aEntries = new Complex128[]{new Complex128(1, -9.234), new Complex128(0, 8.245),
                new Complex128(1.3), Complex128.ZERO};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{new Complex128(1, -9.234), new Complex128(0, 8.245),
                new Complex128(1.3), new Complex128(1.2)};
        b = new CVector(bEntries);
        assertNotEquals(a, b);
    }


    @Test
    void objectTestCase() {
        // ----------------- sub-case 1 -----------------
        aEntries = new Complex128[]{new Complex128(0), new Complex128(8.245, 9.2165),
                new Complex128(1.3, -0.000023465), Complex128.ZERO};
        a = new CVector(aEntries);
        String bString = "Hello World!";
        assertNotEquals(a, bString);

        // ----------------- sub-case 2 -----------------
        aEntries = new Complex128[]{new Complex128(0), new Complex128(8.245, 9.2165),
                new Complex128(1.3, -0.000023465), Complex128.ZERO};
        a = new CVector(aEntries);
        Double num = 123.4;
        assertNotEquals(a, num);

        // ----------------- sub-case 3 -----------------
        aEntries = new Complex128[]{new Complex128(0), new Complex128(8.245, 9.2165),
                new Complex128(1.3, -0.000023465), Complex128.ZERO};
        a = new CVector(aEntries);
        Complex128[] arr = {new Complex128(0), new Complex128(8.245, 9.2165),
                new Complex128(1.3, -0.000023465), Complex128.ZERO};
        assertNotEquals(a, num);
    }
}
