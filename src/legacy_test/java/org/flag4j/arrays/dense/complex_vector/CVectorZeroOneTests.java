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

import org.flag4j.arrays.dense.CVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CVectorZeroOneTests {

    Complex128[] aEntries;
    CVector a;

    @Test
    void zerosTestCase() {
        // ------------------ sub-case 1 ------------------
        a = new CVector(34);
        assertTrue(a.isAllZeros());

        // ------------------ sub-case 2 ------------------
        a = new CVector(0);
        assertTrue(a.isAllZeros());

        // ------------------ sub-case 3 ------------------
        aEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO, Complex128.ZERO};
        a = new CVector(aEntries);
        assertTrue(a.isAllZeros());

        // ------------------ sub-case 4 ------------------
        aEntries = new Complex128[]{Complex128.ZERO, new Complex128(1), Complex128.ZERO};
        a = new CVector(aEntries);
        assertFalse(a.isAllZeros());

        // ------------------ sub-case 5 ------------------
        aEntries = new Complex128[]{Complex128.ZERO, new Complex128(9.4, -6.233), Complex128.ZERO};
        a = new CVector(aEntries);
        assertFalse(a.isAllZeros());

        // ------------------ sub-case 6 ------------------
        aEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO, new Complex128(0, -8.234)};
        a = new CVector(aEntries);
        assertFalse(a.isAllZeros());
    }


    @Test
    void onesTestCase() {
        // ------------------ sub-case 1 ------------------
        a = new CVector(34, 1);
        assertTrue(a.isAllOnes());

        // ------------------ sub-case 2 ------------------
        a = new CVector(0, 1);
        assertTrue(a.isAllOnes());

        // ------------------ sub-case 3 ------------------
        aEntries = new Complex128[]{new Complex128(1), new Complex128(1), new Complex128(1)};
        a = new CVector(aEntries);
        assertTrue(a.isAllOnes());

        // ------------------ sub-case 4 ------------------
        aEntries = new Complex128[]{new Complex128(1), new Complex128(1.2), new Complex128(1)};
        a = new CVector(aEntries);
        assertFalse(a.isAllOnes());

        // ------------------ sub-case 5 ------------------
        aEntries = new Complex128[]{new Complex128(1), new Complex128(5.3, 91.3), new Complex128(1)};
        a = new CVector(aEntries);
        assertFalse(a.isAllOnes());

        // ------------------ sub-case 6 ------------------
        aEntries = new Complex128[]{new Complex128(1), new Complex128(1), new Complex128(1, -1)};
        a = new CVector(aEntries);
        assertFalse(a.isAllOnes());
    }
}
