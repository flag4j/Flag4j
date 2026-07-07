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

package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.decompositions.hess.ComplexHess;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexHessenburgTests {
    String[][] aEntries;
    CMatrix A, Q, H, A_hat;

    ComplexHess hess;

    @Test
    void hessDecompTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new String[][]{
                {"1.55-2i", "0", "i"},
                {"25.66-90.25i", "34.5", "3.4+2i"},
                {"-i", "3.4-2i", "16.67+9.2i"}};
        A = new CMatrix(aEntries);
        hess = new ComplexHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).mult(Q.H());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1e-12));

        // ----------------------- sub-case 1.1 -----------------------
        hess = new ComplexHess();
        hess.decompose(A);

        assertEquals(H, hess.getH());

        // ----------------------- sub-case 2 -----------------------
        aEntries = new String[][]{
                {"1-2i", "0", "-16i", "6-9i"},
                {"4i", "6", "0", "4+3i"},
                {"22+9i", "0", "0", "1+i"},
                {"6+9i", "-25-4i", "1-i", "-1.2+3i"}};
        A = new CMatrix(aEntries);
        hess = new ComplexHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).mult(Q.H());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1e-12));

        // ----------------------- sub-case 2.1 -----------------------
        hess = new ComplexHess();
        hess.decompose(A);

        assertEquals(H, hess.getH());
    }
}
