/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j;

import org.flag4j.arrays.dense.Vector;

public class TempTesting {

    public static void main(String[] args) {
        Vector a = new Vector(new double[]{1, 2, 3});
        Vector b = new Vector(new double[]{-1.1, 2.2, -3.3});

        // ---- Comparisons ----
        a.equals(b);  // Element-wise exact numerical equality
        a.allClose(b);  // True if |a-b| <= (1E-08 + 1E-05*|b|) element-wise
        a.allClose(b, 1e-7, 1e-6); // True if |a-b| <= (1e-7 + 1e-6*|b|) element-wise
        a.isParallel(b);  // True if a and b are parallel
        a.isPerp(b);  // True if a and b are perpindicular

        // ---- Properties ----
        a.isAllFinite();  // True if ALL entries are finite
        a.containsInf();  // True if ANY entry is infinite
        a.containsNaN();  // True if ANY value is NaN
        a.isAllNeg();  // True if ALL values are negative
        a.isAllPos();  // True if ALL values are positive
        a.isAllOnes();  // True if ALL values are ones
        a.isAllZeros();  // True if ALL values are zeros

        // For complex vectors
//        a.isComplex();
//        a.isReal();

//        Matrix mat = new Matrix(new double[][]{
//                {1, 2, 3},
//                {4, 5, 6},
//                {7, 8, 9}});
//        CVector a = new CVector(new Complex128(0, 1),
//                new Complex128(2, 3),
//                new Complex128(4, 5));
//        CVector b = new CVector(new Complex128(-1.1, 2.2),
//                new Complex128(2.2, -3.3),
//                new Complex128(-4.4, 5.5));

//        // ----- Linalg and tensor ops -----
//        var d = a.dot(b);  // Vector dot product
//        d = a.inner(b);  // Vector inner product
//        double innerSelf = a.innerSelf();  // Vector inner product of `a` with itself
//        var c = a.cross(b);  // Vector cross-product (vectors must be length 3)
//
//        // ----- norms -----
//        Complex128 mag = a.mag();  // 2-norm
//        double norm = a.norm();  // 2-norm
//        norm = a.norm(2);  // 2-norm
//        norm = a.norm(1);  // 1-norm
//        norm = a.norm(Double.POSITIVE_INFINITY);  // maximum norm
//
//        // specify arbitrary p-norms
//        norm = a.norm(3.2);
//        norm = a.norm(-2.905);
//        norm = a.norm(Double.NEGATIVE_INFINITY);
    }
}
