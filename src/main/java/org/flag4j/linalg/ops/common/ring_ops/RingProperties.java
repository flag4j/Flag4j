/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.linalg.ops.common.ring_ops;

import org.flag4j.numbers.Ring;

/**
 * Utility class for checking properties of {@link Ring Ring's}
 */
public final class RingProperties {
    
    private RingProperties() {
        // Hide default constructor in utility class.
    }


    /**
     * Checks if all data of two arrays are "close".
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays have the same length and all data are "close" element-wise; i.e.,
     * elements {@code a} and {@code b} at the same positions in the two arrays respectively and satisfy
     * {@code |a-b| <= (1E-08 + 1E-05*|b|)}. Otherwise, returns false.
     * @see #allClose(Ring[], Ring[], double, double)
     */
    public static <T extends Ring<T>> boolean allClose(T[] src1, T[] src2) {
        return allClose(src1, src2, 1e-05, 1e-08);
    }


    /**
     * Checks if all data of two arrays are "close".
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @param relTol The relative tolerance to determine if values are close.
     * @param absTol The absolute tolerance to determine if values are close.
     * @return True if both arrays have the same length and all data are "close" element-wise; i.e.,
     * elements {@code a} and {@code b} at the same positions in the two arrays respectively and satisfy
     * {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     * @see #allClose(Ring[], Ring[])
     */
    public static <T extends Ring<T>> boolean allClose(T[] src1, T[] src2, double relTol, double absTol) {
        if(src1.length!=src2.length) return false;
        
        for(int i=0; i<src1.length; i++) {
            double tol = absTol + relTol*src2[i].abs();
            if(src1[i].sub(src2[i]).abs() > tol) return false; // No need to continue.
        }

        return true; // If we reach this point, the arrays are close.
    }


    /**
     * Checks if two values are "close".
     * @param v1 First value in comparison.
     * @param v2 Second value in comparison.
     * @return True if the following is satisfied: {@code |v1 - v2| <= (1e-08+ 1e-05*|v2|)}. Otherwise, returns false.
     * @see #isClose(Ring, Ring)
     * @see #allClose(Ring[], Ring[], double, double)
     * @see #allClose(Ring[], Ring[])
     */
    public static <T extends Ring<T>> boolean isClose(T v1, T v2) {
        return isClose(v1, v2, 1e-05, 1e-08);
    }


    /**
     * Checks if two values are "close".
     * @param v1 First value in comparison.
     * @param v2 Second value in comparison.
     * @param relTol The relative tolerance to determine if values are close.
     * @param absTol The absolute tolerance to determine if values are close.
     * @return True if the following is satisfied: {@code |v1 - v2| <= (absTol + relTol*|v2|)}. Otherwise, returns false.
     * @see #isClose(Ring, Ring)
     * @see #allClose(Ring[], Ring[], double, double)
     * @see #allClose(Ring[], Ring[]) 
     */
    public static <T extends Ring<T>> boolean isClose(T v1, T v2, double relTol, double absTol) {
        double tol = absTol + relTol*v2.abs();
        return v1.sub(v2).abs() <= tol;
    }
}
