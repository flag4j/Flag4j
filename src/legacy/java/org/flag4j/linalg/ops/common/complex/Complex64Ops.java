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

package org.flag4j.linalg.ops.common.complex;


import org.flag4j.numbers.Complex64;
import org.flag4j.util.ErrorMessages;


/**
 * This class provides low level methods for computing ops on complex tensors. These methods can be applied to
 * either sparse or dense complex tensors.
 */
public final class Complex64Ops {

    private Complex64Ops() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Rounds the values of a tensor with specified precision.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e., the number of decimal places).
     * @return The result of rounding all items of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static Complex64[] round(Complex64[] src, int precision) {
        if(precision<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(precision, "precision"));

        Complex64[] dest = new Complex64[src.length];
        for(int i=0; i<dest.length; i++)
            dest[i] = Complex64.round(src[i], precision);

        return dest;
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this tensor is less than
     *                  the threshold in absolute value then it will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If {@code threshold} is negative.
     */
    public static Complex64[] roundToZero(Complex64[] src, float threshold) {
        if(threshold<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(threshold, "threshold"));

        Complex64[] dest = new Complex64[src.length];
        for(int i=0; i<dest.length; i++)
            dest[i] = Complex64.roundToZero(src[i], threshold);

        return dest;
    }


    /**
     * Checks whether a tensor contains only real values.
     * @param entries Entries of dense tensor or non-zero items of sparse tensor.
     * @return True if the tensor only contains real values. Returns false otherwise.
     */
    public static boolean isReal(Complex64[] entries) {
        if(entries == null) return false;

        for(Complex64 entry : entries)
            if(entry.im != 0) return false;

        return true;
    }


    /**
     * Checks whether a tensor contains at least one non-real value.
     * @param entries Entries of dense tensor or non-zero items of sparse tensor.
     * @return True if the tensor contains at least one non-real value. Returns false otherwise.
     */
    public static boolean isComplex(Complex64[] entries) {
        if(entries == null) return false;

        for(Complex64 entry : entries)
            if(entry.im != 0) return true;

        return false;
    }
}
