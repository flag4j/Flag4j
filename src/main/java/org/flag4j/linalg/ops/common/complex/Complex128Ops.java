/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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


import org.flag4j.numbers.Complex128;
import org.flag4j.util.ErrorMessages;

/**
 * This class provides low level methods for computing operations on complex tensors. These methods can be applied to
 * either sparse or dense complex tensors.
 */
public final class Complex128Ops {

    private Complex128Ops() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Computes the element-wise square root of a tensor as complex values. This allows for the square root of negative numbers.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static Complex128[] sqrt(double[] src) {
        Complex128[] roots = new Complex128[src.length];
        for(int i=0; i<roots.length; i++)
            roots[i] = Complex128.sqrt(src[i]);

        return roots;
    }


    /**
     * Rounds the values of a tensor with specified precision.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e., the number of decimal places).
     * @return The result of rounding all data of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static Complex128[] round(Complex128[] src, int precision) {
        if(precision<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(precision, "precision"));

        Complex128[] dest = new Complex128[src.length];
        for(int i=0; i<dest.length; i++)
            dest[i] = Complex128.round(src[i], precision);

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
    public static Complex128[] roundToZero(Complex128[] src, double threshold) {
        if(threshold<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(threshold, "threshold"));

        Complex128[] dest = new Complex128[src.length];
        for(int i=0; i<dest.length; i++)
            dest[i] = Complex128.roundToZero(src[i], threshold);

        return dest;
    }


    /**
     * Converts a complex tensor to a real tensor by copying the real component and discarding the imaginary component.
     * @param src Entries of complex tensor.
     * @return Equivalent real data for complex tensor.
     */
    public static double[] toReal(Complex128[] src) {
        double[] real = new double[src.length];

        for(int i=0; i<src.length; i++)
            real[i] = src[i].re;

        return real;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide each element ot the tensor by.
     * @return The scalar division of the tensor.
     */
    public static Complex128[] scalDiv(double[] entries, Complex128 divisor) {
        Complex128[] quotient = new Complex128[entries.length];

        double denom = divisor.re*divisor.re + divisor.im*divisor.im;

        for(int i=0, size=entries.length; i<size; i++) {
            double a = entries[i];
            quotient[i] = new Complex128(a*divisor.re / denom, -a*divisor.im / denom);
        }

        return quotient;
    }


    /**
     * Checks whether a tensor contains only real values.
     * @param entries Entries of dense tensor or non-zero data of sparse tensor.
     * @return True if the tensor only contains real values. Returns false otherwise.
     */
    public static boolean isReal(Complex128[] entries) {
        if(entries == null) return false;

        for(Complex128 entry : entries)
            if(entry.im != 0) return false;

        return true;
    }


    /**
     * Checks whether a tensor contains at least one non-real value.
     * @param entries Entries of dense tensor or non-zero data of sparse tensor.
     * @return True if the tensor contains at least one non-real value. Returns false otherwise.
     */
    public static boolean isComplex(Complex128[] entries) {
        if(entries == null) return false;

        for(Complex128 entry : entries)
            if(entry.im != 0) return true;

        return false;
    }
}
