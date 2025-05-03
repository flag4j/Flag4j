/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.linalg.ops.common.real;

import org.flag4j.util.ErrorMessages;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * This class provides low level methods for computing ops on real tensors. These methods can be applied to
 * either sparse or dense real tensors.
 */
public final class RealOps {

    private RealOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the scalar multiplication of a tensor.
     *
     * @param src Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @param dest Array to store result in. May be null.
     *
     * @return A reference to the {@code dest} array if it was not null. Otherwise, a new array will be formed.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not at least the size of {@code src}.
     */
    public static double[] scalMult(double[] src, double factor, double[] dest) {
        int size = src.length;
        if(dest==null) dest = new double[size];

        for(int i=0; i<size; i++)
            dest[i] = src[i]*factor;

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     *
     * @param src Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @param start Starting index of scalar multiplication.
     * @param stop Stopping index of scalar multiplication.
     * @param dest Array to store result in. May be null.
     *
     * @return A reference to the {@code dest} array if it was not null. Otherwise, a new array will be formed.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not the size of {@code src}.
     */
    public static double[] scalMult(double[] src, double factor, int start, int stop, double[] dest) {
        if(dest==null) dest = new double[src.length];

        for(int i=start; i<stop; i++)
            dest[i] = src[i]*factor;

        return dest;
    }


    /**
     * <p>Scales entries by the specified {@code factor} within {@code src} starting at index {@code start}
     * and scaling a total of {@code n} elements spaced by {@code stride}.
     *
     * <p>More formally, this method scales elements by the specified {@code factor} at indices:
     * {@code start}, {@code start + stride}, {@code start + 2*stride}, ..., {@code start + (n-1)*stride}.
     *
     * <p>This method may be used to scale a row or column of a
     * {@link org.flag4j.arrays.dense.Matrix matrix} {@code a} as follows:
     * <ul>
     *     <li>Maximum absolute value within row {@code i}:
     *     <pre>{@code scale(a.data, i*a.numCols, a.numCols, 1, dest);}</pre></li>
     *     <li>Maximum absolute value within column {@code j}:
     *     <pre>{@code scale(a.data, j, a.numRows, a.numRows, dest);}</pre></li>
     * </ul>
     *
     * @param src The array containing values to scale.
     * @param factor Factor by which to scale elements.
     * @param start The starting index in {@code src} to begin scaling.
     * @param n The number of elements to scale within {@code src1}.
     * @param stride The gap (in indices) between consecutive elements to scale within {@code src}.
     * @param dest The array to store the result in. May be {@code null} or the same array as {@code src} to perform the operation
     * in-place. Assumed to be at least as large as {@code src} but this is not explicitly enforced.
     *
     * @return If {@code dest == null} a new array containing all elements of {@code src} with the appropriate values scaled.
     * Otherwise, A reference to the {@code dest} array.
     */
    public static double[] scalMult(double[] src, double factor, int start, int n, int stride, double[] dest) {
        if(dest==null) dest = src.clone();
        int stop = start + n*stride;

        for(int i=start; i<stop; i+=stride)
            dest[i] = src[i]*factor;

        return dest;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param src Entries of the tensor.
     * @param divisor Scalar value to divide.
     * @param dest Array to store result in. May be {@code null}.
     *
     * @return A reference to the {@code dest} array if it was not {@code null}. Otherwise, a new array will be formed.
     */
    public static double[] scalDiv(double[] src, double divisor, double[] dest) {
        double[] quotient = new double[src.length];
        double scale = 1.0 / divisor;

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = src[i]*scale;

        return quotient;
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static double[] sqrt(double[] src) {
        double[] roots = new double[src.length];

        for(int i=0; i<roots.length; i++)
            roots[i] = Math.sqrt(src[i]);

        return roots;
    }


    /**
     * Computes the element-wise absolute value of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise absolute value of the tensor.
     */
    public static double[] abs(double[] src) {
        double[] abs = new double[src.length];

        for(int i=0; i<abs.length; i++)
            abs[i] = Math.abs(src[i]);

        return abs;
    }


    /**
     * Rounds the values of a tensor to the nearest integer. Also see {@link #round(double[], int)}.
     * @param src Entries of the tensor to round.
     * @return The result of rounding all data of the source tensor to the nearest integer.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static double[] round(double[] src) {
        double[] dest = new double[src.length];

        for(int i=0; i<dest.length; i++)
            dest[i] = Math.round(src[i]);

        return dest;
    }


    /**
     * Rounds the values of a tensor with specified precision. Note, if precision is zero, {@link #round(double[])} is
     * preferred.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e., the number of decimal places).
     * @return The result of rounding all data of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static double[] round(double[] src, int precision) {
        if(precision<0) {
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(precision, "precision"));
        }

        BigDecimal bd;
        double[] dest = new double[src.length];

        for(int i=0; i<dest.length; i++) {
            bd = new BigDecimal(Double.toString(src[i]));
            bd = bd.setScale(precision, RoundingMode.HALF_UP);
            dest[i] = bd.doubleValue();
        }

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
    public static double[] roundToZero(double[] src, double threshold) {
        if(threshold<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(threshold, "threshold"));

        double[] dest = new double[src.length];
        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = (Math.abs(src[i]) < threshold) ? 0 : src[i];

        return dest;
    }
}
