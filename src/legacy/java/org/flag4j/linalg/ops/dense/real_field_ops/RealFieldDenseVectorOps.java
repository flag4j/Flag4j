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

package org.flag4j.linalg.ops.dense.real_field_ops;


import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;

/**
 * This class provides low-level implementations for vector ops with one dense real vector and one dense field vector.
 */
public final class RealFieldDenseVectorOps {

    private RealFieldDenseVectorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the vector inner product for a real dense vector and a complex dense vector.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The inner product of the two vectors.
     */
    public static <T extends FieldElement<T>> T inner(double[] src1, T[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        T innerProd = (src2.length > 0) ? src2[0].getZero() : null;

        for(int i=0; i<src1.length; i++)
            innerProd = innerProd.add(src2[i].conj().mult(src1[i]));

        return innerProd;
    }



    /**
     * Computes the vector inner product for a complex dense vector and a real dense vector.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The inner product of the two vectors.
     */
    public static <T extends FieldElement<T>> T inner(T[] src1, double[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        T innerProd = (src1.length > 0) ? src1[0].getZero() : null;

        for(int i=0; i<src1.length; i++)
            innerProd = innerProd.add(src1[i].mult(src2[i]));

        return innerProd;
    }



    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @param dest Array to store the result of the vector outer product in. Must have length {@code src1.length*src2.length}.
     */
    public static <T extends FieldElement<T>> void outerProduct(double[] src1, T[] src2, T[] dest) {
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = i*src2.length;
            double v = src1[i];

            for(T cNumber : src2)
                dest[destIndex++] = cNumber.conj().mult(v);
        }
    }


    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @param dest Array to store the result of the vector outer product in. Must have length {@code src1.length*src2.length}.
     */
    public static <T extends FieldElement<T>> void outerProduct(T[] src1, double[] src2, T[] dest) {
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = i*src2.length;
            T v1 = src1[i];

            for(double v2 : src2)
                dest[destIndex++] = v1.mult(v2);
        }
    }

    /**
     * Adds a scalar value to all items of a tensor.
     * @param src Entries of first tensor.
     * @param a Scalar to add to all items of this tensor.
     * @param dest Array to store the result of the vector-scalar addition. Must have length equal to {@code src.length}.
     */
    public static <T extends FieldElement<T>> void add(double[] src, T a, T[] dest) {
        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = a.add(src[i]);
    }
}
