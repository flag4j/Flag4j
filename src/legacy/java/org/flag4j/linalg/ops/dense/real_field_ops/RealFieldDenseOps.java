/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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


import org.flag4j.arrays.Shape;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * This class provides low level methods for computing ops with at least one real tensor
 * and at least one field tensor.
 */
public final class RealFieldDenseOps {

    private RealFieldDenseOps() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Computes the element-wise addition of two tensors.
     *
     * @param Array to store the result of the element-wise sum in. Must be at least as long as {@code src1}.
     * May be the same array as {@code src1}.
     * @param shape1 Shape of first tensor.
     * @param src1 Entries of first tensor.
     * @param shape2 Shape of second tensor.
     * @param src2 Entries of second tensor.
     *
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     */
    public static <T extends FieldElement<T>> void add(
            Shape shape1, T[] src1, Shape shape2, double[] src2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].add(src2[i]);
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     *
     * @param shape1 Shape of first tensor.
     * @param src1 Entries of first tensor.
     * @param shape2 Shape of second tensor.
     * @param src2 Entries of second tensor.
     * @param dest Array to store the resulting element-wise difference in. Must be at lease as large as {@code src1}.
     * May be the same array as {@code src1}.
     *
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException                  If {@code src1.length != src2.length}
     */
    public static <T extends FieldElement<T>> void sub(
            Shape shape1, T[] src1, Shape shape2, double[] src2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].sub(src2[i]);
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     *
     * @param shape1 Shape of first tensor.
     * @param src1 Entries of first tensor.
     * @param shape2 Shape of second tensor.
     * @param src2 Entries of second tensor.
     * @param dest Array to store the resulting element-wise difference in. Must be at lease as large as {@code src1}.
     * May be the same array as {@code src1}.
     *
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     */
    public static <T extends FieldElement<T>> void sub(
            Shape shape1, double[] src1, Shape shape2, T[] src2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src2[i].negate().add(src1[i]);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param src Entries of the tensor.
     * @param divisor Scalar value to multiply.
     * @param dest Array to store the result of the scalar division in.
     * Must be at least as large as {@code src}. May be the same array as {@code dest}.
     */
    public static <T extends FieldElement<T>> void scalDiv(double[] src, T divisor, T[] dest) {
        T multInv = divisor.invert();

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = multInv.mult(src[i]);
    }



    /**
     * Adds a scalar value to all items of a tensor.
     * @param src Entries of first tensor.
     * @param a Scalar to add to all items of this tensor.
     * @param dest Array to store the result of adding the scalar {@code a} to {@code src}.
     * Must be at least as large as {@code src}. May be the same array as {@code dest}.
     */
    public static <T extends FieldElement<T>> void add(T[] src, double a, T[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].add(a);
    }


    /**
     * Adds a scalar value to all items of a tensor.
     * @param src Entries of first tensor.
     * @param a Scalar to add to all items of this tensor.
     * @param dest Array to store the result of adding the scalar {@code a} to {@code src}.
     * Must be at least as large as {@code src}. May be the same array as {@code dest}.
     */
    public static <T extends FieldElement<T>> void add(double[] src, T a, T[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = a.add(src[i]);
    }


    /**
     * Subtracts a scalar value from all items of a tensor.
     * @param src Entries of first tensor.
     * @param a Scalar to add to all items of this tensor.
     * @param dest Array to store the result of subtracting the scalar {@code a} from {@code src}.
     * Must be at least as large as {@code src}. May be the same array as {@code dest}.
     */
    public static <T extends FieldElement<T>> void sub(T[] src, double a, T[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].sub(a);
    }


    /**
     * Subtracts a scalar value from all items of a tensor.
     * @param src Entries of first tensor.
     * @param a Scalar to add to all items of this tensor.
     * @param dest Array to store the result of subtracting the scalar {@code a} from all items of {@code src}.
     * Must be at least as large as {@code src}. May be the same array as {@code src}.
     */
    public static <T extends FieldElement<T>> void sub(double[] src, T a, T[] dest) {
        T aInv = a.negate();

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = aInv.add(src[i]);
    }


    /**
     * Computes the scalar division of a tensor.
     * @param src Entries of the tensor.
     * @param divisor Scalar value to divide tensor by.
     * @param dest Array to store the result of the scalar division in. Must be at least as large as {@code src}.
     * May be the same array as {@code src}.
     */
    public static <T extends FieldElement<T>> void scalDiv(T[] src, double divisor, T[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].div(divisor);
    }


    /**
     * Computes the element-wise multiplication between two dense tensors.
     * @param shape1 Shape of the first tensor in the element-wise product.
     * @param src1 Entries of the first tensor in the element-wise product.
     * @param shape2 Shape of the second tensor in the element-wise product.
     * @param src2 Entries of the second tensor in the element-wise product.
     * @param dest Array to store the result of the element-wise product in. Must be at least as large as {@code src1}.
     * May be the same ndarrays as {@code src1}.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape1)}.
     */
    public static <T extends FieldElement<T>> void elemMult(
            Shape shape1, T[] src1, Shape shape2, double[] src2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].mult(src2[i]);
    }


    /**
     * Computes the element-wise division between two dense tensors.
     * @param shape1 Shape of the first tensor in the element-wise quotient.
     * @param src1 Entries of the first tensor in the element-wise quotient.
     * @param shape2 Shape of the second tensor in the element-wise quotient.
     * @param src2 Entries of the second tensor in the element-wise quotient.
     * @param dest Array to store the result of the element-wise quotient in.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape1)}.
     */
    public static <T extends FieldElement<T>> void elemDiv(
            Shape shape1, T[] src1, Shape shape2, double[] src2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].div(src2[i]);
    }


    /**
     * Computes the element-wise division between two dense tensors.
     * @param shape1 Shape of the first tensor in the element-wise quotient.
     * @param src1 Entries of the first tensor in the element-wise quotient.
     * @param shape2 Shape of the second tensor in the element-wise quotient.
     * @param src2 Entries of the second tensor in the element-wise quotient.
     * @param dest Array to store the result of the element-wise quotient in.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape1)}.
     */
    public static <T extends FieldElement<T>> void elemDiv(
            Shape shape1, double[] src1, Shape shape2, T[] src2, T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src2[i].invert().mult(src1[i]);
    }
}
