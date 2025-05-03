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

package org.flag4j.arrays.backend.field_arrays;

import org.flag4j.arrays.backend.ring_arrays.TensorOverRing;
import org.flag4j.numbers.Field;

/**
 * This interface specifies methods which any tensor whose data are elements of a field should implement.
 * This includes primitive values.
 *
 * <p>To allow for primitive types, the elements of this tensor do not necessarily have to implement
 * {@link Field}.
 *
 * <p>Formally, a field is a set <b>F</b> with the binary ops addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>F</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition and multiplication are commutative: a + b = b + a and a * b = b * a</li>
 *      <li>Existence of additive and multiplicative identities: There exists two distinct elements 0 and 1 in <b>F</b> such that a + 0 = 0
 *      and a * 1 = 1 (called the additive and multiplicative identities respectively).</li>
 *      <li>Existence of additive inverse: There exists an element -a in <b>F</b> such that a + (-a) = 0.</li>
 *      <li>Existence of multiplicative inverse: There exists an element a<sup>-1</sup> in <b>F</b> such that a * a<sup>-1</sup> = 1.</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. If {@code T} is dense, then this should be the same type as {@code T}.
 * This parameter required because some ops between two sparse tensors may result in a dense tensor.
 * @param <V> Storage for data of this tensor.
 * @param <W> Type (or wrapper) of an element of this tensor. Should satisfy the axioms of a field as stated.
 * @see TensorOverRing
 */
public interface TensorOverField<T extends TensorOverField<T, U, V, W>,
        U extends TensorOverField<U, U, V, W>, V, W> extends TensorOverRing<T, U, V, W> {

    /**
     * Adds a primitive scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero
     * data of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    T add(double b);


    /**
     * Adds a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    void addEq(double b);


    /**
     * Multiplies a primitive scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    T mult(double b);


    /**
     * Multiplies a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    void multEq(double b);


    /**
     * Subtracts a primitive scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    T sub(double b);


    /**
     * Subtracts a scalar primitive value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    void subEq(double b);


    /**
     * Divides each element of this tensor by a scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     * @see #divEq(Object)
     */
    T div(W b);


    /**
     * Divides each element of this tensor by a scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     * @see #div(Object)
     */
    void divEq(W b);


    /**
     * Divides each element of this tensor by a primitive scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     * @see #divEq(Object)
     */
    T div(double b);


    /**
     * Divides each element of this tensor by a primitive scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     * @see #div(Object)
     */
    void divEq(double b);


    /**
     * Computes the element-wise quotient between two tensors.
     * @param b Second tensor in the element-wise quotient.
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    T div(T b);


    /**
     * Computes the element-wise square root of this tensor.
     * @return The element-wise square root of this tensor.
     */
    T sqrt();


    /**
     * Computes the element-wise reciprocals of this tensor.
     * @return The element-wise reciprocals of this tensor.
     */
    T recip();


    /**
     * Checks if this tensor only contains finite values.
     * @return {@code true} if this tensor only contains finite values; {@code false} otherwise.
     * @see #containsInf()
     * @see #containsNaN()
     */
    boolean isAllFinite();


    /**
     * Checks if this tensor contains at least one infinite value.
     * @return {@code true} if this tensor contains at least one infinite value; {@code false} otherwise.
     * @see #isAllFinite()
     * @see #containsNaN()
     */
    boolean containsInf();


    /**
     * Checks if this tensor contains at least one NaN value.
     * @return {@code true} if this tensor contains at least one NaN value; {@code false} otherwise.
     * @see #isAllFinite()
     * @see #containsInf()
     */
    boolean containsNaN();
}
