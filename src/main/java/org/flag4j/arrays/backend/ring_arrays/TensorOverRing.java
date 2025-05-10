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

package org.flag4j.arrays.backend.ring_arrays;


import org.flag4j.arrays.backend.semiring_arrays.TensorOverSemiring;
import org.flag4j.numbers.Ring;
import org.flag4j.util.exceptions.ArrayShapeException;


/**
 * This interface specifies methods which any tensor whose data are elements of a ring should implement. This includes
 * primitive values.
 *
 * <p>To allow for primitive types, the elements of this tensor do not necessarily have to implement
 * {@link Ring}.
 *
 * <p>Formally, a ring is a set <b>R</b> with the binary ops addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>R</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition is commutative: a + b = b + a</li>
 *      <li>Existence of additive and multiplicative identities: There exists two distinct elements 0 and 1 in <b>R</b> such that a + 0 = 0
 *      and a * 1 = 1 (called the additive and multiplicative identities respectively).</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * 
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. If {@code T} is dense, then this should be the same type as {@code T}.
 * This parameter required because some ops between two sparse tensors may result in a dense tensor.
 * @param <V> Storage for data of this tensor.
 * @param <W> Type (or wrapper) of an element of this tensor. Should satisfy the axioms of a semiring as stated.
 *
 * @see TensorOverSemiring
 */
public interface TensorOverRing<T extends TensorOverRing<T, U, V, W>,
        U extends TensorOverRing<U, U, V, W>, V, W> extends TensorOverSemiring<T, U, V, W> {

    /**
     * Subtracts a scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    T sub(W b);


    /**
     * Subtracts a scalar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    void subEq(W b);


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    T sub(T b);


    /**
     * Computes the element-wise absolute value of this tensor.
     * @return The element-wise absolute value of this tensor.
     */
    TensorOverRing abs();


    /**
     * Computes the element-wise conjugation of this tensor.
     * @return The element-wise conjugation of this tensor.
     */
    T conj();


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     * @return The conjugate transpose of this tensor.
     * @see #H(int, int)
     * @see #H(int...)
     */
    default T H() {
        return H(0, getRank()-1);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     * @return The conjugate transpose of this tensor along the specified axes.
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    T H(int axis1, int axis2);


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    T H(int... axes);

    /**
     * Finds the minimum value in this tensor.
     * @return The minimum value in this tensor.
     */
    W min();


    /**
     * Finds the maximum value in this tensor.
     * @return The maximum value in this tensor.
     */
    W max();


    /**
     * Finds the indices of the minimum value in this tensor.
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argmin();


    /**
     * Finds the indices of the maximum value in this tensor.
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argmax();


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argminAbs();


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argmaxAbs();


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     * @return The minimum value, in absolute value, in this tensor.
     */
    double minAbs();


    /**
     * Finds the maximum absolute value in this tensor.
     * @return The maximum absolute value in this tensor.
     */
    double maxAbs();
}
