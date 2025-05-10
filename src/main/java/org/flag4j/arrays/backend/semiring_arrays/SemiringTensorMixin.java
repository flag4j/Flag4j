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

package org.flag4j.arrays.backend.semiring_arrays;


import org.flag4j.linalg.ops.common.semiring_ops.AggregateSemiring;
import org.flag4j.linalg.ops.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringOps;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringProperties;
import org.flag4j.numbers.Semiring;


/**
 * This interface provides default functionality for all tensors whose data are elements of a {@link Semiring}. This includes both
 * sparse and dense tensors.
 * @param <T> Type of this tensor.
 * @param <U> Dense equivalent of this tensor. If this tensor is dense, this should be the same type as {@code T}
 * @param <V> Type of an element of this tensor. Satisfies {@link Semiring semiring} axioms.
 */
public interface SemiringTensorMixin<T extends SemiringTensorMixin<T, U, V>,
        U extends SemiringTensorMixin<U, U, V>, V extends Semiring<V>>
        extends TensorOverSemiring<T, U, V[], V> {


    /**
     * Creates an empty array of the same type as the data array of this tensor.
     * @param length The length of the array to construct.
     * @return An empty array of the same type as the data array of this tensor.
     */
    default V[] makeEmptyDataArray(int length) {
        return (V[]) new Semiring[length];
    }


    /**
     * Adds a scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the non-zero
     * data of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    default T add(V b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        SemiringOps.add(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }

    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    default void addEq(V b) {
        V[] entries = getData();
        SemiringOps.add(entries, b, entries);
    }

    /**
     * Multiplies a scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    default T mult(V b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        SemiringOps.scalMult(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }

    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    default void multEq(V b) {
        V[] entries = getData();
        SemiringOps.scalMult(entries, b, entries);
    }

    /**
     * Checks if this tensor only contains zeros.
     *
     * @return {@code true} if this tensor only contains zeros; {@code false} otherwise.
     */
    @Override
    default boolean isAllZeros() {
        return SemiringProperties.isZeros(getData());
    }

    /**
     * Checks if this tensor only contains ones. If this tensor is sparse, only the non-zero data are considered.
     *
     * @return {@code true} if this tensor only contains ones; {@code false} otherwise.
     */
    @Override
    default boolean isAllOnes() {
        return SemiringProperties.isOnes(getData());
    }

    /**
     * Computes the sum of all values in this tensor.
     *
     * @return The sum of all values in this tensor.
     */
    @Override
    default V sum() {
        return AggregateSemiring.sum(getData());
    }

    /**
     * Computes the product of all values in this tensor (or non-zero values if this tensor is sparse).
     *
     * @return The product of all values (or non-zero values if sparse) in this tensor.
     */
    @Override
    default V prod() {
        return AggregateSemiring.prod(getData());
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex-valued tensor) in this tensor.
     */
    default V min() {
        return CompareSemiring.min(getData());
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex-valued tensor) in this tensor.
     */
    default V max() {
        return CompareSemiring.max(getData());
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    default int[] argmin() {
        return getShape().getNdIndices(CompareSemiring.argmin(getData()));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    default int[] argmax() {
        return getShape().getNdIndices(CompareSemiring.argmax(getData()));
    }
}
