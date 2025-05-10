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

import org.flag4j.arrays.backend.ring_arrays.RingTensorMixin;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.common.semiring_ops.AggregateSemiring;
import org.flag4j.linalg.ops.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringOps;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringProperties;
import org.flag4j.numbers.Field;

/**
 * <p>This interface provides default functionality for all tensors whose data are elements of a
 * {@link Field}. This includes both sparse and dense tensors.
 *
 * <p>The default methods in this interface can be overridden if desired, but it is generally recommended to use them as is.
 * @param <T> Type of this tensor.
 * @param <U> Dense equivalent of this tensor. If this tensor is dense, this should be the same type as {@code T}
 * @param <V> Type of an element of this tensor. Satisfies {@link Field field} axioms.
 */
public interface FieldTensorMixin<T extends FieldTensorMixin<T, U, V>,
        U extends FieldTensorMixin<U, U, V>, V extends Field<V>>
        extends TensorOverField<T, U, V[], V>, RingTensorMixin<T, U, V> {

    /**
     * Creates an empty array of the same type as the data array of this tensor.
     * @param length The length of the array to construct.
     * @return An empty array of the same type as the data array of this tensor.
     */
    default V[] makeEmptyDataArray(int length) {
        return (V[]) new Field[length];
    }


    /**
     * Subtracts a scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    default T sub(V b) {
        V[] data = getData();
        V[] diff = makeEmptyDataArray(data.length);
        RingOps.sub(data, b, diff);
        return makeLikeNDArray(getShape(), diff);
    }


    /**
     * Subtracts a scalar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    default void subEq(V b) {
        V[] data = getData();
        RingOps.sub(data, b, data);
    }


    /**
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    default T conj() {
        V[] data = getData();
        V[] conj = makeEmptyDataArray(data.length);
        RingOps.conj(data, conj);
        return makeLikeNDArray(getShape(), conj);
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     * @return The element-wise reciprocals of this tensor.
     */
    @Override
    default T recip() {
        V[] data = getData();
        V[] recip = makeEmptyDataArray(data.length);
        FieldOps.recip(data, recip);
        return makeLikeNDArray(getShape(), recip);
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


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    default int[] argminAbs() {
        return getShape().getNdIndices(CompareRing.argminAbs(getData()));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    default int[] argmaxAbs() {
        return getShape().getNdIndices(CompareRing.argmaxAbs(getData()));
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    default double minAbs() {
        return CompareRing.minAbs(getData());
    }


    /**
     * Finds the maximum absolute value in this tensor.
     *
     * @return The maximum absolute value in this tensor.
     */
    @Override
    default double maxAbs() {
        return CompareRing.maxAbs(getData());
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
        SemiringOps.add(getData(), b, dest);
        return makeLikeNDArray(getShape(), dest);
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    default void addEq(V b) {
        V[] data = getData();
        SemiringOps.add(data, b, data);
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
        SemiringOps.scalMult(getData(), b, dest);
        return makeLikeNDArray(getShape(), dest);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    default void multEq(V b) {
        V[] data = getData();
        SemiringOps.scalMult(data, b, data);
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
     * Adds a primitive scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero
     * data of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    default T add(double b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.add(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }

    /**
     * Adds a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    default void addEq(double b) {
        V[] data = getData();
        FieldOps.add(data, b, data);
    }

    /**
     * Multiplies a primitive scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    default T mult(double b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.mult(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }

    /**
     * Multiplies a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    default void multEq(double b) {
        V[] data = getData();
        FieldOps.mult(data, b, data);
    }

    /**
     * Subtracts a primitive scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    default T sub(double b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.sub(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }

    /**
     * Subtracts a scalar primitive value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    default void subEq(double b) {
        V[] data = getData();
        FieldOps.sub(data, b, data);
    }

    /**
     * Divides each element of this tensor by a scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(Field)
     * @see #div(double)
     * @see #divEq(double)
     */
    @Override
    default T div(V b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.div(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }

    /**
     * Divides each element of this tensor by a scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #divEq(double)
     * @see #div(Field)
     * @see #divEq(double)
     */
    @Override
    default void divEq(V b) {
        V[] data = getData();
        FieldOps.div(data, b, data);
    }

    /**
     * Divides each element of this tensor by a primitive scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(Field)
     * @see #div(Field)
     * @see #divEq(double)
     */
    @Override
    default T div(double b) {
        V[] data = getData();
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.div(data, b, dest);
        return makeLikeNDArray(getShape(), dest);
    }


    /**
     * Divides each element of this tensor by a primitive scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #div(Field)
     * @see #divEq(Field)
     * @see #div(double)
     */
    @Override
    default void divEq(double b) {
        V[] data = getData();
        FieldOps.div(data, b, data);
    }


    /**
     * Computes the Euclidean norm of this vector.
     *
     * @return The Euclidean norm of this vector.
     */
    default double norm() {
        return VectorNorms.norm(getData());
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    default double norm(double p) {
        return VectorNorms.norm(getData(), p);
    }
}
