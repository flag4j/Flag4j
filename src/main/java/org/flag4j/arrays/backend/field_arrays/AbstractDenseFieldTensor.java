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

package org.flag4j.arrays.backend.field_arrays;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingTensor;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldElemDiv;
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;

/**
 * <p>The base class for all dense {@link Field} tensors.
 * <p>The {@link #data} of an AbstractDenseFieldTensor are mutable but the {@link #shape} is fixed.
 *
 * @param <T> The type of this dense field tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}. This type parameter is required because some ops (e.g.
 * {@link #toCoo()}) may result in a sparse tensor.
 * @param <V> The type of the {@link Field} which this tensor's data belong to.
 */
public abstract class AbstractDenseFieldTensor<T extends AbstractDenseFieldTensor<T, V>, V extends Field<V>>
        extends AbstractDenseRingTensor<T, V>
        implements FieldTensorMixin<T, T, V> {

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseFieldTensor(Shape shape, V[] entries) {
        super(shape, entries);
        ValidateParameters.ensureAllEqual(shape.numelIntValueExact(), entries.length);
        this.zeroElement = (entries.length > 0 && entries[0] != null) ? entries[0].getZero() : null;
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public T H(int... axes) {
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensorHermitian(shape, data, axes, dest);
        return makeLikeNDArray(shape.permuteAxes(axes), dest);
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    @Override
    public T div(T b) {
        V[] dest = makeEmptyDataArray(data.length);
        DenseFieldElemDiv.dispatch(data, shape, b.data, b.shape, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.sqrt(data, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values; {@code false} otherwise.
     *
     * @see #containsInf()
     * @see #containsNaN()
     */
    @Override
    public boolean isAllFinite() {
        return FieldOps.isFinite(data);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value; {@code false} otherwise.
     *
     * @see #isAllFinite()
     * @see #containsNaN()
     */
    @Override
    public boolean containsInf() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value; {@code false} otherwise.
     *
     * @see #isAllFinite()
     * @see #containsInf()
     */
    @Override
    public boolean containsNaN() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if all data of this matrix are "close" as defined below. Custom tolerances may be specified using
     * {@link #allClose(AbstractDenseFieldTensor, double, double)}.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same shape and all data are "close" element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (1E-08 + 1E-05*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldTensor, double, double)
     */
    public boolean allClose(T b) {
        return hasSameShape(b) && RingProperties.allClose(data, b.data);
    }


    /**
     * Checks if all data of this matrix are "close" as defined below.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same length and all data are "close" element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (absTol + relTol*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldTensor)
     */
    public boolean allClose(T b, double relTol, double absTol) {
        return hasSameShape(b) && RingProperties.allClose(data, b.data, relTol, absTol);
    }
}
