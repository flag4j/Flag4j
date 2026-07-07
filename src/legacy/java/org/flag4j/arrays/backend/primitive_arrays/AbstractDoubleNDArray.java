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

package org.flag4j.arrays.backend.primitive_arrays;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.field_arrays.TensorOverField;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.ops.common.real.AggregateReal;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.dense.real.RealDenseOps;
import org.flag4j.util.ArrayMapper;
import org.flag4j.util.ArrayReducer;
import org.flag4j.util.Flag4jConstants;

import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

/**
 * This is the base class of all real primitive double tensors, matrices, or vectors. The methods implemented in this class are
 * agnostic to whether the tensor is dense or sparse.
 */
public abstract class AbstractDoubleNDArray<T extends AbstractDoubleNDArray<T>>
        extends AbstractNDArray<T, double[], Double>
        implements TensorOverField<T, T, double[], Double> {

    // TODO: Adjust method JavaDocs to reflect that it may compute a solution only using non-zero
    //  values if the tensor is sparse.

    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     */
    protected AbstractDoubleNDArray(Shape shape, double[] entries) {
        super(shape, entries);
    }


    /**
     * Gets the size of the 1D items array backing this tensor.
     *
     * @return The size of the 1D items array backing this tensor.
     */
    @Override
    public int dataLength() {
        return data.length;
    }


    /**
     * Computes the sum of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the sum. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     */
    @Override
    public AbstractDoubleNDArray<?> sum(int... axes) {
        return reduce(null, (double x, double y)->x+y, axes);
    }


    /**
     * <p>Computes the product of all values in this tensor along the specified {@code axes}.
     * <p>For sparse tensors, this method will only consider the non-zero values.
     * @param axes Axes along which to compute the product. All axes must be in the range {@code [0, this.rank() - 1]}.
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     * @see #prod()
     */
    @Override
    public AbstractDoubleNDArray<?> prod(int... axes) {
        return reduce(null, (double x, double y)->x*y, axes);
    }


    /**
     * Applies a map to each item in this nD array.
     * This operation is done in-place.
     *
     * If this nD array is sparse, the {@code mapper} operation will only be applied to the non-zero elements in this nD array.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A reference to this nD array.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     * @see #map(DoubleUnaryOperator) 
     */
    @Override
    public T map(UnaryOperator<Double> mapper) {
        map((double x)->mapper.apply(x));
        return (T) this;
    }


    /**
     * Applies a map to each item in this nD array.
     * This operation is done in-place.
     *
     * If this nD array is sparse, the {@code mapper} operation will only be applied to the non-zero elements in this nD array.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A reference to this nD array.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     * @see #map(UnaryOperator)
     */
    public T map(DoubleUnaryOperator mapper) {
        ArrayMapper.map(data, mapper);
        return (T) this;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the
     * starting value of the
     * reduction.
     * @param accumulator A binary operator that combines the current accumulated
     * result with the next array element and returns the updated result.
     *
     * @return The final accumulated scalar. If this nD array is empty, {@code identity} will be returned.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(Double, DoubleBinaryOperator)
     */
    @Override
    public Double reduce(Double identity, BinaryOperator<Double> accumulator) {
        // Wrap the accumulator as a DoubleBinaryOperator.
        return reduce(identity, (double x, double y)->accumulator.apply(x, y));
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the
     * starting value of the
     * reduction.
     * @param accumulator A binary operator that combines the current accumulated
     * result with the next array element and returns the updated result.
     *
     * @return The final accumulated scalar. If this nD array is empty, {@code identity} will be returned.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     *
     * @see #reduce(Double, BinaryOperator)
     */
    public Double reduce(Double identity, DoubleBinaryOperator accumulator) {
        return ArrayReducer.reduce(data, identity, accumulator);
    }


    /**
     * Reduces elements of this array, along a specified set of axes, by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to elements of this nD array along the specified axes in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the starting value of the
     * reduction.
     * @param accumulator The binary operator used to accumulate elements of this nD array.
     * For the results to be well-defined, the accumulator must be associative and communitive.
     * @param axes The axes along which reduce this nD array.
     *
     * @return An nD array of the same shape as this nD array, but with the specified {@code axes} removed, containing the result of
     * the reduction operation.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(Double, BinaryOperator)
     */
    public AbstractDoubleNDArray<?> reduce(Double identity, BinaryOperator<Double> accumulator, int... axes) {
        // Wrap the accumulator as a DoubleBinaryOperator.
        return reduce(identity, (double x, double y)->accumulator.apply(x, y), axes);
    }


    /**
     * Reduces elements of this array, along a specified set of axes, by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to elements of this nD array along the specified axes in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the starting value of the
     * reduction.
     * @param accumulator The binary operator used to accumulate elements of this nD array.
     * For the results to be well-defined, the accumulator must be associative and communitive.
     * @param axes The axes along which reduce this nD array.
     *
     * @return An nD array of the same shape as this nD array, but with the specified {@code axes} removed, containing the result of
     * the reduction operation.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(Double, BinaryOperator)
     */
    public abstract AbstractDoubleNDArray<?> reduce(Double identity, DoubleBinaryOperator accumulator, int... axes);


    /**
     * Checks if <em>any</em> element in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return {@code true} if <em>any</em> element in this nD array satisfies the {@code predicate}; otherwise {@code false}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #all(Function)
     */
    @Override
    public boolean any(Function<Double, Boolean> predicate) {
        for(double v : data)
            if (predicate.apply(v)) return true;

        return false;
    }


    /**
     * Checks if <em>all</em> elements in this nD array satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return {@code true} if <em>all</em> elements in this nD array satisfy the {@code predicate}; otherwise {@code false}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #any(Function)
     */
    @Override
    public boolean all(Function<Double, Boolean> predicate) {
        for(double v : data)
            if (!predicate.apply(v)) return false;

        return true;
    }


    /**
     * Extracts elements of this nD array that satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return A flat 1D array containing the elements of this nD array that satisfy the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #where(Function)
     */
    @Override
    public Vector filter(Function<Double, Boolean> predicate) {
        List<Double> filtered = new ArrayList<>(data.length / 4);

        for(double v : data)
            if(predicate.apply(v)) filtered.add(v);

        return new Vector(filtered);
    }


    /**
     * Rounds each entry of this tensor to the nearest whole number.
     *
     * @return A copy of this tensor with each entry rounded to the nearest whole number.
     * @see #round(int)
     * @see #roundToZero()
     * @see #roundToZero(double)
     */
    public T round() {
        return makeLikeNDArray(this.shape, RealOps.round(this.data));
    }


    /**
     * Rounds each entry in this tensor to the nearest whole number.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If {@code precision} is negative.
     * @see #round()
     * @see #roundToZero()
     * @see #roundToZero(double)
     */
    public T round(int precision) {
        return makeLikeNDArray(this.shape, RealOps.round(this.data, precision));
    }


    /**
     * Rounds values in this tensor which are close to zero in absolute value to zero.
     * If the matrix is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within {@link Flag4jConstants#EPS_F64} of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this matrix with rounded values.
     * @see #roundToZero(double)
     * @see #round()
     * @see #round(int)
     */
    public T roundToZero() {
        return makeLikeNDArray(this.shape, RealOps.roundToZero(this.data, Flag4jConstants.EPS_F64));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     * If the matrix is complex, both the real and imaginary components will be rounded independently.
     * @param threshold Threshold for rounding values to zero.
     * That is, if a value in this matrix is less than the threshold in absolute value, then it will be rounded to zero.
     * This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If the threshold is negative.
     * @see #roundToZero()
     * @see #round()
     * @see #round(int)
     */
    public T roundToZero(double threshold) {
        return makeLikeNDArray(this.shape, RealOps.roundToZero(this.data, threshold));
    }


    /**
     * Checks if this tensor only contains positive values.
     * @return Returns {@code true} if this tensor only contains positive values; {@code false} otherwise.
     */
    public boolean isAllPos() {
        return RealProperties.isAllPos(data);
    }


    /**
     * Checks if this tensor only contains negative values.
     * @return Returns {@code true} if this tensor only contains negative values; {@code false} otherwise.
     */
    public boolean isAllNeg() {
        return RealProperties.isAllNeg(data);
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeNDArray(shape, data.clone());
    }


    /**
     * Subtracts a scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(Double b) {
        return sub((double) b);
    }


    /**
     * Subtracts a scalar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Double b) {
        subEq((double) b);
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public T abs() {
        return makeLikeNDArray(shape, RealOps.abs(data));
    }


    /**
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    public T conj() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        return T(axis1, axis2);
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
        return H(axes);
    }


    /**
     * Finds the minimum value in this tensor.
     *
     * @return The minimum value in this tensor.
     */
    @Override
    public Double min() {
        return RealProperties.min(data);
    }


    /**
     * Finds the maximum value in this tensor.
     *
     * @return The maximum value in this tensor.
     */
    @Override
    public Double max() {
        return RealProperties.max(data);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return RealProperties.minAbs(data);
    }


    /**
     * Finds the maximum absolute value in this tensor.
     *
     * @return The maximum absolute value in this tensor.
     */
    @Override
    public double maxAbs() {
        return RealProperties.maxAbs(data);
    }


    /**
     * Adds a scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the non-zero
     * items of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(Double b) {
        return add((double) b);
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        addEq((double) b);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public T mult(Double b) {
        return mult((double) b);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(Double b) {
        multEq((double) b);
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return {@code true} if this tensor only contains zeros; {@code false} otherwise.
     */
    @Override
    public boolean isAllZeros() {
        return RealProperties.isAllZeros(data);
    }


    /**
     * Checks if this tensor only contains ones. If this tensor is sparse, only the non-zero items are considered.
     *
     * @return {@code true} if this tensor only contains ones; {@code false} otherwise.
     */
    @Override
    public boolean isAllOnes() {
        return RealProperties.isAllOnes(data);
    }


    /**
     * Computes the sum of all values in this tensor.
     *
     * @return The sum of all values in this tensor.
     */
    @Override
    public Double sum() {
        return AggregateReal.sum(data);
    }


    /**
     * Computes the product of all values in this tensor.
     *
     * @return The product of all values in this tensor.
     */
    @Override
    public Double prod() {
        return AggregateReal.prod(data);
    }


    /**
     * Adds a primitive scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero items of the tensor.
     *
     * @param b Scalar value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(double b) {
        return makeLikeNDArray(shape, RealDenseOps.add(data, b, null));
    }


    /**
     * Adds a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(double b) {
        RealDenseOps.add(data, b, data);
    }


    /**
     * Multiplies a primitive scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public T mult(double b) {
        return makeLikeNDArray(shape, RealOps.scalMult(data, b, null));
    }


    /**
     * Multiplies a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(double b) {
        RealOps.scalMult(data, b, data);
    }


    /**
     * Subtracts a primitive scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(double b) {
        return makeLikeNDArray(shape, RealDenseOps.sub(data, b, null));
    }


    /**
     * Subtracts a scalar primitive value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(double b) {
        RealDenseOps.sub(data, b, data);
    }


    /**
     * Divides each element of this tensor by a scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(double)
     */
    @Override
    public T div(Double b) {
        return div((double) b);
    }


    /**
     * Divides each element of this tensor by a scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #div(double)
     */
    @Override
    public void divEq(Double b) {
        divEq((double) b);
    }


    /**
     * Divides each element of this tensor by a primitive scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(double)
     */
    @Override
    public T div(double b) {
        return makeLikeNDArray(shape, RealOps.scalDiv(data, b, null));
    }


    /**
     * Divides each element of this tensor by a primitive scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #div(double)
     */
    @Override
    public void divEq(double b) {
        RealOps.scalDiv(data, b, data);
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        return makeLikeNDArray(shape, RealOps.sqrt(data));
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return The element-wise reciprocals of this tensor.
     */
    @Override
    public T recip() {
        return makeLikeNDArray(shape, RealDenseOps.recip(data));
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
        return RealProperties.isAllFinite(data);
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
        return RealProperties.containsInf(data);
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
        return RealProperties.containsNaN(data);
    }


    /**
     * Counts the number of elements in this nD array which satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return The number of elements in this nD array which satisfy the specified {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     */
    @Override
    public int countTrue(Function<Double, Boolean> predicate) {
        int count = 0;

        for (Double v : data)
            if (predicate.apply(v)) count++;

        return count;
    }
}
