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


import org.flag4j.arrays.ArrayMask;
import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.dense.real.RealDenseElemDiv;
import org.flag4j.linalg.ops.dense.real.RealDenseElemMult;
import org.flag4j.linalg.ops.dense.real.RealDenseOps;
import org.flag4j.linalg.ops.dense.real.RealDenseTensorDot;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringOps;
import org.flag4j.util.ArrayReducer;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;

import java.util.BitSet;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Function;

/**
 * This is the base class of all real primitive double tensors.
 * The methods implemented in this class are agnostic to whether the tensor is dense or sparse.
 * @param <T> The type of the tensor.
 */
public abstract class AbstractDenseDoubleNDArray<T extends AbstractDoubleNDArray<T>>
        extends AbstractDoubleNDArray<T> {

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseDoubleNDArray(Shape shape, double[] entries) {
        super(shape, entries);
        ValidateParameters.ensureAllEqual(shape.totalEntriesIntValueExact(), entries.length);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    @Override
    public Double get(int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        return data[shape.get1DIndex(indices)];
    }


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     *
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     *
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     *
     * @throws ArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    @Override
    public Vector get(ArrayMask mask) {
        ValidateParameters.ensureEqualShape(shape, mask.shape);

        BitSet maskBits = mask.data;
        int trueCount = maskBits.cardinality();
        double[] values = new double[trueCount];

        int count = 0;
        for (int i = maskBits.nextSetBit(0); i >= 0; i = maskBits.nextSetBit(i + 1))
            values[count++] = data[i];

        return new Vector(values);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return A reference to this tensor.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(Double value, int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        data[shape.get1DIndex(indices)] = value;
        return (T) this;
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public T T(int axis1, int axis2) {
        return TransposeDispatcher.dispatchTensor((T) this, axis1, axis2);
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public T T(int... axes) {
        return makeLikeNDArray(shape.permuteAxes(axes),
                TransposeDispatcher.dispatchTensor(data, shape, axes));
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
     * @see #reduce(Double, DoubleBinaryOperator)
     * @see #reduce(Double, BinaryOperator, int...)
     */
    @Override
    public Tensor reduce(Double identity, DoubleBinaryOperator accumulator, int... axes) {
        Pair<Shape, double[]> reduced = ArrayReducer.reduce(shape, data, identity, accumulator, axes);
        return new Tensor(reduced.first(), reduced.second());
    }


    /**
     * Checks if each entry in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each entry in this nD array against.
     *
     * @return An {@link ArrayMask} of the same shape as this nD array containing the boolean results from evaluating each
     * entry in the nD array against the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #filter(Function)
     */
    @Override
    public ArrayMask where(Function<Double, Boolean> predicate) {
        BitSet mask = new BitSet(data.length);

        for(int i=0, size=data.length; i<size; i++)
            mask.set(i, predicate.apply(data[i]));

        return new ArrayMask(shape, mask);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        // TODO: For methods like RealDenseOperations.sub which only take arrays, the shape check should be done in the method itself
        //   and we will perform no bounds check in RealDenseOperations.sub (which needs to be documented within the method itself).
        ValidateParameters.ensureEqualShape(shape, b.shape);
        return makeLikeNDArray(shape, RealDenseOps.sub(data, b.data, null));
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void subEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        RealDenseOps.sub(data, b.data, data);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        return makeLikeNDArray(shape, RealDenseOps.add(data, b.data, null));
    }

    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     */
    public void addEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        RealDenseOps.add(data, b.data, data);
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T elemMult(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        return makeLikeNDArray(shape, RealDenseElemMult.elemMult(data, b.data, null));
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public void elemMultEq(T b) {
        RealDenseElemMult.elemMult(data,b.data, data);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 Tensor to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensor's shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public T tensorDot(T src2, int[] aAxes, int[] bAxes) {
        RealDenseTensorDot problem = new RealDenseTensorDot(
                shape, data, src2.shape, src2.data, aAxes, bAxes);
        return makeLikeNDArray(problem.getOutputShape(), problem.compute());
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values in the 2D subarrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param axis1 First axis for 2D subarray.
     * @param axis2 Second axis for 2D subarray.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensor's rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e., the axes are equal, or the tensor does not have the same length along the two axes.)
     */
    @Override
    public Tensor tensorTr(int axis1, int axis2) {
        Shape destShape = DenseSemiringOps.getTrShape(shape, axis1, axis2);
        double[] destEntries = new double[destShape.totalEntriesIntValueExact()];
        RealDenseOps.tensorTr(shape, data, axis1, axis2, destShape, destEntries);
        return new Tensor(destShape, destEntries);
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
        ValidateParameters.ensureEqualShape(shape, b.shape);
        return makeLikeNDArray(shape, RealDenseElemDiv.dispatch(data, b.data, null));
    }


    /**
     * Computes the element-wise quotient between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise quotient.
     */
    public void divEq(T b) {
        RealDenseElemDiv.dispatch(data, b.data, data);
    }



    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        if(data.length == 0) return new int[]{};
        return shape.getNdIndices(RealProperties.argminAbs(data));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        if(data.length == 0) return new int[]{};
        return shape.getNdIndices(RealProperties.argmaxAbs(data));
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        if(data.length == 0) return new int[]{};
        return shape.getNdIndices(RealProperties.argmin(data));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        if(data.length == 0) return new int[]{};
        return shape.getNdIndices(RealProperties.argmax(data));
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws ArrayShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        ValidateParameters.ensureTotalEntriesEqual(shape, newShape);
        return makeLikeNDArray(newShape, data.clone());
    }


    /**
     * Rounds all data in this matrix to the nearest integer.
     * @return A new matrix containing the data of this matrix rounded to the nearest integer.
     */
    public T round() {
        return round(0);
    }


    /**
     * Rounds all data within this matrix to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new matrix containing the data of this matrix rounded to the specified precision.
     */
    public T round(int precision) {
        return makeLikeNDArray(shape, RealOps.round(data, precision));
    }


    /**
     * Sets all elements of this matrix to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this matrix with all data within {@code tol} of zero set to zero.
     */
    public T roundToZero(double tolerance) {
        return makeLikeNDArray(shape, RealOps.roundToZero(data, tolerance));
    }


    /**
     * Checks if all data of this matrix are "close" as defined below. Custom tolerances may be specified using
     * {@link #allClose(AbstractDoubleNDArray, double, double)}.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same shape and all data are "close" element-wise, i.e.,
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (1E-08 + 1E-05*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDoubleNDArray, double, double)
     */
    public boolean allClose(T b) {
        return hasSameShape(b) && RealProperties.allClose(data, b.data);
    }


    /**
     * Checks if all data of this matrix are "close" as defined below.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same length and all data are "close" element-wise, i.e.,
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (absTol + relTol*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDoubleNDArray)
     */
    public boolean allClose(T b, double relTol, double absTol) {
        return hasSameShape(b) && RealProperties.allClose(data, b.data, relTol, absTol);
    }
}
