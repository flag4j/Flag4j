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

package org.flag4j.arrays.backend.semiring_arrays;

import org.flag4j.arrays.ArrayMask;
import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.ops.dense.DenseSemiringTensorDot;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringConversions;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringElemMult;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringOps;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ArrayMapper;
import org.flag4j.util.ArrayReducer;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * <p>The base class for all dense {@link SemiringElement} tensors.
 * <p>The {@link #data} of an AbstractDenseSemiringTensor are mutable but the {@link #shape} is fixed.
 *
 * @param <T> The type of this dense semiring tensor.
 * @param <V> The type of the {@link SemiringElement} which this tensor's items belong to.
 */
public abstract class AbstractDenseSemiringTensor<T extends AbstractDenseSemiringTensor<T, V>, V extends SemiringElement<V>>
        extends AbstractNDArray<T, V[], V>
        implements SemiringTensorMixin<T, T, V> {

    /**
     * The zero-element for the semiring that this tensor's elements belong to.
     */
    protected V zeroElement;


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     * @throws IllegalArgumentException If {@code shape.totalEntriesIntValueExact() != items.length}
     */
    protected AbstractDenseSemiringTensor(Shape shape, V[] data) {
        super(shape, data);
        ValidateParameters.ensureAllEqual(shape.numelIntValueExact(), data.length);
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
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
     * Sets the zero-element for this tenor's field.
     * @param zeroElement The zero-element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the field.
     *
     * @see #getZeroElement()
     */
    public void setZeroElement(V zeroElement) {
        if (zeroElement.isZero())
            this.zeroElement = zeroElement;
        else
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
    }


    /**
     * Gets the zero-element for this tenor's field.
     * @return The zero-element for this tenor's field. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(SemiringElement)} then {@code null} will be returned.
     *
     * @see #setZeroElement(SemiringElement)
     */
    public V getZeroElement() {
        return zeroElement;
    }


    /**
     * Constructs a sparse COO tensor, which is of a similar type as this dense tensor.
     * @param shape Shape of the COO tensor.
     * @param data Non-zero items of the COO tensor.
     * @param rowIndices Non-zero row indices of the COO tensor.
     * @param colIndices Non-zero column indices of the COO tensor.
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    protected abstract AbstractNDArray<?, V[], V> makeLikeCooTensor(
            Shape shape, V[] data, int[][] indices);


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws IndexOutOfBoundsException If any indices are not within this tensor.
     */
    @Override
    public V get(int... indices) {
        return (V) data[shape.get1DIndex(indices)];
    }


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     *
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     *
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     *
     * @throws NDArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    @Override
    public AbstractNDArray<?, ?, V> get(ArrayMask mask) {
        ValidateParameters.ensureEqualShape(this.shape, mask.shape);
        V[] values = makeEmptyDataArray(mask.cardinality());
        BitSet maskData = mask.data;

        for (int i = maskData.nextSetBit(0), vIdx = 0; i >= 0; i = maskData.nextSetBit(i + 1))
            values[vIdx++] = data[i];

        return makeLikeNDArray(new Shape(values.length), values);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(V value, int... indices) {
        data[shape.get1DIndex(indices)] = value;
        return (T) this;
    }


    /**
     * Flattens tensor to a single dimension while preserving the order of items.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return makeLikeNDArray(shape.flatten(), data.clone());
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.numel().intValueExact();
        Shape flatShape = new Shape(dims);

        return makeLikeNDArray(flatShape, data.clone());
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws NDArrayShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        // No need to make explicit total entries check as the constructor should verify that the number of items in the shape
        // matches the number of items in the array.
        return makeLikeNDArray(newShape, data.clone());
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws NDArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        V[] sum = makeEmptyDataArray(data.length);
        DenseSemiringOps.add(data, shape, b.data, b.shape, sum);
        return makeLikeNDArray(shape, sum);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     */
    public void addEq(T b) {
        DenseSemiringOps.add(data, shape, b.data, b.shape, data);
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
        V[] prod = makeEmptyDataArray(data.length);
        DenseSemiringElemMult.dispatch(data, shape, b.data, b.shape, prod);
        return makeLikeNDArray(shape, prod);
    }


    /**
     * Computes the element-wise multiplication of two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public void elemMultEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=data.length; i<size; i++)
            data[i] = data[i].mult(b.data[i]);
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
        DenseSemiringTensorDot<V> dot = new DenseSemiringTensorDot(shape, data, src2.shape, src2.data, aAxes, bAxes);
        V[] dest = makeEmptyDataArray(dot.getOutputSize());
        dot.compute(dest);
        return makeLikeNDArray(dot.getOutputShape(), dest);
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
    public T tensorTr(int axis1, int axis2) {
        Shape destShape = DenseSemiringOps.getTrShape(shape, axis1, axis2);
        V[] destEntries = makeEmptyDataArray(destShape.numelIntValueExact());
        DenseSemiringOps.tensorTr(shape, data, axis1, axis2, destShape, destEntries);
        return makeLikeNDArray(destShape, destEntries);
    }


    /**
     * Computes the sum of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the sum. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     *
     * @see #sum()
     */
    @Override
    public TensorOverSemiring<?, ?, ?, V> sum(int... axes) {
        return reduce(getZeroElement(), (V a, V b) -> a.add(b), axes);
    }


    /**
     * Computes the product of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the product. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     *
     * @see #prod()
     */
    @Override
    public TensorOverSemiring<?, ?, ?, V> prod(int... axes) {
        return reduce(getZeroElement(), (V a, V b) -> a.mult(b), axes);
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex-valued tensor) in this tensor.
     */
    public V min() {
        return CompareSemiring.min(data);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex-valued tensor) in this tensor.
     */
    public V max() {
        return CompareSemiring.max(data);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    public int[] argmin() {
        return shape.getNdIndices(CompareSemiring.argmin(data));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    public int[] argmax() {
        return shape.getNdIndices(CompareSemiring.argmax(data));
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
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensor(data, shape, axis1, axis2, dest);
        return makeLikeNDArray(shape.swapAxes(axis1, axis2), dest);
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
        ValidateParameters.ensureValidAxes(shape, axes);
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensor(data, shape, axes, dest);
        return makeLikeNDArray(shape.permuteAxes(axes), dest);
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
     * Applies a map to each item in this nD array.
     * This operation is done in-place.
     * If this nD array is sparse, the {@code mapper} operation will only be applied to the non-zero
     * elements in this nD array.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A reference to this nD array.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     */
    @Override
    public T map(UnaryOperator<V> mapper) {
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
     * @return The final accumulated scalar of type {@code V}. If this nD array is empty, {@code identity} will be returned.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator, int...)
     */
    @Override
    public V reduce(V identity, BinaryOperator<V> accumulator) {
        return ArrayReducer.reduce(data, identity, accumulator);
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
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
     * @return An nD array of the same shape as this nD array but with the specified {@code axes} removed.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator)
     */
    @Override
    public AbstractDenseSemiringTensor<?, V> reduce(V identity, BinaryOperator<V> accumulator, int... axes) {
        V[] dest = makeEmptyDataArray(data.length);
        Pair<Shape, V[]> reduced = ArrayReducer.reduce(shape, data, identity, accumulator, dest, axes);
        return makeLikeNDArray(reduced.first(), reduced.second());
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
    public ArrayMask where(Function<V, Boolean> predicate) {
        BitSet mask = new BitSet(data.length);

        for(int i=0, size=data.length; i<size; i++)
            if(predicate.apply(data[i])) mask.set(i);

        return new ArrayMask(shape, mask);
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
    public AbstractDenseSemiringTensor<?, V> filter(Function<V, Boolean> predicate) {
        List<V> dest = new ArrayList<>();

        for(V value : data)
            if(predicate.apply(value)) dest.add(value);

        return makeLikeNDArray(shape, dest.toArray(makeEmptyDataArray(dest.size())));
    }


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
    public boolean any(Function<V, Boolean> predicate) {
        for(V value : data)
            if(predicate.apply(value)) return true;

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
    public boolean all(Function<V, Boolean> predicate) {
        for(V value : data)
            if(!predicate.apply(value)) return false;

        return true;
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
    public int countTrue(Function<V, Boolean> predicate) {
        // TODO: Implement this method
        return 0;
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public AbstractNDArray<?, V[], V> toCoo() {
        return toCoo(0.9);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the primary concern, it is better to underestimate the sparsity.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo()
     */
    public AbstractNDArray<?, V[], V> toCoo(double estimatedSparsity) {
        SparseTensorData<V> data = DenseSemiringConversions.toCooTensor(shape, this.data, estimatedSparsity);
        V[] cooEntries = data.data().toArray(makeEmptyDataArray(data.data().size()));

        // TODO: First check if this tensor is a vector then delegate to specialized toCooVector
        //  or toCooTensor methods.
        if(this instanceof VectorMixin<?,?,?,?>) {
            return makeLikeCooTensor(
                    data.shape(), cooEntries,
                    RealDenseTranspose.standardIntMatrix(data.indicesToArray()));
        } else {
            return makeLikeCooTensor(data.shape(), cooEntries, data.indicesToArray());
        }
    }
}
