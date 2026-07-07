/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
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

package org.flag4j.arrays_new.backend;

import org.flag4j.arrays.ArrayMask;
import org.flag4j.arrays.Shape;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.BinaryOperator;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;


/**
 * <p>The base abstract class for all numeric nD ndarrays including tensors, matrices, and vectors.
 *
 * <p>An nD array is a multidimensional array characterized by
 * <ul>
 *     <li><strong>Shape:</strong> The {@link #shape} of the nD array specifies the dimensions of the nD array along each axis.
 *     The number of axes in the nD array is referred to as the "{@link #getRank() rank}" and corresponds to the number of
 *     indices required to uniquely identify an element within the nD array.
 *     If the nD array's rank is 4, then it is a 4D array.</li>
 *     <li><strong>Data Buffer:</strong> A one-dimensional container for the {@link #dataBuffer} of the nD array.
 *     If the nD array is dense, this contains <em>all</em> the items of the nD array.
 *     If the nD array is sparse, this contains only the non-zero-elements of the nD array.</li>
 *     <li><strong>Strides:</strong> Specifies the offset between two consecutive elements along each axis.
 *     If the array is contiguous, then this may be <code>null</code>.
 *     </li>
 * </ul>
 *
 * <p>This abstract class provides common functionality and properties for all nD array types.
 * Subclasses should implement the abstract methods to provide specific behaviors for different
 * nD array types and items storage mechanisms (e.g., dense or sparse).
 *
 * @param <T> The specific type of the nD array (used for fluent API).
 * @param <U> The type of the items storage container for this nD array.
 * This should be a Java array, list, or similar list-like structure.
 * @param <V> The type (or wrapper) of the individual items elements in this nD array.
 * If the nD array's elements are primitive types, this should be their corresponding wrapper class.
 * If the elements are an {@link Object}, then this should be the same type as the object.
 */
public abstract class AbstractNDArray<T extends AbstractNDArray<T, U, V>, U, V> {

    /**
     * Entry items of this nD array. This is assumed to be a 1D, fixed size, homogeneous, container. Breaking <em>any</em> of
     * these assumptions may result in undefined behavior.
     * <ul>
     *     <li>If this nD array is dense, then this specifies <em>all</em> items within this nD array.</li>
     *     <li>If this nD array is sparse, this specifies <em>only</em> the non-zero items of this nD array.</li>
     * </ul>
     */
    public final U dataBuffer;
    /**
     * The shape of this nD array.
     */
    public final Shape shape;
    /**
     * The rank of this nD array.
     * That is, the number of indices required to uniquely specify an element in the nD array
     * (i.e., the number of axes within this nD array).
     */
    public final int rank;


    /**
     * Creates an nD array with the specified items and shape.
     * @param shape Shape of this nD array. Must be non-null.
     * @param dataBuffer Entries of this nD array. Must be non-null.
     * <ul>
     *     <li>If this nD array is dense, this specifies <em>all</em> items within this nD array.</li>
     *     <li>If this nD array is sparse, this specifies <em>only</em> the non-zero items of this nD array.</li>
     * </ul>
     */
    protected AbstractNDArray(Shape shape, U dataBuffer) {
        Objects.requireNonNull(shape, "Shape cannot be null.");
        Objects.requireNonNull(dataBuffer, "nD Array items cannot be null.");

        this.dataBuffer = dataBuffer;
        this.shape = shape;
        rank = shape.getRank();
    }


    /**
     * Gets the shape of this nD array.
     * @return The shape of this nD array.
     * @see #getRank()
     * @see #getShape(int)
     */
    public Shape getShape() {
        return shape;
    }


    /**
     * Gets the size of this nD array along the specified axis.
     * @param axis The axis along which to get the size of.
     * @return The size of this nD array along {@code axis}.
     * @see #getShape()
     * @see #getShape(int)
     */
    public int getShape(int axis) {
        return shape.getSize(axis);
    }


    /**
     * Gets the element of this nD array at the specified index.
     * @param index Index of the element to get.
     * @return The element of this nD array at the specified index.
     * @throws IndexOutOfBoundsException If {@code index} is not within the bounds of this nD array.
     * @throws org.flag4j.util.exceptions.NDArrayIndexException If {@code index} does not have the same rank as this nD array.
     */
    public abstract V get(int... index);


    /**
     * Gets a specified item from the internal items storage of this nD array.
     * @param i The index of the item to get within the internal items storage.
     * @return The item at the specified index within the internal items storage.
     */
    protected abstract V getStored(int i);


    /**
     * Gets the zero element of this nD array.
     * @return The zero value of this nD array.
     */
    protected abstract V getZeroElement();


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     * @throws NDArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    public abstract AbstractNDArray<?, U, V> get(ArrayMask mask);


    /**
     * Sets the element of this nD array at the specified index.
     * @param value New value to set the specified index of this nD array to.
     * @param index Index of the element to set.
     * @return If this nD array is dense, a reference to this nD array is returned.
     * If this nD array is sparse, a copy of this nD array with the updated value is returned.
     * @throws IndexOutOfBoundsException If {@code index} is not within the bounds of this nD array.
     */
    public abstract T set(V value, int... index);


    /**
     * <p>Gets the rank (order, degree, dimension, etc.) of this nD array.
     * That is, the number of indices needed to uniquely select an element of the nD array.
     *
     * <p>Note, this method is distinct from the {@code matrix rank}.
     *
     * @return The rank of this nD array.
     */
    public int getRank() {
        return rank;
    }


    /**
     * Gets a view of the entry items backing this nD array.
     * @return The view of the items of this nD array. Modifying this view <em>will</em> modify the items of this nD array.
     * @see #getDataCopy()
     */
    public U getDataView() {
        return dataBuffer;
    }


    /**
     * Gets a copy of the entry items backing this nD array.
     * @return A copy of the items of this nD array. modifying this copy <em>will not</em> modify the items of this nD array.
     * @see #getDataView()
     */
    public abstract U getDataCopy();


    /**
     * Gets the number of implicit zeros of this nD array.
     * An "implicit zero" is a zero not explicitly stored in the items of this object.
     * For instance, sparse ndarrays need not store zeros explicitly.
     * @return The number of implicit zeros of this nD array. If this array is dense, this will <em>@return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.).</em> be zero.
     */
    public BigInteger implicitZeroCount() {
        return numel().subtract(BigInteger.valueOf(dataBufferSize()));
    }


    /**
     * Checks if this nD array is sparse.
     * @return {@code true} if this nD array is sparse; otherwise {@code false}.
     * @see #isDense()
     * @see #getSparsity()
     */
    public boolean isSparse() {
        return implicitZeroCount().signum() > 0;
    }


    /**
     * Checks if this nD array is dense.
     * @return {@code true} if this nD array is dense; otherwise {@code false}.
     * @see #isSparse()
     * @see #getSparsity()
     */
    public boolean isDense() {
        return implicitZeroCount().signum() == 0;
    }


    /**
     * Computes the sparsity of this nD array. That is, the ratio of zero entries to total entries as a decimal percentage.
     * @return The sparsity of this nD array.
     *
     * @see #getDensity()
     * @see #isSparse()
     * @see #isDense()
     */
    public double getSparsity() {
        return SparseUtils.computeSparsity(shape, dataBufferSize());
    }


    /**
     * Computes the density of this nD array. That is, the ratio of non-zero entries to total entries as a decimal percentage.
     * @return The density of this nD array.
     *
     * @see #getSparsity()
     * @see #isSparse()
     * @see #isDense()
     */
    public double getDensity() {
        return 1.0 - getSparsity();
    }


    /**
     * Gets the size of the 1D items buffer object backing this nD array.
     * @return The size of the 1D items object backing this nD array.
     */
    public abstract int dataBufferSize();


    /**
     * Creates a deep copy of the specified 1D items buffer object.
     * @return A deep copy of the specified 1D items buffer object.
     */
    public abstract U copyBuffer(U buffer);


    /**
     * Gets the size of a 1D items buffer object.
     * @param buffer The 1D items buffer object. Must be non-null.
     * @return The size of the 1D items buffer object.
     */
    protected abstract int getSize(U buffer);


    /**
     * Gets the total number of elements in this nD array.
     * @return The total number of elements in this nD array.
     */
    public BigInteger numel() {
        return shape.numel();
    }


    /**
     * Checks if another nD array has the same shape as this nD array.
     * @param other The other nD array.
     * @return {@code true} if this nD array and {@code other} have the same shape; otherwise {@code false}.
     */
    public boolean hasEqualShape(AbstractNDArray<?, ?, ?> other) {
        return shape.equals(other.shape);
    }


    /**
     * Checks if another nD array has the same rank as this nD array.
     * @param other The other nD array.
     * @return {@code true} if this nD array and {@code other} have the same rank; otherwise {@code false}.
     */
    public boolean hasEqualRank(AbstractNDArray<?, ?, ?> other) {
        return rank == other.rank;
    }


    /**
     * Checks if another nD array has the same number of elements as this nD array.
     * @param other The other nD array.
     * @return {@code true} if this nD array and {@code other} have the same number of elements; otherwise {@code false}.
     */
    public boolean hasEqualNumel(AbstractNDArray<?, ?, ?> other) {
        return shape.numel().equals(other.shape.numel());
    }


    /**
     * <p>Flattens this nD array to a single dimension.
     * To preserve the rank of the nD array but flatten to single axes, use {@link #flatten(int)}.
     * <p>This preserves the order of the entries in the {@link #dataBuffer} object.
     *
     * @return The flattened nD array.
     * @see #flatten(int)
     */
    public AbstractNDArray<?, U, V> flatten() {
        return reshape(new Shape(shape.numel().intValueExact()));
    }


    /**
     * <p>Flattens an nD array along the specified axis. Unlike {@link #flatten()}, this method preserves the rank of the nD array.
     * <p>This preserves the order of the entries in the {@link #dataBuffer} object.
     *
     * @param axis Axis along which to flatten nD array.
     * @throws IndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()} - 1}.
     * @see #flatten()
     */
    public T flatten(int axis) {
        int[] dims = new int[rank];
        Arrays.fill(dims, 1);
        dims[axis] = shape.numel().intValueExact();
        return reshapeKeepRank(new Shape(dims));
    }


    /**
     * Reshapes this nD array. This will return a view whenever possible.
     * @param newShape New shape for the nD array.
     * @return When possible, a view of this nD array with the new shape. Otherwise, a copy of this nD array with the new shape.
     * @throws NDArrayShapeException If {@code newShape} does not have the same total number of entries as {@link #shape this.shape}.
     * @see #reshape(int...)
     * @see #reshapeKeepRank(Shape)
     * @see #reshapeKeepRank(int...)
     */
    public abstract AbstractNDArray<?, U, V> reshape(Shape newShape);


    /**
     * Copies and reshapes this nD array. This will return a view whenever possible.
     * @param dims The dimensions of the new shape.
     * @return When possible, a view of this nD array with the new shape. Otherwise, a copy of this nD array with the new shape.
     * @throws NDArrayShapeException If {@code dims} does not represent a shape with the same total number
     * of entries as {@link #shape this.shape}.
     * @see #reshape(Shape)
     * @see #reshapeKeepRank(Shape)
     * @see #reshapeKeepRank(int...)
     */
    public AbstractNDArray<?, U, V> reshape(int... dims) {
        return reshape(new Shape(dims));
    }


    /**
     * Reshapes this nD array without changing the rank. This will return a view whenever possible.
     * @param newShape The new shape. Must be the same rank as {@link #shape this.shape} and define the same number of entries.
     * @return When possible, a view of this nD array with the new shape. Otherwise, a copy of this nD array with the new shape.
     *
     * @see #reshape(Shape)
     * @see #reshape(int...)
     * @see #reshapeKeepRank(int...)
     */
    public abstract T reshapeKeepRank(Shape newShape);


    /**
     * Reshapes this nD array without changing the rank. This will return a view whenever possible.
     * @param newShape The new shape. Must be the same rank as {@link #shape this.shape} and define the same number of entries.
     * @return When possible, a view of this nD array with the new shape. Otherwise, a copy of this nD array with the new shape.
     *
     * @see #reshape(Shape)
     * @see #reshape(int...)
     * @see #reshapeKeepRank(int...)
     */
    public T reshapeKeepRank(int... dims) {
        return reshapeKeepRank(new Shape(dims));
    }


    /**
     * <p>Constructs an nD array of the same type as this nD array with the given {@code shape} and {@code items}.
     * <p>If this nD array is sparse, the resulting nD array will also have the same non-zero indices as this nD array.
     *
     * @param shape Shape of the nD array to construct.
     * @param data Entries of the nD array to construct.
     * @return An nD array of the same type and with the same non-zero indices as this nD array with the given the {@code shape} and
     * {@code items}.
     */
    abstract T makeLikeNDArray(Shape shape, U data);


    /**
     * Computes the transpose of a nD array by exchanging the first and last axes of this nD array.
     * @return The transpose of this nD array.
     * @see #T(int, int)
     * @see #T(int...)
     */
    public T T() {
        return T(0, getRank() - 1);
    }


    /**
     * Computes the transpose of a nD array by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this nD array along the specified axes.
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this nD array.
     * @see #T()
     * @see #T(int...)
     */
    public abstract T T(int axis1, int axis2);


    /**
     * Computes the transpose of this nD array.
     * That is, permutes the axes of this nD array so that it matches the permutation specified by {@code axes}.
     *
     * @param axes Permutation of nD array axis.
     * If the nD array has rank {@code n}, then this must be an array of length {@code n}
     * which is a permutation of {@code {0, 1, 2, ..., n-1}}.
     * @return The transpose of this nD array with its axes permuted by the {@code axes} array.
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this nD array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    public abstract T T(int... axes);


    /**
     * Creates a copy of this nD array.
     * @return A copy of this nD array. This will be a contiguous array.
     *
     * @see #copy(Shape)
     */
    public abstract T copy();


    /**
     * Creates a copy of this nD array with a new shape.
     * @return A copy of this nD array. This will be a contiguous array.
     *
     * @see #copy()
     */
    public abstract T copy(Shape shape);


    /**
     * Applies a mapping function to each items element of this nD array.
     * This function does <em>not</em> modify this nD array.
     * @param mapper The operation to apply to each items element.
     * @return The result of applying {@code mapper} to each items element of this nD array.
     * @see #mapDataInPlace(UnaryOperator)
     */
    protected abstract U mapData(UnaryOperator<V> mapper);


    /**
     * Applies a mapping function to each items element of this nD array.
     * @param mapper The operation to apply to each items element.
     * @return A copy of this nD array with the mapping function applied to each items element.
     *
     * @see #mapInPlace(UnaryOperator)
     * @throws IllegalArgumentException If {@code mapper} is not zero-preserving <em>and</em> this nD array is sparse.
     */
    public T map(UnaryOperator<V> mapper) {
        if (implicitZeroCount().signum() > 0 && !Objects.equals(mapper.apply(getZeroElement()), getZeroElement()))
            throw new IllegalArgumentException("Mapper is not zero-preserving; densify before mapping a sparse array.");
        return makeLikeNDArray(shape, mapData(mapper));
    }


    /**
     * Applies a mapping function to each items element of this nD array. This is done in-place.
     * @param mapper The operation to apply to each items element.
     * @return A reference to the items of this nD array.
     *
     * @see #mapData(UnaryOperator)
     */
    protected abstract U mapDataInPlace(UnaryOperator<V> mapper);


    /**
     * Applies a mapping function to each items element of this nD array. This is done in-place.
     * @param mapper The operation to apply to each items element.
     * @return A reference to this nD array.
     *
     * @see #map(UnaryOperator)
     * @throws IllegalArgumentException If {@code mapper} is not zero-preserving <em>and</em> this nD array is sparse.
     */
    public T mapInPlace(UnaryOperator<V> mapper) {
        if (implicitZeroCount().signum() > 0 && !Objects.equals(mapper.apply(getZeroElement()), getZeroElement()))
            throw new IllegalArgumentException("Mapper is not zero-preserving; densify before mapping a sparse array.");
        mapDataInPlace(mapper);
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
     * @return The final accumulated scalar of type {@code V}. If this nD array is empty, {@code identity} will be returned.
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @throws IllegalArgumentException If {@code identity} is {@code null} and this nD array is empty.
     *
     * @see #reduce(Object, BinaryOperator, int...)
     */
    public V reduce(V identity, BinaryOperator<V> accumulator) {
        if (dataBufferSize() == 0 && identity == null)
            throw new IllegalArgumentException("An identity value must be provided if the nD array is empty.");

        ensureZeroIsIdentity(identity, accumulator);

        V r = identity != null ? identity : getStored(0);
        int start = identity != null ? 0 : 1;

        for (int i = start, n = dataBufferSize(); i < n; i++)
            r = accumulator.apply(r, getStored(i));

        return r;
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
     * For the results to be well-defined, the accumulator must be associative and commutative.
     * @param axes The axes along which reduce this nD array.
     * @return An nD array of the same shape as this nD array but with the specified {@code axes} removed.
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     *
     * @see #reduce(Object, BinaryOperator)
     */
    public abstract AbstractNDArray<?, U, V> reduce(V identity, BinaryOperator<V> accumulator, int... axes);


    /**
     * Checks if each entry in this nD array satisfies the specified {@code predicate}.
     * @param predicate The predicate to check each entry in this nD array against.
     * @return An {@link ArrayMask} of the same shape as this nD array containing the boolean results from evaluating each
     * entry in the nD array against the {@code predicate}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     *
     * @see #filter(Predicate)
     */
    public abstract ArrayMask where(Predicate<V> predicate);


    /**
     * Extracts elements of this nD array that satisfy the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return A flat 1D array containing the elements of this nD array that satisfy the {@code predicate}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     *
     * @see #where(Predicate)
     */
    public abstract AbstractNDArray<?, ?, V> filter(Predicate<V> predicate);


    /**
     * Checks if <em>any</em> element in this nD array satisfies the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return {@code true} if <em>any</em> element in this nD array satisfies the {@code predicate}; otherwise {@code false}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #all(Predicate)
     */
    public boolean any(Predicate<V> p) {
        for (int i = 0, n = dataBufferSize(); i < n; i++) if (p.test(getStored(i))) return true;
        return implicitZeroCount().signum() > 0 && p.test(getZeroElement());
    }


    /**
     * Checks if <em>all</em> elements in this nD array satisfy the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return {@code true} if <em>all</em> elements in this nD array satisfy the {@code predicate}; otherwise {@code false}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #any(Predicate)
     */
    public boolean all(Predicate<V> p) {
        for (int i = 0, n = dataBufferSize(); i < n; i++) if (!p.test(getStored(i))) return false;
        return implicitZeroCount().signum() == 0 || p.test(getZeroElement());
    }


    /**
     * Counts the number of elements in this nD array which satisfy the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return The number of elements in this nD array which satisfy the specified {@code predicate}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     */
    public BigInteger count(Predicate<V> predicate) {
        long stored = 0;
        for (int i = 0, n = dataBufferSize(); i < n; i++) if (predicate.test(getStored(i))) stored++;
        BigInteger total = BigInteger.valueOf(stored);

        var implicitZeroCount = implicitZeroCount();
        var requiresImplicitZero = implicitZeroCount.signum() > 0 && predicate.test(getZeroElement());

        return requiresImplicitZero ? total.add(implicitZeroCount) : total;
    }


    /**
     * Checks if this nD array is equal to the specified object.
     * @param other the reference object with which to compare.
     * @return {@code true} if this object is the same as the {@code other} object; {@code false} otherwise.
     */
    public abstract boolean equals(Object other);


    /**
     * Computes a hash code for this nD array.
     * @return The hash code for this nD array.
     */
    public abstract int hashCode();


    /**
     * Verifies that the implicit zeros of a sparse array may be ignored during a reduction with the given
     * {@code accumulator}. They may be ignored iff {@link #getZeroElement() zero} is an identity element of
     * {@code accumulator}, i.e. {@code accumulator.apply(x, zero).equals(x)}. This is checked heuristically
     * by probing the accumulator at {@code identity}.
     *
     * @throws IllegalArgumentException If this array has implicit zeros and {@code accumulator} does not treat
     * zero as an identity element; densify before reducing in that case.
     */
    protected void ensureZeroIsIdentity(V identity, BinaryOperator<V> accumulator) {
        var isSparse = implicitZeroCount().signum() > 0;

        if (isSparse && identity == null) {
            throw new IllegalArgumentException("An identity value must be provided if the nD array is sparse.");
        }

        if (isSparse
                && !Objects.equals(accumulator.apply(identity, getZeroElement()), identity)) {
            throw new IllegalArgumentException(
                    "Accumulator does not treat zero as an identity; densify before reducing a sparse array.");
        }
    }
}
