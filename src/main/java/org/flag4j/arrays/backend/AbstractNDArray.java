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

package org.flag4j.arrays.backend;


import org.flag4j.arrays.ArrayMask;
import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.ArrayShapeException;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Objects;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * <p>The base abstract class for all nD arrays including tensors, matrices, and vectors.
 *
 * <p>An nD is a multidimensional array characterized by
 * <ul>
 *     <li><strong>Shape:</strong> The {@link #shape} of the nD array specifies the dimensions of the nD array along each axis.
 *     The number of axes in the nD array is referred to as the "{@link #getRank() rank}" and corresponds to the number of
 *     indices required to uniquely identify an element within the nD array.
 *     If the nD array's rank is 4, then it is a 4D array.</li>
 *     <li><strong>Data:</strong> A one-dimensional container for the {@link #data} of the nD array.
 *     If the nD array is dense, this contains <em>all</em> the data of the nD array.
 *     If the nD array is sparse, this contains only the non-zero-elements of the nD array.</li>
 * </ul>
 *
 * <p>This abstract class provides common functionality and properties for all nD array types.
 * Subclasses should implement the abstract methods to provide specific behaviors for different
 * nD array types and data storage mechanisms (e.g., dense or sparse).
 *
 * @param <T> The specific type of the nD array (used for fluent API).
 * @param <U> The type of the data storage container for this nD array.
 * This should be a Java array, list, or similar list-like structure.
 * @param <V> The type (or wrapper) of the individual data elements in this nD array.
 * If the nD array's elements are primitive types, this should be their corresponding wrapper class.
 * If the elements are an {@link Object}, then this should be the same type as the object.
 *
 * @see org.flag4j.arrays.dense
 * @see org.flag4j.arrays.sparse
 */
public abstract class AbstractNDArray<T extends AbstractNDArray<T, U, V>, U, V>
        implements Serializable {

    // TODO: An nD index should be called an "index" not "indices". Need to change verbiage in this class (and others).
    //  "indices" should only be used 

    /**
     * Entry data of this nD array.
     * <ul>
     *     <li>If this nD array is dense, then this specifies <em>all</em> data within this nD array.</li>
     *     <li>If this nD array is sparse, this specifies <em>only</em> the non-zero data of this nD array.</li>
     * </ul>
     */
    public final U data;
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
     * Creates an nD array with the specified data and shape.
     * @param shape Shape of this nD array.
     * @param data Entries of this nD array.
     * <ul>
     *     <li>If this nD array is dense, this specifies <em>all</em> data within this nD array.</li>
     *     <li>If this nD array is sparse, this specifies <em>only</em> the non-zero data of this nD array.</li>
     * </ul>
     */
    protected AbstractNDArray(Shape shape, U data) {
        Objects.requireNonNull(shape, "Shape cannot be null.");
        Objects.requireNonNull(data, "Tensor data cannot be null.");

        this.shape = shape;
        this.data = data;
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
     */
    public abstract V get(int... index);


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     * @throws ArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    public abstract AbstractNDArray<?, ?, V> get(ArrayMask mask);

    // TODO: Need to add the get(set)Slice and get(set)Items methods definitions here.

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
     * Gets the entry data of this nD array as a 1D array.
     * @return The data of this nD array.
     */
    public U getData() {
        return data;
    }


    /**
     * Gets the size of the 1D data object backing this nD array.
     * @return The size of the 1D data object backing this nD array.
     */
    public abstract int dataLength();


    /**
     * Gets the total number of elements in this nD array.
     * @return The total number of elements in this nD array.
     */
    public BigInteger totalEntries() {
        return shape.totalEntries();
    }


    /**
     * Checks if another nD array has the same shape as this nD array.
     * @param b Second nD array.
     * @return {@code true} if this nD array and {@code b} have the same shape; otherwise {@code false}.
     */
    public boolean hasSameShape(AbstractNDArray<?, ?, ?> b) {
        return shape.equals(b.shape);
    }


    /**
     * <p>Flattens this nD array to a single dimension.
     * To preserve the rank of the nD array but flatten to single axes, use {@link #flatten(int)}.
     * <p>This preserves the order of the entries in the {@link #data} object.
     *
     * @return The flattened nD array.
     * @see #flatten(int)
     */
    public abstract T flatten();


    /**
     * <p>Flattens an nD array along the specified axis. Unlike {@link #flatten()}, this method preserves the rank of the nD array.
     * <p>This preserves the order of the entries in the {@link #data} object.
     *
     * @param axis Axis along which to flatten nD array.
     * @throws IndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()} - 1}.
     * @see #flatten()
     */
    public abstract T flatten(int axis);


    /**
     * Copies and reshapes this nD array.
     * @param newShape New shape for the nD array.
     * @return A copy of this nD array with the new shape.
     * @throws ArrayShapeException If {@code newShape} does not have the same total number of entries as {@link #shape this.shape}.
     * @see #reshape(int...) 
     */
    public abstract T reshape(Shape newShape);


    /**
     * Copies and reshapes this nD array.
     * @param dims The dimensions of the new shape.
     * @return A copy of this nD array with the new shape.
     * @throws ArrayShapeException If {@code dims} does not represent a shape with the same total number
     * of entries as {@link #shape this.shape}.
     * @see #reshape(Shape)
     */
    public T reshape(int... dims) {
        return reshape(new Shape(dims));
    }


    /**
     * <p>Constructs an nD array of the same type as this nD array with the given {@code shape} and {@code data}.
     * <p>If this nD array is sparse, the resulting nD array will also have the same non-zero indices as this nD array.
     * @param shape Shape of the nD array to construct.
     * @param data Entries of the nD array to construct.
     * @return An nD array of the same type and with the same non-zero indices as this nD array with the given the {@code shape} and
     * {@code data}.
     */
    public abstract T makeLikeNDArray(Shape shape, U data);


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
     * @return A copy of this nD array.
     */
    public abstract T copy();


    /**
     * Applies a map to each item in this nD array. This operation is done in-place.
     * If this nD array is sparse, the {@code mapper} operation will only be applied to the non-zero
     * elements in this nD array.
     * @param mapper The operation to apply to each item in this nD array.
     * @return A reference to this nD array.
     * @throws NullPointerException If {@code mapper} is {@code null}.
     */
    public abstract T map(UnaryOperator<V> mapper);


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
     *
     * @see #reduce(V, BinaryOperator, int...)
     */
    public abstract V reduce(V identity, BinaryOperator<V> accumulator);


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
     * @return An nD array of the same shape as this nD array but with the specified {@code axes} removed.
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     *
     * @see #reduce(V, BinaryOperator)
     */
    public abstract AbstractNDArray<?, ?, V> reduce(V identity, BinaryOperator<V> accumulator, int... axes);


    /**
     * Checks if each entry in this nD array satisfies the specified {@code predicate}.
     * @param predicate The predicate to check each entry in this nD array against.
     * @return An {@link ArrayMask} of the same shape as this nD array containing the boolean results from evaluating each
     * entry in the nD array against the {@code predicate}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     *
     * @see #filter(Function)
     */
    public abstract ArrayMask where(Function<V, Boolean> predicate);


    /**
     * Extracts elements of this nD array that satisfy the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return A flat 1D array containing the elements of this nD array that satisfy the {@code predicate}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     *
     * @see #where(Function)
     */
    public abstract AbstractNDArray<?, ?, V> filter(Function<V, Boolean> predicate);


    /**
     * Checks if <em>any</em> element in this nD array satisfies the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return {@code true} if <em>any</em> element in this nD array satisfies the {@code predicate}; otherwise {@code false}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #all(Function)
     */
    public abstract boolean any(Function<V, Boolean> predicate);


    /**
     * Checks if <em>all</em> elements in this nD array satisfy the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return {@code true} if <em>all</em> elements in this nD array satisfy the {@code predicate}; otherwise {@code false}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #any(Function)
     */
    public abstract boolean all(Function<V, Boolean> predicate);


    /**
     * Counts the number of elements in this nD array which satisfy the specified {@code predicate}.
     * @param predicate The predicate to check each element in this nD array against.
     * @return The number of elements in this nD array which satisfy the specified {@code predicate}.
     * @throws NullPointerException If {@code predicate} is {@code null}.
     */
    public abstract int countTrue(Function<V, Boolean> predicate);
}
