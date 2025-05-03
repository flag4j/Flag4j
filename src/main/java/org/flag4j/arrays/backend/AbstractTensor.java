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

package org.flag4j.arrays.backend;


import org.flag4j.arrays.Shape;

import java.io.Serializable;
import java.math.BigInteger;

/**
 * <p>The base abstract class for all tensors, including matrices and vectors.
 *
 * <p>A tensor is a multidimensional array that consists of:
 * <ul>
 *     <li><b>Shape:</b> The {@link #shape} of the tensor specifies the dimensions of the tensor along each axis.
 *     The number of axes in the tensor is referred to as the "{@link #rank}" and corresponds to the number of
 *     indices required to uniquely identify an element within the tensor.</li>
 *     <li><b>Data:</b> A one-dimensional container for the {@link #data} of the tensor.
 *     If the tensor is dense, this contains all the data of the tensor.
 *     If the tensor is sparse, this contains only the non-zero-elements of the tensor.</li>
 * </ul>
 *
 * <p>This abstract class provides common functionality and properties for all tensor types.
 * Subclasses should implement the abstract methods to provide specific behaviors for different
 * tensor types and data storage mechanisms (e.g., dense or sparse).
 *
 * @param <T> The specific type of the tensor (used for method return types in a fluent API).
 * @param <U> The type of the data storage container for this tensor. This should be an array, list, or similar list-like structure.
 * @param <V> The type (or wrapper) of the individual data elements in this tensor. If the tensors elements are primitive types, this
 * should be their corresponding wrapper class. If the elements are an {@link Object}, then this should be the same type as the
 * object.
 * @see org.flag4j.arrays.dense
 * @see org.flag4j.arrays.sparse
 */
public abstract class AbstractTensor<T extends AbstractTensor<T, U, V>, U, V>
        implements Serializable {

    /**
     * Entry data of this tensor. If this tensor is dense, then this specifies all data within this tensor. If this tensor is
     * sparse, this specifies only the non-zero data of this tensor.
     */
    public final U data;
    /**
     * The shape of this tensor.
     */
    public final Shape shape;
    /**
     * The rank of this tensor. That is, the number of indices required to uniquely specify an element in the tensor (the number of
     * axes within this tensor).
     */
    public final int rank;


    /**
     * Creates a tensor with the specified data and shape.
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     *                If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractTensor(Shape shape, U data) {
        this.shape = shape;
        this.data = data;
        rank = shape.getRank();
    }


    /**
     * Gets the shape of this tensor.
     * @return The shape of this tensor.
     */
    public Shape getShape() {
        return shape;
    }


    /**
     * Gets the element of this tensor at the specified indices.
     * @param indices Indices of the element to get.
     * @return The element of this tensor at the specified indices.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    public abstract V get(int... indices);


    /**
     * Sets the element of this tensor at the specified indices.
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    public abstract T set(V value, int... indices);


    /**
     * <p>Gets the rank of this tensor. That is, number of indices needed to uniquely select an element of the tensor. This is also te
     * number of dimensions (i.e., order/degree) of the tensor.
     *
     * <p>Note, this method is distinct from the {@code matrix rank}.
     *
     * @return The rank of this tensor.
     */
    public int getRank() {
        return rank;
    }


    /**
     * Gets the entry data of this tensor as a 1D array.
     * @return The data of this tensor.
     */
    public U getData() {
        return data;
    }


    /**
     * Gets the size of the 1D data array backing this tensor.
     * @return The size of the 1D data array backing this tensor.
     */
    public abstract int dataLength();


    /**
     * Gets the total number of data in this tensor.
     * @return The total number of data in this tensor.
     */
    public BigInteger totalEntries() {
        return shape.totalEntries();
    }


    /**
     * Checks if a tensor has the same shape as this tensor.
     * @param B Second tensor.
     * @return True if this tensor and B have the same shape. False otherwise.
     */
    public boolean sameShape(AbstractTensor<?, ?, ?> B) {
        return shape.equals(B.shape);
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
     *
     * @return The flattened tensor.
     * @see #flatten(int)
     */
    public abstract T flatten();


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    public abstract T flatten(int axis);


    /**
     * Copies and reshapes this tensor.
     * @param newShape New shape for the tensor.
     * @return A copy of this tensor with the new shape.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    public abstract T reshape(Shape newShape);


    /**
     * Copies and reshapes this tensor.
     * @param dims The dimensions of the new shape.
     * @return A copy of this tensor with the new shape.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code dims} does not represent a shape with the same total number
     * of entries as {@link #shape this.shape}.
     */
    public T reshape(int... dims) {
        return reshape(new Shape(dims));
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    public abstract T makeLikeTensor(Shape shape, U entries);


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     * @return The transpose of this tensor.
     * @see #T(int, int)
     * @see #T(int...)
     */
    public T T() {
        return T(0, getRank()-1);
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor along the specified axes.
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    public abstract T T(int axis1, int axis2);


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    public abstract T T(int... axes);


    /**
     * Creates a deep copy of this tensor.
     * @return A deep copy of this tensor.
     */
    public abstract T copy();
}
