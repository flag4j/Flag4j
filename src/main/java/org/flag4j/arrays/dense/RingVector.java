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

package org.flag4j.arrays.dense;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingVector;
import org.flag4j.arrays.sparse.CooRingVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.numbers.Ring;

import java.util.Arrays;


/**
 * <p>Instances of this class represent a dense vector backed by a {@link Ring} array.
 * The {@code RingVector} class provides functionality for matrix operations whose elements are members of a ring, supporting
 * mutable data with a fixed shape.
 *
 * <p>A {@code RingVector} is essentially equivalent to a rank-1 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general rank-n tensors.
 *
 * <p><b>Key Features:</b>
 * <ul>
 *   <li>Support for standard vector operations like addition, subtraction, and inner/outer products.</li>
 *   <li>Conversion methods to other representations, such as {@link RingMatrix}, {@link RingTensor}, or COO (Coordinate).</li>
 *   <li>Utility methods for checking properties like being the zero vector.</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Constructing a complex matrix from an array of complex numbers
 * RealInt32[] data = {
 *     new RealInt32(1), new RealInt32(2),
 *     new RealInt32(3), new RealInt32(4)
 * };
 * RingVector<RealInt32> vector = new RingVector(data);
 *
 * // Performing vector inner/outer product.
 * RealInt32 inner = vector.inner(vector);
 * RingMatrix<RealInt32> outer = vector.outer(vector);
 *
 * // Checking if the vector only contains zeros.
 * boolean isZero = vector.isZeros();
 * }</pre>
 *
 * @param <T> Type of the {@link Ring ring} element for the matrix.
 *
 * @see RingVector
 * @see RingTensor
 * @see org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingMatrix
 */
public class RingVector<T extends Ring<T>> extends AbstractDenseRingVector<RingVector<T>, RingMatrix<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a ring vector with the specified data and shape.
     *
     * @param data Entries of the vector.
     */
    public RingVector(T[] data) {
        super(new Shape(data.length), data);
    }


    /**
     * Creates a ring vector with the specified data and shape.
     *
     * @param shape Shape of the vector to construct.
     * @param data Entries of the vector.
     */
    public RingVector(Shape shape, T[] data) {
        super(shape, data);
    }


    /**
     * Constructs a dense vector with the specified {@code data} of the same type as the vector.
     *
     * @param entries Entries of the dense vector to construct.
     */
    @Override
    public RingVector<T> makeLikeTensor(T[] entries) {
        return new RingVector<>(entries);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public RingVector<T> makeLikeNDArray(Shape shape, T[] data) {
        return new RingVector<>(shape, data);
    }


    /**
     * Constructs a matrix of a similar type to this vector with the specified {@code shape} and {@code data}.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of a similar type to this vector with the specified {@code shape} and {@code data}.
     */
    @Override
    protected RingMatrix<T> makeLikeMatrix(Shape shape, T[] entries) {
        return new RingMatrix<>(shape, entries);
    }


    /**
     * Constructs a sparse COO tensor, which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param data Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooRingVector<T> makeLikeCooTensor(Shape shape, T[] data, int[][] indices) {
        return new CooRingVector<>(shape, data, indices[0]);
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor equivalent to this vector.
     */
    public RingTensor<T> toTensor() {
        return new RingTensor(shape, data.clone());
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @param shape The desired shape of the resulting tensor.
     * @return A tensor with the specified {@code shape} containing the entries of this vector.
     */
    public RingTensor<T> toTensor(Shape shape) {
        return new RingTensor(shape, data.clone());
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Vector abs() {
        double[] dest = new double[data.length];
        RingOps.abs(data, dest);
        return new Vector(shape, dest);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return {@code true} if the two vectors have the same shape, are numerically equivalent, and are of type {@link RingVector}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        RingVector<T> src2 = (RingVector<T>) object;

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of data to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");

        result.append(PrettyPrint.abbreviatedArray(data,
                PrintOptions.getMaxColumns(),
                PrintOptions.getPadding(),
                PrintOptions.getPrecision(),
                PrintOptions.useCentering()));

        return result.toString();
    }
}
