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
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingTensor;
import org.flag4j.arrays.sparse.CooRingTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.dense.DenseEquals;
import org.flag4j.numbers.Ring;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;


/**
 * <p>Instances of this class represent a dense tensor backed by a {@link Ring} array. The {@code RingTensor} class
 * provides functionality for tensor operations whose elements are members of a ring, supporting mutable data with a fixed shape.
 *
 * <p>A {@code RingTensor} is a generalization of the {@link  RingMatrix}, allowing for higher-dimensional data and operations
 * while maintaining the benefits of Ring-based arithmetic and dense storage.
 *
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Support for standard tensor operations like addition, element-wise multiplication, and reshaping.</li>
 *   <li>Conversion methods to other representations, including {@link RingMatrix}, {@link RingVector}, and COO
 *   format.</li>
 *   <li>Utility methods for computing properties like rank and shape</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <ul>
 *     <li>
 * Constructing a tensor from a {@code Shape shape} and flat data array.
 * This is generally the preferred and most efficient method of constructing a tensor.
 * <pre>{@code
 * // Constructing a ring tensor from a shape and flat data array.
 * Complex128[] data = {
 *     new RealInt32(1), new RealInt32(2),
 *     new RealInt32(3), new RealInt32(4),
 *     new RealInt32(5), new RealInt32(6),
 *     new RealInt32(7), new RealInt32(8)
 * };
 *
 * RingTensor<RealInt32> tensor = new RingTensor<>(data);
 * }</pre>
 *     </li>
 *
 *     <li>
 * Constructing a tensor from an nD array. This is provided for convenience but is generally much less efficient than
 * {@link #RingTensor(Shape, T[])}.
 * <pre>{@code
 * // Constructing a complex tensor from a 3D array of complex numbers
 * Complex128[][][] complexData = {
 *     {{ new RealInt32(1), new RealInt32(2) },
 *     {  new RealInt32(3), new RealInt32(4) }},
 *
 *     {{ new RealInt32(5), new RealInt32(6) },
 *     {  new RealInt32(7), new RealInt32(8) }}
 * };
 * RingTensor<RealInt32> tensor = new RingTensor<>(complexData);
 * }</pre>
 *     </li>
 *     <li>
 * Operations with/on tensors.
 * <pre>{@code
 * // Performing element-wise addition
 * RingTensor<RealInt32> result = tensor.add(tensor);
 *
 * // Reshape tensor
 * RingTensor<RealInt32> reshape = tensor.reshape(new Shape(4, 1, 2));
 *
 * // Converting the tensor to a matrix
 * RingMatrix<RealInt32> matrix = tensor.toMatrix(new Shape(4, 2));
 *
 * // Computing the tensor dot product.
 * RingTensor<RealInt32> dot = tensor.tensorDot(tensor,
 *      new int[]{0, 1},
 *      new int[]{2, 0}
 * );
 * }</pre>
 *     </li>
 * </ul>
 *
 * @param <T> Type of the {@link Ring ring} element for the tensor.
 *
 * @see Ring
 * @see RingMatrix
 * @see RingVector
 * @see AbstractDenseRingTensor
 */
public class RingTensor<T extends Ring<T>> extends AbstractDenseRingTensor<RingTensor<T>, T> {

    private static final long serialVersionUID = 1L;

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    public RingTensor(Shape shape, T[] data) {
        super(shape, data);
    }


    /**
     * Creates a tensor from an nD array. The tensors shape will be inferred from.
     * @param nDArray Array to construct tensor from. Must be a rectangular array.
     * @throws IllegalArgumentException If {@code nDArray} is not an array or not rectangular.
     */
    public RingTensor(Object nDArray) {
        super(ArrayUtils.nDArrayShape(nDArray),
                (T[]) new Ring[ArrayUtils.nDArrayShape(nDArray).totalEntriesIntValueExact()]);
        ArrayUtils.nDFlatten(nDArray, shape, data, 0);
    }


    /**
     * Creates a dense ring tensor with the specified data and filled with {@code filledValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Entries of this tensor.
     */
    public RingTensor(Shape shape, T fillValue) {
        super(shape, (T[]) new Ring[shape.totalEntriesIntValueExact()]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param data Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooRingTensor<T> makeLikeCooTensor(Shape shape, T[] data, int[][] indices) {
        return new CooRingTensor<>(shape, data, indices);
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
    public RingTensor<T> makeLikeNDArray(Shape shape, T[] data) {
        return new RingTensor<>(shape, data);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public RingVector<T> toVector() {
        return new RingVector<T>(new Shape(data.length), data.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must have the same number of {@link ValidateParameters#ensureTotalEntriesEqual(Shape, Shape) total entries}.
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public RingMatrix<T> toMatrix(Shape matShape) {
        ValidateParameters.ensureTotalEntriesEqual(shape, matShape);
        ValidateParameters.ensureRank(matShape, 2);

        return new RingMatrix<T>(matShape, data.clone());
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Tensor abs() {
        double[] dest = new double[data.length];
        RingOps.abs(data, dest);
        return new Tensor(shape, dest);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return {@code true} if the two tensors have the same shape, are numerically equivalent, and are of type {@link RingTensor}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        RingTensor<T> src2 = (RingTensor<T>) object;

        return DenseEquals.tensorEquals(this.data, this.shape, src2.data, src2.shape);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flattened data of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = shape.totalEntries().intValueExact();
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get data up until the stopping point.
        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();

        for(int i = 0; i<stopIndex; i++) {
            value = data[i].toString();
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = padding + 3;
            value = "...";
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry.
        value = data[size-1].toString();
        width = padding + value.length();
        value = centering ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
