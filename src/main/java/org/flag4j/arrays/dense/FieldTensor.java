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
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldTensor;
import org.flag4j.arrays.sparse.CooFieldTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.dense.DenseEquals;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;


/**
 * <p>Instances of this class represent a dense tensor backed by a {@link Field} array. The {@code FieldTensor} class
 * provides functionality for tensor operations whose elements are members of a field, supporting mutable data with a fixed shape.
 *
 * <p>A {@code FieldTensor} is a generalization of the {@link  FieldMatrix}, allowing for higher-dimensional data and operations
 * while maintaining the benefits of field-based arithmetic and dense storage.
 *
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Support for standard tensor operations like addition, subtraction, element-wise multiplication, and reshaping.</li>
 *   <li>Conversion methods to other representations, including {@link FieldMatrix}, {@link FieldVector}, and COO
 *   format.</li>
 *   <li>Utility methods for computing properties like rank and shape</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 *
 * <ul>
 *     <li>
 * Constructing a tensor from a {@code Shape shape} and flat data array.
 * This is generally the preferred and most efficient method of constructing a tensor.
 * <pre>{@code
 * // Constructing a complex tensor from a shape and flat data array.
 * Complex128[] complexData = {
 *     new Complex128(1, 2), new Complex128(3, 4),
 *     new Complex128(5, 6), new Complex128(7, 8),
 *     new Complex128(9, 10), new Complex128(11, 12),
 *     new Complex128(13, 14), new Complex128(15, 16)
 * };
 *
 * FieldTensor<Complex128> tensor = new FieldTensor<>(complexData);
 * }</pre>
 *     </li>
 *
 *     <li>
 * Constructing a tensor from an nD array. This is provided for convenience but is generally much less efficient than
 * {@link #FieldTensor(Shape, T[])}.
 * <pre>{@code
 * // Constructing a complex tensor from a 3D array of complex numbers
 * Complex128[][][] complexData = {
 *     {{ new Complex128(1, 2), new Complex128(3, 4) },
 *     {  new Complex128(5, 6), new Complex128(7, 8) }},
 *
 *     {{ new Complex128(9, 10),  new Complex128(11, 12) },
 *     {  new Complex128(13, 14), new Complex128(15, 16) }}
 * };
 * FieldTensor<Complex128> tensor = new FieldTensor<>(complexData);
 * }</pre>
 *     </li>
 *     <li>
 * Operations with/on tensors.
 * <pre>{@code
 * // Performing element-wise addition
 * FieldTensor<Complex128> result = tensor.add(tensor);
 *
 * // Reshape tensor
 * FieldTensor<Complex128> reshape = tensor.reshape(new Shape(4, 1, 2));
 *
 * // Converting the tensor to a matrix
 * FieldMatrix<Complex128> matrix = tensor.toMatrix(new Shape(4, 2));
 *
 * // Computing the tensor dot product.
 * FieldTensor<Complex128> dot = tensor.tensorDot(tensor, new int[]{0, 1}, new int[]{2, 0});
 * }</pre>
 *     </li>
 * </ul>
 *
 * @param <T> Type of the {@link Field field} element for the tensor.
 *
 * @see Field
 * @see FieldMatrix
 * @see FieldVector
 * @see AbstractDenseFieldTensor
 */
public class FieldTensor<T extends Field<T>> extends AbstractDenseFieldTensor<FieldTensor<T>, T> {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    public FieldTensor(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a tensor from an nD array. The tensors shape will be inferred from.
     * @param nDArray Array to construct tensor from. Must be a rectangular array.
     * @throws IllegalArgumentException If {@code nDArray} is not an array or not rectangular.
     */
    public FieldTensor(Object nDArray) {
        super(ArrayUtils.nDArrayShape(nDArray),
                (T[]) new Field[ArrayUtils.nDArrayShape(nDArray).totalEntriesIntValueExact()]);
        ArrayUtils.nDFlatten(nDArray, shape, data, 0);
    }


    /**
     * Creates a tensor with the specified shape filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill tensor with.
     */
    public FieldTensor(Shape shape, T fillValue) {
        super(shape, (T[]) new Field[shape.totalEntries().intValueExact()]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooFieldTensor<T> makeLikeCooTensor(Shape shape, T[] entries, int[][] indices) {
        return new CooFieldTensor<>(shape, entries, indices);
    }



    /**
     * Constructs a tensor of the same type as this tensor with the given shape and data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given shape and data.
     */
    @Override
    public FieldTensor<T> makeLikeTensor(Shape shape, T[] entries) {
        return new FieldTensor<T>(shape, entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public FieldVector<T> toVector() {
        return new FieldVector<T>(this.data.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must have the same number of {@link ValidateParameters#ensureTotalEntriesEqual(Shape, Shape) total entries}.
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public FieldMatrix<T> toMatrix(Shape matShape) {
        ValidateParameters.ensureTotalEntriesEqual(shape, matShape);
        ValidateParameters.ensureRank(matShape, 2);

        return new FieldMatrix<T>(matShape, data.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public FieldMatrix<T> toMatrix() {
        FieldMatrix<T> mat;

        if(this.getRank()==2) {
            mat = new FieldMatrix<T>(this.shape, this.data.clone());
        } else {
            mat = new FieldMatrix<T>(1, this.data.length, this.data.clone());
        }

        return mat;
    }


    /**
     * <p>Computes the element-wise absolute value of this tensor.
     * <p>Note: the absolute value may not be defined for all fields.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Tensor abs() {
        double[] abs = new double[data.length];
        RingOps.abs(data, abs);
        return new Tensor(shape, abs);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link FieldTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldTensor<T> src2 = (FieldTensor<T>) object;

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
        int precision = PrintOptions.getPrecision();

        for(int i = 0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(data[i], precision);
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
        value = StringUtils.ValueOfRound(data[size-1], precision);
        width = padding + value.length();
        value = centering ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
