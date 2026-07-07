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

package org.flag4j.arrays.dense;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldVector;
import org.flag4j.arrays.sparse.CooFieldVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;

/**
 * <p>Instances of this class represent a dense vector backed by a {@link FieldElement} array.
 * The {@code FieldVector} class provides functionality for matrix operations whose elements are members of a field, supporting
 * mutable items with a fixed shape.
 *
 * <p>A {@code FieldVector} is essentially equivalent to a rank-1 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general rank n tensors.
 *
 * <p><b>Key Features:</b>
 * <ul>
 *   <li>Support for standard vector operations like addition, subtraction, and inner/outer products.</li>
 *   <li>Conversion methods to other representations, such as {@link FieldMatrix}, {@link FieldTensor}, or COO (Coordinate).</li>
 *   <li>Utility methods for checking properties like being the zero vector.</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Constructing a complex matrix from an array of complex numbers
 * Complex128[] complexData = {
 *     new Complex128(1, 2), new Complex128(3, 4),
 *     new Complex128(5, 6), new Complex128(7, 8)
 * };
 * FieldVector<Complex128> vector = new FieldVector(complexData);
 *
 * // Performing vector inner/outer product.
 * Complex128 inner = vector.inner(vector);
 * FieldMatrix<Complex128> outer = vector.outer(vector);
 *
 * // Checking if the vector only contains zeros.
 * boolean isZero = vector.isZeros();
 * }</pre>
 *
 * @param <T> Type of the {@link FieldElement field} element for the matrix.
 *
 * @see FieldVector
 * @see FieldTensor
 * @see AbstractDenseFieldMatrix
 * @see CVector
 */
public class FieldVector<T extends FieldElement<T>> extends AbstractDenseFieldVector<FieldVector<T>, FieldMatrix<T>, T> {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a vector with the specified items.
     *
     * @param entries Entries of this vector.
     */
    public FieldVector(T... entries) {
        super(new Shape(entries.length), entries);
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param size
     * @param fillValue Value to fill this vector with.
     */
    public FieldVector(int size, T fillValue) {
        super(new Shape(size), (T[]) new FieldElement[size]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Constructs a dense complex vector with the given shape and entries.
     * @param shape The shape of the vector. Must be rank-1 and satisfy {@code shape.totalEntriesIntValueExact() == items.length}.
     * @param data The entries of the vector.
     * @throws LinearAlgebraException If {@code shape.getRank() != 1}
     * @throws IllegalArgumentException If {@code shape.totalEntriesIntValueExact() != items.length}
     */
    public FieldVector(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a vector with the specified {@code items}.
     *
     * @param entries Entries of this vector.
     */
    @Override
    public FieldVector<T> makeLikeTensor(T... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Constructs a matrix of a similar type to this vector with the specified {@code shape} and {@code items}.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of a similar type to this vector with the specified {@code shape} and {@code items}.
     */
    @Override
    public FieldMatrix<T> makeLikeMatrix(Shape shape, T[] entries) {
        return new FieldMatrix<>(shape, entries);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given shape and items.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given shape and items.
     */
    @Override
    public FieldVector<T> makeLikeNDArray(Shape shape, T[] data) {
        ValidateParameters.ensureAllEqual(shape.numelIntValueExact(), data.length);
        ValidateParameters.ensureRank(shape, 1);
        return new FieldVector<T>(data);
    }


    /**
     * Constructs a sparse COO tensor, which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero items of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooFieldVector<T> makeLikeCooTensor(Shape shape, T[] entries, int[][] indices) {
        return new CooFieldVector<>(shape.numelIntValueExact(), entries, indices[0]);
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor equivalent to this vector.
     */
    public FieldTensor<T> toTensor() {
        return new FieldTensor(shape, data.clone());
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @param shape The desired shape of the resulting tensor.
     * @return A tensor with the specified {@code shape} containing the entries of this vector.
     */
    public FieldTensor<T> toTensor(Shape shape) {
        return new FieldTensor(shape, data.clone());
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link FieldVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldVector<T> src2 = (FieldVector<T>) object;

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
     * Converts this vector to a human-readable string format. To specify the maximum number of items to print, use
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
