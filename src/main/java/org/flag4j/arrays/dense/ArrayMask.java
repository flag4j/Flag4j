/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.linalg.ops.ArrayMapper;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;

public class ArrayMask extends AbstractNDArray<ArrayMask, boolean[], Boolean> {

    // TODO: Investigate the possibility of packing bits into long[] for large masks.

    /**
     * Creates an nD array with the specified data and shape.
     *
     * @param shape Shape of this nD array.
     * @param data Entries of this nD array.
     * <ul>
     *     <li>If this nD array is dense, this specifies <em>all</em> data within this nD array.</li>
     *     <li>If this nD array is sparse, this specifies <em>only</em> the non-zero data of this nD array.</li>
     * </ul>
     */
    public ArrayMask(Shape shape, boolean[] data) {
        super(shape, data);
    }


    /**
     * Creates an nD array with the specified data and shape.
     *
     * @param shape Shape of this nD array.
     * @param data Entries of this nD array.
     * <ul>
     *     <li>If this nD array is dense, this specifies <em>all</em> data within this nD array.</li>
     *     <li>If this nD array is sparse, this specifies <em>only</em> the non-zero data of this nD array.</li>
     * </ul>
     */
    public ArrayMask(Shape shape, List<Boolean> data) {
        super(shape, new boolean[data.size()]);

        for(int i=0, size=this.data.length; i<size; i++)
            this.data[i] = data.get(i);
    }


    /**
     * Gets the element of this nD array at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this nD array at the specified indices.
     *
     * @throws IndexOutOfBoundsException If any indices are not within this nD array.
     */
    @Override
    public Boolean get(int... indices) {
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
    public ArrayMask get(ArrayMask mask) {
        ValidateParameters.ensureEqualShape(shape, mask.shape);
        ArrayList<Boolean> values = new ArrayList<>();

        for(int i=0, size=data.length; i<size; i++)
            if(mask.data[i]) values.add(data[i]);

        return new ArrayMask(new Shape(values.size()), values);
    }


    /**
     * Sets the element of this nD array at the specified indices. This is done in place.
     *
     * @param value New value to set the specified index of this nD array to.
     * @param indices Indices of the element to set.
     *
     * @return A reference to this nD array is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this nD array.
     */
    @Override
    public ArrayMask set(Boolean value, int... indices) {
        data[shape.get1DIndex(indices)] = value;
        return this;
    }


    /**
     * Gets the size of the 1D data object backing this nD array.
     *
     * @return The size of the 1D data object backing this nD array.
     */
    @Override
    public int dataLength() {
        return data.length;
    }


    /**
     * <p>Flattens this nD array to a single dimension.
     * To preserve the rank of the nD array but flatten to single axes, use {@link #flatten(int)}.
     * <p>This preserves the order of the entries in the {@link #data} object.
     *
     * @return The flattened nD array.
     *
     * @see #flatten(int)
     */
    @Override
    public ArrayMask flatten() {
        return new ArrayMask(new Shape(data.length), data.clone());
    }


    /**
     * <p>Flattens an nD array along the specified axis. Unlike {@link #flatten()}, this method preserves the rank of the nD array.
     * <p>This preserves the order of the entries in the {@link #data} object.
     *
     * @param axis Axis along which to flatten nD array.
     *
     * @throws IndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()} - 1}.
     * @see #flatten()
     */
    @Override
    public ArrayMask flatten(int axis) {
        // TODO: I feel the wording of the javadoc is a bit confusing. It flattens to an axis not along.
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = new int[rank];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntriesIntValueExact();
        return new ArrayMask(new Shape(dims), data.clone());
    }


    /**
     * Copies and reshapes this nD array.
     *
     * @param newShape New shape for the nD array.
     *
     * @return A copy of this nD array with the new shape.
     *
     * @throws ArrayShapeException If {@code newShape} does not have the same total number of entries as {@link #shape this.shape}.
     * @see #reshape(int...)
     */
    @Override
    public ArrayMask reshape(Shape newShape) {
        return new ArrayMask(newShape, data.clone());
    }


    /**
     * <p>Constructs an nD array of the same type as this nD array with the given {@code shape} and {@code data}.
     * <p>If this nD array is sparse, the resulting nD array will also have the same non-zero indices as this nD array.
     *
     * @param shape Shape of the nD array to construct.
     * @param data Entries of the nD array to construct.
     *
     * @return An nD array of the same type and with the same non-zero indices as this nD array with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public ArrayMask makeLikeNDArray(Shape shape, boolean[] data) {
        return new ArrayMask(shape, data);
    }


    /**
     * Computes the transpose of a nD array by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this nD array along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this nD array.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public ArrayMask T(int axis1, int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        boolean[] dest = new boolean[shape.totalEntries().intValue()];
        Shape destShape = shape.swapAxes(axis1, axis2);
        int[] destIndices;

        for(int i=0; i<data.length; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
            dest[destShape.get1DIndex(destIndices)] = data[i]; // Apply transpose for the element
        }

        return new ArrayMask(destShape, dest);
    }


    /**
     * Computes the transpose of this nD array.
     * That is, permutes the axes of this nD array so that it matches the permutation specified by {@code axes}.
     *
     * @param axes Permutation of nD array axis.
     * If the nD array has rank {@code n}, then this must be an array of length {@code n}
     * which is a permutation of {@code {0, 1, 2, ..., n-1}}.
     *
     * @return The transpose of this nD array with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this nD array.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public ArrayMask T(int... axes) {
        ValidateParameters.ensurePermutation(axes);
        ValidateParameters.ensureAllEqual(shape.getRank(), axes.length);
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        boolean[] dest = new boolean[shape.totalEntries().intValue()];
        Shape destShape = shape.permuteAxes(axes);
        int[] destIndices;

        for(int i=0; i<data.length; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.permute(destIndices, axes); // Compute destination indices.
            dest[destShape.get1DIndex(destIndices)] = data[i]; // Apply transpose for the element
        }

        return new ArrayMask(destShape, dest);
    }


    /**
     * Creates a copy of this nD array.
     * @return A copy of this nD array.
     */
    @Override
    public ArrayMask copy() {
        return new ArrayMask(shape, data.clone());
    }


    /**
     * Applies a map to each item in this nD array.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A new nD array with the same shape as this nD array containing the mapped values.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     */
    @Override
    public ArrayMask map(UnaryOperator<Boolean> mapper) {
        // TODO: Need to mention in docs that this is done in place.
        //      Also this documentation should change in the AbstractNDArray class as well.
        ArrayMapper.map(data, mapper);
        return this;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will only be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction.
     * @param accumulator A binary operator that combines the current accumulated
     * result with the next array element and returns the updated result.
     *
     * @return The final accumulated scalar of type {@code V}. If this nD array is empty, {@code identity} will be returned.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator, int...)
     */
    @Override
    public Boolean reduce(Boolean identity, BinaryOperator<Boolean> accumulator) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will only be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction.
     * @param accumulator A binary operator that combines the current accumulated
     * result with the next array element and returns the updated result.
     * @param axes The axes along which reduce this nD array.
     *
     * @return An nD array of the same shape as this nD array but with the specified {@code axes} removed.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator)
     */
    @Override
    public ArrayMask reduce(Boolean identity, BinaryOperator<Boolean> accumulator, int... axes) {
        // TODO: Implement this method
        return null;
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
    public ArrayMask where(Function<Boolean, Boolean> predicate) {
        // TODO: Implement this method
        return null;
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
    public ArrayMask filter(Function<Boolean, Boolean> predicate) {
        // TODO: Implement this method
        return null;
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
    public boolean any(Function<Boolean, Boolean> predicate) {
        // TODO: Implement this method
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
    public boolean all(Function<Boolean, Boolean> predicate) {
        // TODO: Implement this method
        return false;
    }
}
