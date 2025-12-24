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

package org.flag4j.arrays;

import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.util.ArrayMapper;
import org.flag4j.util.ArrayReducer;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;

import java.util.*;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * <p>An {@code ArrayMask} is a specialized n-dimensional array of boolean values that serves as a mask
 * for selecting, filtering, and operating on elements of other nD arrays.
 *
 * <p>Each element in the {@code ArrayMask} corresponds to a position in a target array, where a {@code true}
 * value indicates selection or inclusion, and {@code false} indicates exclusion.
 *
 * <p>This class provides full support for:
 * <ul>
 *     <li>Boolean masking and indexing.</li>
 *     <li>Logical operations ({@code and}, {@code or}, {@code xor}, {@code not}) both in-place and out-of-place.</li>
 *     <li>Reduction and filtering using predicates.</li>
 *     <li>Flattening, reshaping, and transposing of the mask.</li>
 * </ul>
 *
 * <p>Internally, data is stored as a {@link java.util.BitSet}.
 *
 * <p>{@code ArrayMask} is part of the Flag4j nD Array API and inherits from {@link AbstractNDArray}.
 */
public class ArrayMask extends AbstractNDArray<ArrayMask, BitSet, Boolean> {

    /**
     * The true (logical) length of the data backing this ArrayMask. This is exactly equal to
     * {@code this.shape.totalEntriesIntValueExact();}.
     */
    private final int dataLength;


    /**
     * Creates an {@code ArrayMask} with the specified data and shape.
     *
     * @param shape Shape of this nD array.
     * @param data Entries of this nD array.
     */
    public ArrayMask(Shape shape, BitSet data) {
        super(shape, data);
        dataLength = shape.totalEntriesIntValueExact();
    }


    /**
     * Creates an {@code ArrayMask} with the specified data and shape.
     *
     * @param shape Shape of this nD array.
     * @param data Entries of this nD array.
     */
    public ArrayMask(Shape shape, boolean[] data) {
        super(shape, new BitSet(data.length));
        dataLength = shape.totalEntriesIntValueExact();

        for(int i=0, size=data.length; i<size; i++)
            this.data.set(i, data[i]);
    }


    /**
     * Creates an nD array with the specified data and shape.
     *
     * @param shape Shape of this nD array.
     * @param data Entries of this nD array.
     */
    public ArrayMask(Shape shape, List<Boolean> data) {
        super(shape, new BitSet(data.size()));
        dataLength = shape.totalEntriesIntValueExact();

        for(int i=0, size=dataLength; i<size; i++)
            this.data.set(i, data.get(i));
    }


    /**
     * Constructs an {@code ArrayMask} from a set of indices marking {@code true} locations in the mask.
     * @param shape The shape of the {@code ArrayMask} to create.
     * @param indices The indices
     * @return
     */
    public static ArrayMask fromTrueIndices(Shape shape, int[] indices) {
        int newDataLength = shape.totalEntriesIntValueExact();
        BitSet bits = new BitSet(newDataLength);
        for (int i : indices) {
            if (i < 0 || i >= newDataLength)
                throw new IndexOutOfBoundsException("Index " + i + " is out of bounds for length " + newDataLength);

            bits.set(i);
        }
        return new ArrayMask(shape, bits);
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
        ValidateParameters.validateTensorIndices(shape, indices);
        return data.get(shape.unsafeGet1DIndex(indices));
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

        BitSet values = new BitSet();
        int count = 0;

        for (int i = mask.data.nextSetBit(0); i >= 0; i = mask.data.nextSetBit(i + 1))
            values.set(count++, this.data.get(i));

        return new ArrayMask(new Shape(count), values);
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
        ValidateParameters.validateTensorIndices(shape, indices);
        data.set(shape.get1DIndex(indices), value);
        return this;
    }


    /**
     * <p>Gets the flat 1D data index of the next {@code true}
     * element in this {@code ArrayMask}, on or after {@code fromIndex}.
     *
     * <p>To get the nD index of the next {@code true} element use {@link #nextSetElement(int...)}.
     *
     * @param fromIndex The 1D data index to start searching from.
     * @return The flat 1D data index of the next element, on or after {@code fromIndex}
     * that is set to {@code true} in this {@code ArrayMask}.
     *
     * @throws IndexOutOfBoundsException If {@code fromIndex} is not within the bounds of this {@code ArrayMask}.
     * @see #nextSetElement(int...)
     */
    public int nextSetElement1D(int fromIndex) {
        if(fromIndex < 0 || fromIndex >= dataLength) {
            throw new IndexOutOfBoundsException("Index " + fromIndex
                    + " is out of bounds for ArrayMask with dataLength=" + dataLength);
        }

        return data.nextSetBit(fromIndex);
    }


    /**
     * <p>Gets the nD index of the next {@code true} element in this {@code ArrayMask}, on or after {@code fromIndex}.
     * @param fromIndex The nD index to start searching from.
     * @return The nD index of the next {@code true} element in this {@code ArrayMask}, on or after {@code fromIndex}.
     * @throws IndexOutOfBoundsException If {@code fromIndex} is not within the bounds of this {@code ArrayMask}.
     */
    public int[] nextSetElement(int... fromIndex) {
        int flatIdx = data.nextSetBit(shape.get1DIndex(fromIndex));
        return shape.getNdIndices(flatIdx);
    }


    /**
     * <p>Gets the flat 1D data index of the next {@code false}
     * element in this {@code ArrayMask}, on or after {@code fromIndex}.
     *
     * <p>To get the nD index of the next {@code false} element use {@link #nextClearElement(int...)}.
     *
     * @param fromIndex The 1D data index to start searching from.
     * @return The flat 1D data index of the next element, on or after {@code fromIndex}
     * that is set to {@code false} in this {@code ArrayMask}.
     *
     * @throws IndexOutOfBoundsException If {@code fromIndex} is not within the bounds of this {@code ArrayMask}.
     * @see #nextClearElement(int...)
     */
    public int nextClearElement1D(int fromIndex) {
        if(fromIndex < 0 || fromIndex >= dataLength) {
            throw new IndexOutOfBoundsException("Index " + fromIndex
                    + " is out of bounds for ArrayMask with dataLength=" + dataLength);
        }

        return data.nextSetBit(fromIndex);
    }


    /**
     * <p>Gets the nD index of the next {@code false} element in this {@code ArrayMask}, on or after {@code fromIndex}.
     * @param fromIndex The nD index to start searching from.
     * @return The nD index of the next {@code false} element in this {@code ArrayMask}, on or after {@code fromIndex}.
     * @throws IndexOutOfBoundsException If {@code fromIndex} is not within the bounds of this {@code ArrayMask}.
     */
    public int[] nextClearElement(int... fromIndex) {
        int flatIdx = data.nextSetBit(shape.get1DIndex(fromIndex));
        return shape.getNdIndices(flatIdx);
    }


    /**
     * Gets the size of the 1D data object backing this nD array.
     *
     * @return The size of the 1D data object backing this nD array.
     * @see #cardinality()
     */
    @Override
    public int dataLength() {
        return dataLength;
    }


    /**
     * Gets the cardinality (i.e., number of {@code true} elements) in this array mask.
     * @return The cardinality in this array mask.
     */
    public int cardinality() {
        return data.cardinality();
    }


    /**
     * Gets a list of the indices where this {@code ArrayMask} is {@code true}.
     * @return A list of the indices where this {@code ArrayMask} is {@code true}.
     */
    public int[] getTrueIndices() {
        return data.stream().toArray();
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
        return new ArrayMask(new Shape(dataLength), (BitSet) data.clone());
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
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = new int[rank];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntriesIntValueExact();
        return new ArrayMask(new Shape(dims), (BitSet) data.clone());
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
        return new ArrayMask(newShape, (BitSet) data.clone());
    }


    /**
     * <p>Constructs an nD array of the same type as this nD array with the given {@code shape} and {@code data}.
     *
     * @param shape Shape of the nD array to construct.
     * @param data Entries of the nD array to construct.
     *
     * @return An nD array of the same type and with the same non-zero indices as this nD array with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public ArrayMask makeLikeNDArray(Shape shape, BitSet data) {
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

        BitSet dest = new BitSet(shape.totalEntries().intValue());

        Shape destShape = shape.swapAxes(axis1, axis2);
        int[] destIndices;

        for(int i=0, size=dataLength; i<size; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2);  // Compute destination indices.
            dest.set(destShape.get1DIndex(destIndices), data.get(i));  // Apply transpose for the element
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

        BitSet dest = new BitSet(shape.totalEntries().intValue());
        Shape destShape = shape.permuteAxes(axes);
        int[] destIndices;

        for(int i=0, size=dataLength; i<size; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.permute(destIndices, axes);  // Compute destination indices.
            dest.set(destShape.get1DIndex(destIndices), data.get(i));  // Apply transpose for the element.
        }

        return new ArrayMask(destShape, dest);
    }


    /**
     * Creates a copy of this nD array.
     * @return A copy of this nD array.
     */
    @Override
    public ArrayMask copy() {
        return new ArrayMask(shape, (BitSet) data.clone());
    }


    /**
     * Applies a map to each item in this nD array. This is done in place.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A reference to this nD array.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     */
    @Override
    public ArrayMask map(UnaryOperator<Boolean> mapper) {
        ArrayMapper.map(data, mapper);
        return this;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
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
    @Override
    public Boolean reduce(Boolean identity, BinaryOperator<Boolean> accumulator) {
        return ArrayReducer.reduce(data, identity, accumulator);
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
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
    @Override
    public ArrayMask reduce(Boolean identity, BinaryOperator<Boolean> accumulator, int... axes) {
        Pair<Shape, BitSet> reduced = ArrayReducer.reduce(shape, data, identity, accumulator, axes);
        return new ArrayMask(reduced.first(), reduced.second());
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
        BitSet dest = new BitSet(dataLength);

        for(int i=0, size=dataLength; i<size; i++)
            dest.set(i, predicate.apply(data.get(i)));

        return new ArrayMask(shape, dest);
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
        BitSet filtered = new BitSet();
        int count = 0;

        for (int i = 0; i < dataLength; i++) {
            boolean v = data.get(i);
            if (predicate.apply(v)) filtered.set(count++, v);
        }

        return new ArrayMask(new Shape(count), filtered);
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
        for(int i=0, size=data.length(); i<size; i++)
            if (predicate.apply(data.get(i))) return true;

        return false;
    }


    /**
     * Checks if any element of this {@code ArrayMask} is {@code true}.
     * @return {@code true} if any element of this mask is {@code true}.
     */
    public boolean any() {
        return data.cardinality() > 0;
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
        for(int i=0, size=dataLength; i<size; i++)
            if (!predicate.apply(data.get(i))) return false;

        return true;
    }


    /**
     * Checks if <em>all</em> elements of this {@code ArrayMask} are {@code true}.
     * @return {@code true} if <em>all</em> elements of this mask are {@code true}; otherwise {@code false}.
     */
    public boolean all() {
        return data.cardinality() == dataLength - 1;
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
    public int countTrue(Function<Boolean, Boolean> predicate) {
        int count = 0;

        for(int i=0, size=dataLength; i<size; i++)
            if (predicate.apply(data.get(i))) count++;

        return count;
    }


    /**
     * Counts the total number of {@code true} elements in this {@code ArrayMask}.
     * @return The total number of {@code true} elements in this {@code ArrayMask}.
     */
    public int countTrue() {
        return data.cardinality();
    }


    /**
     * Counts the total number of {@code false} elements in this {@code ArrayMask}.
     * @return The total number of {@code false} elements in this {@code ArrayMask}.
     */
    public int countFalse() {
        return dataLength - data.cardinality();
    }


    /**
     * Computes the element-wise logical NOT of this nD array. To do this operation in-place, use {@link #notEq()}.
     * @return An nD array with the same shape as this nD array containing the logical NOT of each element in this nD array.
     *
     * @see #notEq()
     */
    public ArrayMask not() {
        BitSet notData = ((BitSet) data.clone());
        notData.flip(0, dataLength);
        return new ArrayMask(shape, notData);
    }


    /**
     * Computes the element-wise logical NOT of this nD array in-place. To do this operation out-of-place, use {@link #not()}.
     * @return A reference to this nD array.
     * @see #not()
     */
    public ArrayMask notEq() {
        data.flip(0, dataLength);
        return this;
    }


    /**
     * <p>Computes the element-wise logical AND of this nD array with the specified {@code mask}.
     * <p>To do this operation in-place, use {@link #andEq(ArrayMask)}.
     * @param mask The mask to compute logical AND with. Must have the same shape as this nD array.
     * @return The element-wise logical AND of this nD array with the specified {@code mask}.
     * @throws ArrayShapeException If {@code mask} has a different shape than this nD array.
     * @see #andEq(ArrayMask)
     */
    public ArrayMask and(ArrayMask mask) {
        ValidateParameters.ensureEqualShape(shape, mask.shape);
        BitSet andData = ((BitSet) data.clone());
        andData.and(mask.data);
        return new ArrayMask(shape, andData);
    }


    /**
     * <p>Computes the element-wise logical AND of this nD array with the specified {@code mask} and stores the result in this nD array.
     * <p>To do this operation out-of-place, use {@link #and(ArrayMask)}.
     *
     * @param mask The mask to compute logical AND with. Must have the same shape as this nD array.
     * @return A reference to this nD array.
     * @throws ArrayShapeException If {@code mask} has a different shape than this nD array.
     * @see #and(ArrayMask)
     */
    public ArrayMask andEq(ArrayMask mask) {
        data.and(mask.data);
        return this;
    }


    /**
     * <p>Computes the element-wise logical OR of this nD array with the specified {@code mask}.
     * <p>To do this operation in-place, use {@link #orEq(ArrayMask)}.
     *
     * @param mask The mask to compute logical OR with. Must have the same shape as this nD array.
     * @return The element-wise logical OR of this nD array with the specified {@code mask}.
     * @throws ArrayShapeException If {@code mask} has a different shape than this nD array.
     * @see #orEq(ArrayMask)
     */
    public ArrayMask or(ArrayMask mask) {
        ValidateParameters.ensureEqualShape(shape, mask.shape);
        BitSet orData = ((BitSet) data.clone());
        orData.or(mask.data);
        return new ArrayMask(shape, orData);
    }


    /**
     * <p>Computes the element-wise logical OR of this nD array with the specified {@code mask} and stores the result in this nD array.
     * <p>To do this operation out-of-place, use {@link #or(ArrayMask)}.
     *
     * @param mask The mask to compute logical OR with. Must have the same shape as this nD array.
     * @return A reference to this nD array.
     * @throws ArrayShapeException If {@code mask} has a different shape than this nD array.
     * @see #or(ArrayMask)
     */
    public ArrayMask orEq(ArrayMask mask) {
        data.or(mask.data);
        return this;
    }


    /**
     * <p>Computes the element-wise logical XOR of this nD array with the specified {@code mask}.
     * <p>To do this operation in-place, use {@link #orEq(ArrayMask)}.
     *
     * @param mask The mask to compute logical XOR with. Must have the same shape as this nD array.
     * @return The element-wise logical XOR of this nD array with the specified {@code mask}.
     * @throws ArrayShapeException If {@code mask} has a different shape than this nD array.
     * @see #xorEq(ArrayMask)
     */
    public ArrayMask xor(ArrayMask mask) {
        ValidateParameters.ensureEqualShape(shape, mask.shape);
        BitSet xorData = ((BitSet) data.clone());
        xorData.xor(mask.data);
        return new ArrayMask(shape, xorData);
    }


    /**
     * <p>Computes the element-wise logical XOR of this nD array with the specified {@code mask} and stores the result in this nD array.
     * <p>To do this operation out-of-place, use {@link #or(ArrayMask)}.
     *
     * @param mask The mask to compute logical XOR with. Must have the same shape as this nD array.
     * @return A reference to this nD array.
     * @throws ArrayShapeException If {@code mask} has a different shape than this nD array.
     * @see #xor(ArrayMask)
     */
    public ArrayMask xorEq(ArrayMask mask) {
        data.xor(mask.data);
        return this;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ArrayMask other)) return false;
        return shape.equals(other.shape) && data.equals(other.data);
    }


    @Override
    public int hashCode() {
        return Objects.hash(shape, data);
    }


    /**
     * Converts this {@code ArrayMask} to a human-readable string.
     * Specifically, the nD shape of this {@code ArrayMask}, the cardinality (i.e., the number of {@code true} elements) of
     * this {@code ArrayMask}, and the indices of all {@code true} elements.
     *
     * @return A string representation of this {@code ArrayMask}.
     */
    @Override
    public String toString() {
        int size = shape.totalEntries().intValueExact();
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));

        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();

        List<Integer> indices = new ArrayList<>();
        for (int i = data.nextSetBit(0); i >= 0; i = data.nextSetBit(i + 1))
            indices.add(i);

        int[][] nDIndices = new int[indices.size()][rank];

        result.append("Cardinality: ").append(data.cardinality()).append("\n");
        result.append("True Indices: " +
                PrettyPrint.abbreviatedArray(nDIndices, PrintOptions.getMaxRows(), maxCols, padding, 20, centering));

        return result.toString();
    }
}
