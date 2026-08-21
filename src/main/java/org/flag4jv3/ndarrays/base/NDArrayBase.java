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

package org.flag4jv3.ndarrays.base;

import org.flag4jv3.ndarrays.*;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Objects;

// todo now: DOCS.

/// @param <T> The type of the nD-array (for fluent API).
/// @param <U> The type of the nD-array's buffer that stores individual elements of the nD-array. The elements
/// of this type may be mutable, but the size *must not* be mutable. For example, standard Java arrays are valid but
/// [java.util.ArrayList]'s are *not valid*. In addition to this, the buffer type must be "deeply" copyable.
/// @param <V> The type of an individual element of the buffer (i.e., a representation of what is stored in [U]).
/// The buffer, of type [U], need not store this type exactly, but it must be able to convert any element it stores to
/// this type.
public abstract class NDArrayBase<T extends NDArrayBase<T, U, V>, U, V> implements AnyNDArray {

    /// The rank of this nD-array (i.e., the number of dimensions). This specifies the minimum number of indices
    /// required to uniquely specify an individual element of the nD-array.
    public final int rank;

    /// The shape of this nD-array. This is a tuple of <span class="latex-inline">n</span> non-negative integers
    /// that specify the size of each dimension within this nD-array.
    public final Shape shape; // Fully immutable so it can be public.

    /// The data buffer containing the elements of this nD-array. This *may* be shared between multiple nD-array instances.
    protected final U buffer; // Elements likely mutable so force getting via bufferView() and bufferCopy().

    /// The size (in [#buffer] indices) of an individual item in the nD-array.
    /// If this is `2`, then a single item would occupy two successive positions in the [#buffer].
    public final int itemSize;

    /// The storage descriptor of this nD-array.
    private final StorageDescriptor storageDescriptor;


    protected NDArrayBase(Shape shape, int itemSize, U buffer, StorageDescriptor storageDescriptor) {
        Objects.requireNonNull(shape, "Shape cannot be null.");
        Objects.requireNonNull(buffer, "nD Array buffer cannot be null.");

        this.shape = shape;
        this.rank = shape.rank();
        this.itemSize = itemSize;
        this.buffer = buffer;
        this.storageDescriptor = storageDescriptor;
    }


    /// Constructs an nD-array that is the same type as `this` nD-array.
    ///
    /// @param shape Shape of the nD-array.
    /// @param dataBuffer The dataBuffer of the nD-array.
    /// @return A new nD-array of the same type as `this` nD-array.
    ///
    /// @implSpec `dataBuffer` must be deeply copied to guarantee the resulting nD-array *does not* share
    /// memory with any other nD-array.
    public abstract T makeLike(Shape shape, U dataBuffer);


    /// Gets a reference to `this` object.
    ///
    /// @return A reference to `this` object.
    @SuppressWarnings("unchecked")
    protected final T self() {
        return (T) this;
    }


    // TODO NOW: Do we want public shape *and* a getter (similarly for rank)? Seems confusing. I feel it should be one or the other.
    //  always having to do `.shape()` feels annoying but consistent? Not too sure. I think we need the getters for
    //  interface reasons.
    @Override
    public Shape shape() {
        return shape;
    }


    @Override
    public int rank() {
        return shape.rank();
    }


    public int getSize(int axis) {
        return shape.getSize(axis);
    }


    public abstract V get(int... index);

    public abstract V get(Mask mask);

    public abstract V get(Slice... slices);


    public final V slice(Slice... slices) {
        return get(slices); // Just an alias for get(slices).
    }


    public abstract V getFromBuffer(int index);


    public abstract U bufferCopy();


    public U bufferView() {
        return buffer;
    }


    public abstract BigInteger bufferSize();


    public BigInteger numel() {
        return shape.numel();
    }


    public T copy() {
        return makeLike(shape, bufferCopy());
    }


    public abstract NDArrayBase<?, U, V> flatten();


    public T flatten(int axis) {
        int[] dims = new int[shape.rank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.numel().intValueExact();
        return reshape(new Shape(dims));
    }


    public abstract T reshape(Shape newShape);

    public abstract T broadcastTo(Shape newShape);

    public abstract T T();

    public abstract T T(int... axes);


    public T squeeze() {
        Shape squeezedShape = shape.squeeze();
        if (squeezedShape == shape) return self();
        return makeLike(squeezedShape, bufferCopy());
    }


    public T squeeze(int axis) {
        Shape squeezedShape = shape.squeeze(axis);
        if (squeezedShape == shape) return self();
        return makeLike(squeezedShape, bufferCopy());
    }


    public T squeeze(int... axes) {
        Shape squeezedShape = shape.squeeze(axes);
        if (squeezedShape == shape) return self();
        return makeLike(squeezedShape, bufferCopy());
    }


    public boolean shapeEquals(NDArrayBase<?, ?, ?> other) {
        return shape.equals(other.shape);
    }


    public boolean rankEquals(NDArrayBase<?, ?, ?> other) {
        return shape.rank() == other.shape.rank();
    }


    public boolean numelEquals(NDArrayBase<?, ?, ?> other) {
        return numel().equals(other.numel());
    }


    @Override
    public abstract boolean equals(Object other);

    @Override
    public abstract int hashCode();
}
