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

import org.flag4jv3.ndarrays.AnyNDArray;
import org.flag4jv3.ndarrays.NDArrayMask;
import org.flag4jv3.ndarrays.NDArraySlice;
import org.flag4jv3.ndarrays.Shape;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Objects;

/// // todo now: DOCS.
///
/// @param <T> The type of the nD array (for fluent API).
/// @param <U> The type of the nD array's buffer that stores individual elements of the nD array. The elements
///                 of this type may be mutable, but the size *must not* be mutable. For example, standard Java arrays are valid but
///                 [java.util.ArrayList] are *not valid*.
/// @param <V> The type of an individual element of the buffer (i.e., a representation of what is stored in [U]).
///                 The buffer, of type [U], need not store this type exactly, but it must be able to convert any element it stores to
///                 this type.
public abstract class NDArrayBase<T extends NDArrayBase<T, U, V>, U, V> implements AnyNDArray {
    public final int rank;
    public final Shape shape; // Fully immutable so it can be public.
    protected final U buffer; // Elements likely mutable so force getting via bufferView() and bufferCopy().


    protected NDArrayBase(Shape shape, U buffer) {
        Objects.requireNonNull(shape, "Shape cannot be null.");
        Objects.requireNonNull(buffer, "nD Array buffer cannot be null.");

        this.shape = shape;
        this.buffer = buffer;
        this.rank = shape.rank;
    }


    public abstract T makeLike(Shape shape, U dataBuffer);


    /// Gets a reference to `this` object.
    ///
    /// @return A reference to `this` object.
    @SuppressWarnings("unchecked")
    protected final T self() {
        return (T) this;
    }


    @Override
    public Shape shape() {
        return shape;
    }


    public int getSize(int axis) {
        return shape.getSize(axis);
    }


    public abstract V get(int... index);

    public abstract V get(NDArrayMask mask);

    public abstract V get(NDArraySlice slice);

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
        int[] dims = new int[rank];
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
        return rank == other.rank;
    }


    public boolean numelEquals(NDArrayBase<?, ?, ?> other) {
        return numel().equals(other.numel());
    }


    @Override
    public abstract boolean equals(Object other);

    @Override
    public abstract int hashCode();
}
