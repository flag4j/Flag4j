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

package org.flag4jv3.ndarrays.dense;

import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.Shape;
import org.flag4jv3.ndarrays.StorageDescriptor;
import org.flag4jv3.ndarrays.base.NDArrayBase;
import org.flag4jv3.util.tuples.Pair;

// TODO NOW: DOCS


/// @param <T> The type of the dense nD-array. Used for fluent API.
/// @param <U> The type of the nD-array's buffer that stores individual elements of the nD-array.
/// The elements of this type may be mutable, but the size must not be mutable. For example, standard Java arrays are
/// valid but [java.util.ArrayList]'s are not valid. In addition to this, the buffer type must be "deeply" copyable. In addition to this,
/// two buffer objects *may not* share memory unless they are the *same* object. See implementation requirements for more.
/// @param <V> The type of an individual element of the buffer (i.e., a representation of what is stored in [U]).
/// /// The buffer, of type [U], need not store this type exactly, but it must be able to convert any element it stores to
/// /// this type.
/// @implSpec The [#buffer] objects implementations of this class must not share memory unless they are the *same* object.
/// In particular, if `buffer1 != buffer2`, modifications to one buffer *must not* affect the contents of the other. Methods such
/// as [#mayShareMemory(org.flag4jv3.ndarrays.dense.DenseNDArrayBase)] rely on this assumption holding.
public abstract class DenseNDArrayBase<T extends DenseNDArrayBase<T, U, V>, U, V> extends NDArrayBase<T, U, V> {
    /// The layout of this nD-array in memory.
    final Layout layout;

    /// The strides of this nD-array in memory.
    final int[] strides; // Cached strides from layout. Absolutely must not be modified or exposed publicly.

    /// The base array of this nD-array. If this nD-array is a base, then this will be a reference to `this`.
    final T base;


    protected DenseNDArrayBase(U buffer, Layout layout, T base) {
        super(layout.shape(), layout.itemSize(), buffer, StorageDescriptor.Dense);
        this.layout = layout;
        this.strides = layout.strides(); // Layout is immutable, cache strides here to avoid re-cloning.
        this.base = (base == null) ? self() : base;
    }


    /// Constructs a new dense nD-array of the same type as this nD-array.
    ///
    /// @param dataBuffer The backing data buffer of the nD-array.
    /// @param layout The layout of the nD-array in memory.
    /// @param base The base nD-array. Unlike with [DenseNDArrayBase], this *can not* be `null`.
    protected abstract T makeLike(U dataBuffer, Layout layout, T base);


    /// Constructs a new dense nD-array of the same type as this nD-array.
    ///
    /// Note that this method differs from [#makeLike(org.flag4jv3.ndarrays.Shape, java.lang.Object)] in that
    /// `dataBuffer`'s size *need not* be equal to `layout.shape().numelIntValueExact()`.
    ///
    /// @param layout The layout of the nD-array in memory.
    /// @param dataBuffer The buffer contains the data for the new nD-array.
    /// @return A new dense nD-array of the same type as this nD-array. The returned nD-array will be a [base array][#isBase()] and be
    /// [contiguous][#isContiguous()].
    ///
    /// @implSpec `dataBuffer` *must* be deeply copied to guarantee that the new nD-array does not share memory with any other.
    /// @see #asContiguous()
    public abstract T makeLike(Layout layout, U dataBuffer);


    /// Checks if this nD-array is contiguous in memory.
    ///
    /// @return `true` if this nD-array is contiguous in memory; otherwise, `false`.
    ///
    /// @see #isCContiguous()
    /// @see #isFContiguous()
    public boolean isContiguous() {
        return layout.isContiguous();
    }


    /// Checks if this nD-array is [C-contiguous][org.flag4jv3.ndarrays.ContiguousOrder#C] in memory.
    ///
    /// @return `true` if this nD-array is [C-contiguous][org.flag4jv3.ndarrays.ContiguousOrder#C] in memory; otherwise, `false`.
    ///
    /// @see #isCContiguous()
    /// @see #isFContiguous()
    public boolean isCContiguous() {
        return layout.isCContiguous();
    }


    /// Checks if this nD-array is [F-contiguous][org.flag4jv3.ndarrays.ContiguousOrder#F] in memory.
    ///
    /// @return `true` if this nD-array is [F-contiguous][org.flag4jv3.ndarrays.ContiguousOrder#F] in memory; otherwise, `false`.
    ///
    /// @see #isCContiguous()
    /// @see #isFContiguous()
    public boolean isFContiguous() {
        return layout.isFContiguous();
    }


    /// Checks if this nD-array is a base array or a view.
    ///
    /// @return `true` if this nD-array is a base array; otherwise, `false`.
    ///
    /// @see #base
    /// @see #isView()
    public boolean isBase() {
        return base == this;
    }


    /// Check if this nD-array is a view (i.e., shares memory with another array but it *not* the base array).
    ///
    /// @return `true` if this nD-array is a view; otherwise, `false`.
    ///
    /// @see #isBase
    public boolean isView() {
        return !isBase();
    }


    /// Gets the initial offset into the backing [data buffer][#bufferCopy()] of this nD-array.
    /// For [base][#isBase()] arrays, this is always `0`.
    ///
    /// @return The initial offset into the backing data [data buffer][#bufferCopy()] of this nD-array.
    public int offset() {
        return layout().offset();
    }


    /// Gets the strides of this nD-array.
    ///
    /// @return The strides of this array.
    public int[] strides() {
        /* Use layout strides as they are immutable. Retuning `this.strides` would publicly
         leak the reference to this array's strides */
        return layout().strides();
    }


    /// Returns a contiguous in memory array containing the same data as `this` array.
    ///
    /// @return If `this` array is already contiguous, then `this` array is returned. Otherwise, a new (non-view)
    ///                                 array with the same data as `this` array is returned.
    ///
    /// @see #isContiguous()
    public abstract T asContiguous();


    @Override
    public T squeeze() {
        Shape squeezedShape = shape.squeeze();
        if (squeezedShape == shape) return self();
        return makeLike(squeezedShape, bufferCopy());
    }


    @Override
    public T squeeze(int axis) {
        Shape squeezedShape = shape.squeeze(axis);
        if (squeezedShape == shape) return self();
        return makeLike(squeezedShape, bufferCopy());
    }


    @Override
    public T squeeze(int... axes) {
        Shape squeezedShape = shape.squeeze(axes);
        if (squeezedShape == shape) return self();
        return makeLike(squeezedShape, bufferCopy());
    }


    // TODO NOW: DOCS: note that the returned arrays are always views.


    /// Broadcasts two dense nD-arrays together.
    ///
    /// Two dense nD-arrays can be broadcast together if starting from the right-most dimension of each array's [shape][#shape()],
    /// 1. they are [equal][Shape#equals(Object)], or
    /// 2. one of them is `1`.
    ///
    /// @param other The other dense nD-array to be broadcast with `this` nD-array.
    /// @return A [pair][Pair] of [dense nD-array's][DenseNDArrayBase] containing the broadcasted result of `this` and `other`.
    ///
    /// @throws IllegalArgumentException If `this` and `other` cannot be broadcast together.
    public Pair<T, T> broadcast(T other) {
        // Layout.boradcast will throw exception if layouts cannot be broadcast.
        Pair<Layout, Layout> bInfo = Layout.broadcast(layout, other.layout);

        T a = makeLike(buffer, bInfo.first(), base);
        T b = makeLike(other.buffer, bInfo.second(), base);

        return new Pair<>(a, b);
    }


    /// Gets the [layout][Layout] of this nD-array.
    ///
    /// @return The [layout][Layout] of this nD-array.
    public Layout layout() {
        return layout;
    }


    // TODO NOW: convert this to be `mayShareMemory`


    /// Checks if two dense nD-arrays share memory. Two dense nD-arrays share memory if and only if they have the
    /// same [buffer][#bufferView()].
    ///
    /// @param other The other dense nD-array to compare with.
    /// @return `true` if `this` and `other` share memory; otherwise, `false`.
    public boolean mayShareMemory(T other) {
        if (buffer != other.buffer) {
            return false; // Note: this makes an assumption that two buffer objects cannot share memory (e.g., java arrays).
        } else {
            // TODO NOW:
            return false;
        }
    }
}
