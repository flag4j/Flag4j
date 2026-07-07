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
import org.flag4jv3.ndarrays.base.NDArrayBase;
import org.flag4jv3.util.tuples.Pair;

// TODO NOW: DOCS
public abstract class DenseNDArrayBase<T extends DenseNDArrayBase<T, U, V>, U, V> extends NDArrayBase<T, U, V> {
    // TODO NOW: We need to make 100% sure that a user cannot accidently construct an array as a base when it is really a view.
    //  This means we should always copy data in ANY public facing constructors/factories.

    /// Indicates if this array is a base array (`true`) or a view (`false`).
    public final boolean isBase;
    public final Layout layout; // Layout is immutable, safe to be public.
    public final boolean isContiguous;

    protected final int[] strides; // Cached strides. Must not be modified or exposed externally.


    protected DenseNDArrayBase(U buffer, Layout layout, boolean isBase) {
        super(layout.shape, buffer);

        this.layout = layout;
        this.isBase = isBase;

        this.strides = layout.strides(); // Layout is immutable, cache strides here to avoid re-cloning.
        this.isContiguous = layout.isContiguous; // Just for convenience.
    }


    public abstract T makeLike(U dataBuffer, Layout layout, boolean isBase);


    public boolean isBase() {
        return isBase;
    }


    public boolean isView() {
        return !isBase;
    }


    public int offset() {
        return layout.offset();
    }


    public int[] strides() {
        return layout.strides(); // Use layout strides as they are immutable.
    }


    /// Returns a contiguous in memory array containing the same data as `this` array.
    ///
    /// @return If `this` array is already contiguous, then `this` array is returned. Otherwise, a new (non-view)
    ///                                 array with the same data as `this` array is returned.
    public abstract T contiguous();


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
    public Pair<T, T> broadcast(T other) {
        Pair<Layout, Layout> bInfo = Layout.broadcast(layout, other.layout);

        T a = makeLike(buffer, bInfo.first(), false);
        T b = makeLike(other.buffer, bInfo.second(), false);

        return new Pair<>(a, b);
    }
}
