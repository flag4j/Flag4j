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

package org.flag4jv3.ndarrays;


import org.flag4jv3.ndarrays.base.NDArrayBase;
import org.flag4jv3.ndarrays.dense.DenseNDArrayBase;

import java.math.BigInteger;

// TODO NOW: Implementation + DOCS. This is a specialized implementation of a boolean array used for masking.
//  We actually don't want a whole other mask object. Just use a DenseBoolNDArray as the mask object.
//  This should be implemented as a `byte[]` array rather than a `boolean[]` array.

public class Mask extends DenseNDArrayBase<Mask, byte[], Boolean> {

    protected Mask(byte[] buffer, Layout layout, Mask base) {
        super(buffer, layout, base);
    }


    /// Constructs a new dense nD-array of the same type as this nD-array.
    ///
    /// @param dataBuffer The backing data buffer of the nD-array.
    /// @param layout The layout of the nD-array in memory.
    /// @param base The base nD-array. Unlike with [DenseNDArrayBase], this *can not* be `null`.
    @Override
    protected Mask makeLike(byte[] dataBuffer, Layout layout, Mask base) {
        // TODO: Implement this method
        return null;
    }


    /// Constructs a new dense nD-array of the same type as this nD-array.
    ///
    /// Note that this method differs from [#makeLike(Shape, Object)] in that
    /// `dataBuffer`'s size *need not* be equal to `layout.shape().numelIntValueExact()`.
    ///
    /// @param layout The layout of the nD-array in memory.
    /// @param dataBuffer The buffer contains the data for the new nD-array.
    /// @return A new dense nD-array of the same type as this nD-array. The returned nD-array will be a [base array][#isBase()] and be
    /// [contiguous][#isContiguous()].
    ///
    /// @implSpec `dataBuffer` *must* be deeply copied to guarantee that the new nD-array does not share memory with any other.
    /// @see #asContiguous()
    @Override
    public Mask makeLike(Layout layout, byte[] dataBuffer) {
        // TODO: Implement this method
        return null;
    }


    /// Returns a contiguous in memory array containing the same data as `this` array.
    ///
    /// @return If `this` array is already contiguous, then `this` array is returned. Otherwise, a new (non-view)
    ///                                 array with the same data as `this` array is returned.
    ///
    /// @see #isContiguous()
    @Override
    public Mask asContiguous() {
        // TODO: Implement this method
        return null;
    }


    /// Constructs an nD-array that is the same type as `this` nD-array.
    ///
    /// @param shape Shape of the nD-array.
    /// @param dataBuffer The dataBuffer of the nD-array.
    /// @return A new nD-array of the same type as `this` nD-array.
    ///
    /// @implSpec `dataBuffer` must be deeply copied to guarantee the resulting nD-array *does not* share
    /// memory with any other nD-array.
    @Override
    public Mask makeLike(Shape shape, byte[] dataBuffer) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Boolean get(int... index) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Boolean get(Mask mask) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Boolean get(Slice... slices) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Boolean getFromBuffer(int index) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public byte[] bufferCopy() {
        // TODO: Implement this method
        return new byte[0];
    }


    @Override
    public BigInteger bufferSize() {
        // TODO: Implement this method
        return null;
    }


    @Override
    public NDArrayBase<?, byte[], Boolean> flatten() {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Mask reshape(Shape newShape) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Mask broadcastTo(Shape newShape) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Mask T() {
        // TODO: Implement this method
        return null;
    }


    @Override
    public Mask T(int... axes) {
        // TODO: Implement this method
        return null;
    }


    @Override
    public boolean equals(Object other) {
        // TODO: Implement this method
        return false;
    }


    @Override
    public int hashCode() {
        // TODO: Implement this method
        return 0;
    }


    @Override
    public StorageDescriptor storage() {
        // TODO: Implement this method
        return null;
    }
}
