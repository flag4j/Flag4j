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

package org.flag4j.arrays_new.backend;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_new.NDArraySlice;
import org.flag4j.util.ValidateNDArray;
import org.flag4j.util.ValidateParameters;

public abstract class AbstractDenseNDArray<T extends AbstractDenseNDArray<T, U, V>, U, V>
        extends AbstractNDArray<T, U, V> {

    /**
     * If {@code true}, this nD array is a base array. If {@code false}, this nD array is a view.
     */
    protected final boolean isBase;

    /**
     * Strides of this nD array. If this is null, then this nD array is not strided meaning it is contiguous and packed.
     * If this is not null, then this nD array is strided, meaning it is not contiguous and packed.
     */
    protected int[] strides = null;

    /**
     * If strides are not {@code null}, then this is the offset of the first element of this nD array.
     */
    protected int offset = 0;


    /**
     * Creates a dense nD array with the specified dataBuffer and shape.
     *
     * @param shape Shape of this nD array. Must be non-null.
     * @param dataBuffer The 1D data buffer of this nD array. Must be non-null.
     * If <em>isBase</em> is {@code false}, then this buffer <em>must not</em> be shared with any other nD array.
     * This is not strictly enforced so callers must be <em>extreemly</em> careful.
     * @param offset Offset of the first element of this nD array. This must be {@code 0} if {@code strides} is {@code null}.
     * @param strides Strides of this nD array. Can be <code>null</code> if this nD array is contiguous.
     * @param isBase If {@code true}, this nD array is a base array. If {@code false}, this nD array is a view.
     */
    protected AbstractDenseNDArray(Shape shape, U dataBuffer, int offset, int[] strides, boolean isBase) {
        super(shape, dataBuffer);

        this.isBase = isBase;

        ValidateParameters.ensureNonNegative(offset);
        if (strides == null && offset != 0) {
            throw new IllegalArgumentException("Offset must be 0 if strides is null.");
        }

        if (strides != null) {
            this.strides = strides;
            this.offset = offset;

            ValidateNDArray.ensureValidLayout(shape, getSize(dataBuffer), offset, strides);
        } else {
            ValidateNDArray.ensureValidLayout(shape, getSize(dataBuffer));
        }
    }


    abstract T makeLikeNDArray(Shape shape, U buffer, int offset, int[] strides, boolean isBase);


    @Override
    T makeLikeNDArray(Shape shape, U buffer) {
        // Force the data to be copied since this constructs a contiguous layout.
        return makeLikeNDArray(shape, copyBuffer(buffer), 0, null, true);
    }


    /**
     * Checks if this nD array is strided.
     * @return {@code true} if this nD array is strided; otherwise {@code false}.
     * @see #getStridesView()
     * @see #getStridesCopy()
     * @see #isContiguous()
     */
    public boolean isStrided() {
        return strides != null;
    }


    /**
     * Checks if this nD array is contiguous in memory.
     * @return {@code true} if this nD array is contiguous in memory; otherwise {@code false} (i.e., strided in memory).
     * @see #getStridesView()
     * @see #isStrided()
     * @see #getStridesCopy()
     */
    public boolean isContiguous() {
        return strides == null;
    }


    /**
     * <p>Gets a copy of the strides of this nD array.
     *
     * @return A copy of the strides of this nD array if it is strided; otherwise {@code null}.
     * Modifying this copy <em>will not</em> affect the internal strides of this nD array.
     * @see #getStridesView()
     * @see #isStrided()
     * @see #isContiguous()
     */
    public int[] getStridesCopy() {
        if (strides == null) return null;
        else return strides.clone();
    }


    /**
     * <p>Gets a view of the strides of this nD array. Modifying the view <em>will</em> affect the strides of this nD array.
     *
     * <p><strong>Warning</strong>: This is generally intended for internal use only. Only modify if you absolutely know
     * what you are doing.</p>
     * @return A view of the strides of this nD array if it is strided; otherwise {@code null}.
     * Modifying this view <em>will</em> affect the internal strides of this nD array.
     * @see #getStridesCopy()
     * @see #isStrided()
     * @see #isContiguous()
     */
    public int[] getStridesView() {
        return strides;
    }


    /**
     * <p>Converts this nD array to a contiguous array if it is not already contiguous.</p>
     *
     * <p>This method is similar to {@link #copy()} except in the case where the array is contiguous. In that case,
     * a refrence to <em>this</em> array is returend rather than a copy.</p>
     *
     * @return If this nD array is contiguous, then this nD array is returned. Otherwise, a contiguous copy
     * is constructed and returned.
     */
    public abstract T asContiguous();


    /**
     * Checks if this nD array is a base array (i.e., it is <em>not</em> a view).
     * @return {@code true} if this nD array is a base array; otherwise {@code false}.
     * @see #isView()
     */
    public boolean isBase() {
        return isBase;
    }


    /**
     * Checks if this nD array is a view (i.e., it shares data with another nD array).
     * @return {@code true} if this nD array is a view; otherwise {@code false}.
     * @see #isBase()
     */
    public boolean isView() {
        return !isBase;
    }

    /**
     * Gets a view of this nD array. Manipulating the view <em>will</em> affect the original array.
     * Use {@link #copy()} to create a copy of this nD array.
     * @return A view of this nD array.
     *
     * @see #view()
     * @see #view(NDArraySlice)
     * @see #copy()
     */
    public T view() {
        return makeLikeNDArray(shape, dataBuffer, offset, strides, false);
    }


    /**
     * Gets a view of this nD array. Manipulating the view <em>will</em> affect the original array.
     * Use {@see #copy()} to create a copy of this nD array.
     * @param axes The axes to extract from this nD array. If not all axes are specified or reordered,
     * then the resulting view will be strided.
     * @return A view of this nD array extracted from the specified {@code axes}. The shape of this view
     * will match {@code new Shape(axes)}.
     *
     * @see #view()
     * @see #view(NDArraySlice)
     * @see #copy()
     */
    public abstract AbstractDenseNDArray<?, U, V> view(int... axes);


    /**
     * Gets a view of this nD array. Manipulating the view <em>will</em> affect the original array.
     * Use {@see #copy()} to create a copy of this nD array.
     * @param slice The slice to extract from this nD array.
     * If the slice is strided, then the resulting view will also be strided.
     * @return A view of this nD array extracted from the specified {@code slice}. The shape of this view
     * will have shape {@code slice.getResultShape()}.
     *
     * @see #view()
     * @see #view(NDArraySlice)
     * @see #copy()
     */
    public abstract AbstractDenseNDArray<?, U, V> view(NDArraySlice slice);


    /**
     * Gets a view of this nD array but preserves the rank of the original array.
     * @param slice The slice to extract from this nD array. This slice <em>must</em> have the same rank as this nD array.
     * If the slice is strided, then the resulting view will also be strided.
     * @return A view of this nD array extracted from the specified {@code slice}. The shape of this view
     * will have shape {@code slice.getResultShape()}.
     * @throws IllegalArgumentException If {@code slice} does not have the same rank as this nD array.
     */
    public abstract T viewKeepRank(NDArraySlice slice);
}
