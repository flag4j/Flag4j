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

package org.flag4j.arrays_new;

import org.flag4j.arrays.IntTriple;
import org.flag4j.arrays.Shape;
import org.flag4j.util.SliceSupport;

import java.util.Objects;

import static org.flag4j.util.SliceSupport.validateAxis;


/**
 * Represents a fully specified n-dimensional strided array slice.
 *
 * <p>Each axis is described by a {@code (start, end, stride)} triple.
 * The {@code start} index is inclusive and the {@code end} index is
 * exclusive, with exclusivity interpreted in the direction of traversal:
 *
 * <ul>
 *   <li>For {@code stride > 0}, the selected indices are
 *       {@code start, start + stride, ...}, while {@code index < end}.</li>
 *   <li>For {@code stride < 0}, the selected indices are
 *       {@code start, start + stride, ...}, while {@code index > end}.</li>
 * </ul>
 *
 * <p>For negative strides, {@link #REVERSE_END} may be used as the exclusive
 * endpoint immediately before index {@code 0}. For example,
 * {@code (4, REVERSE_END, -1)} selects indices {@code 4, 3, 2, 1, 0}.
 *
 * <p>A {@code Slice} specifies index traversal independently of a source
 * array shape. Use {@link #canSlice(Shape)} to verify that this slice is
 * valid for a particular source shape. Use {@link #getResultShape()} or
 * {@link #isResultShape(Shape)} to inspect the shape produced by the slice.
 *
 * <p>Instances of this class are immutable and thread-safe.
 *
 * @see NDArraySlice
 * @see Slice1D
 * @see Slice2D
 */
public class Slice implements NDArraySlice {

    /**
     * Stores the starts of the stride.
     */
    private final int[] starts;

    /**
     * Stores the ends of the slice.
     */
    private final int[] ends;

    /**
     * Stores the strides of the slice.
     */
    private final int[] strides;

    /**
     * The resulting shape of the slice.
     */
    private final Shape resultShape;


    /**
     * Exclusive endpoint used with negative strides to include index {@code 0}.
     */
    public static final int REVERSE_END = NDArraySlice.REVERSE_END;


    /**
     * Constructs an nD array slice.
     * @param starts The starts of the slice (inclusive).
     * @param ends The slice ends (exclusive). For a negative stride,
     *             {@link #REVERSE_END} is permitted and represents the exclusive
     *             endpoint immediately before index {@code 0}.
     * @param strides The strides of the slice.
     */
    public Slice(int[] starts, int[] ends, int[] strides) {
        this.starts = Objects.requireNonNull(starts, "starts cannot be null.").clone();
        this.ends = Objects.requireNonNull(ends, "ends cannot be null.").clone();
        this.strides = Objects.requireNonNull(strides, "strides cannot be null.").clone();

        SliceSupport.validateAxes(this.starts, this.ends, this.strides);
        resultShape = SliceSupport.resultShape(this.starts, this.ends, this.strides);
    }


    /**
     * Get the start of the nD slice along the given axis.
     *
     * @param axis the axis to get the slice start of.
     *
     * @return The start of the nD slice along {@code axis}.
     */
    @Override
    public int getStart(int axis) {
        validateAxis(axis, resultShape.getRank());
        return starts[axis];
    }


    /**
     * Get the end of the nD slice along the given axis.
     *
     * @param axis the axis to get the slice end of.
     *
     * @return The end of the nD slice along {@code axis}.
     */
    @Override
    public int getEnd(int axis) {
        validateAxis(axis, resultShape.getRank());
        return ends[axis];
    }


    /**
     * Get the stride of the nD slice along the given axis.
     *
     * @param axis the axis to get the stride of.
     *
     * @return The stride of the nD slice along {@code axis}.
     */
    @Override
    public int getStride(int axis) {
        validateAxis(axis, resultShape.getRank());
        return strides[axis];
    }


    /**
     * Gets the start, end, and stride of the slice along the given axis.
     *
     * @param axis the axis to get the slice start, end, and stride of.
     *
     * @return The start, end, and stride of the nD slice along {@code axis}.
     */
    @Override
    public IntTriple getStartEndStride(int axis) {
        validateAxis(axis, resultShape.getRank());
        return new IntTriple(starts[axis], ends[axis], strides[axis]);
    }


    /**
     * Checks if this slice can be applied to the given shape.
     *
     * @param shape The shape to be sliced.
     *
     * @return {@code true} if this slice can be applied to {@code shape}; {@code false} otherwise.
     */
    @Override
    public boolean canSlice(Shape shape) {
        Objects.requireNonNull(shape, "shape cannot be null.");

        if (shape.getRank() != starts.length) {
            return false;
        }

        for (int axis = 0; axis < starts.length; axis++) {
            if (!SliceSupport.canSliceAxis(
                    starts[axis],
                    ends[axis],
                    strides[axis],
                    shape.getSize(axis)
            )) {
                return false;
            }
        }

        return true;
    }


    @Override
    public Shape getResultShape() {
        return resultShape;
    }


    @Override
    public boolean isResultShape(Shape shape) {
        Objects.requireNonNull(shape, "shape cannot be null.");
        return resultShape.equals(shape);
    }


    /**
     * Gets the rank of the resulting shape produced by this slice.
     *
     * @return THe rank of the resulting shape produced by this slice.
     */
    @Override
    public int getResultRank() {
        return resultShape.getRank();
    }
}
