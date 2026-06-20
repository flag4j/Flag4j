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

import java.util.Objects;

import static org.flag4j.arrays_new.SliceSupport.sliceLength;
import static org.flag4j.arrays_new.SliceSupport.validateAxis;

/**
 * Represents a fully specified one-dimensional strided array slice.
 *
 * <p>This slice is specified by a {@code (start, end, stride)} triple.
 * The {@code start} index is inclusive and the {@code end} index is
 * exclusive, with exclusivity interpreted in the direction of traversal:
 *
 * <ul>
 *     <li>For {@code stride > 0}, selected indices are
 *     {@code start, start + stride, ...}, while {@code index < end}.</li>
 *     <li>For {@code stride < 0}, selected indices are
 *     {@code start, start + stride, ...}, while {@code index > end}.</li>
 * </ul>
 *
 * <p>For negative strides, {@link #REVERSE_END} may be used as the exclusive
 * endpoint immediately before index {@code 0}. For example,
 * {@code (4, REVERSE_END, -1)} selects indices {@code 4, 3, 2, 1, 0}.
 *
 * <p>Use {@link #canSlice(Shape)} to verify that this slice is valid for a
 * particular source shape.
 *
 * <p>Instances of this class are immutable and thread-safe.
 *
 * @see NDArraySlice
 * @see Slice
 * @see Slice2D
 */
public final class Slice1D implements NDArraySlice {

    /**
     * The only axis supported by a one-dimensional slice.
     */
    public static final int AXIS = 0;

    /**
     * Exclusive endpoint used with negative strides to include index {@code 0}.
     */
    public static final int REVERSE_END = NDArraySlice.REVERSE_END;

    /**
     * Inclusive slice start index.
     */
    private final int start;

    /**
     * Exclusive slice end index.
     */
    private final int end;

    /**
     * Slice stride.
     */
    private final int stride;

    /**
     * Shape produced by this slice.
     */
    private final Shape resultShape;


    /**
     * Constructs a one-dimensional strided slice.
     *
     * @param start Inclusive slice start index.
     * @param end Exclusive slice end index. For negative strides,
     *             {@link #REVERSE_END} may be used to include index {@code 0}.
     * @param stride Slice stride.
     *
     * @throws IllegalArgumentException If {@code start} is negative,
     *         {@code stride} is zero, {@code end} is invalid for the stride
     *         direction, or the start/end ordering is inconsistent with the
     *         stride direction.
     */
    public Slice1D(int start, int end, int stride) {
        validateAxis(start, end, stride, AXIS);

        this.start = start;
        this.end = end;
        this.stride = stride;
        this.resultShape = new Shape(sliceLength(start, end, stride));
    }


    /**
     * Gets the inclusive start index of this slice.
     *
     * @param axis The dimension. Must be {@code 0}.
     * @return The inclusive slice start index.
     *
     * @throws IndexOutOfBoundsException If {@codeaxis != 0}.
     */
    @Override
    public int getStart(int axis) {
        validateAxis(axis, resultShape.getRank());
        return start;
    }


    /**
     * Gets the exclusive end index of this slice.
     *
     * @param axis The dimension. Must be {@code 0}.
     * @return The exclusive slice end index.
     *
     * @throws IndexOutOfBoundsException If {@codeaxis != 0}.
     */
    @Override
    public int getEnd(int axis) {
        validateAxis(axis, resultShape.getRank());
        return end;
    }


    /**
     * Gets the stride of this slice.
     *
     * @param axis The dimension. Must be {@code 0}.
     * @return The slice stride.
     *
     * @throws IndexOutOfBoundsException If {@codeaxis != 0}.
     */
    @Override
    public int getStride(int axis) {
        validateAxis(axis, resultShape.getRank());
        return stride;
    }


    /**
     * Gets the start, end, and stride of this slice.
     *
     * @param axis The dimension. Must be {@code 0}.
     * @return The slice's start, end, and stride.
     *
     * @throws IndexOutOfBoundsException If {@codeaxis != 0}.
     */
    @Override
    public IntTriple getStartEndStride(int axis) {
        return new IntTriple(start, end, stride);
    }


    /**
     * Checks whether this slice can be applied to a source shape.
     *
     * @param shape The source shape to slice.
     * @return {@code true} if this slice can be applied to {@code shape};
     *         {@code false} otherwise.
     */
    @Override
    public boolean canSlice(Shape shape) {
        Objects.requireNonNull(shape, "shape cannot be null.");

        return shape.getRank() == 1
                && SliceSupport.canSliceAxis(start, end, stride, shape.getSize(AXIS));
    }


    /**
     * Gets the shape produced by applying this slice.
     *
     * <p>This does not validate that the slice is in bounds for a particular
     * source shape. Use {@link #canSlice(Shape)} when source-shape validation
     * is required.
     *
     * @return The shape produced by this slice.
     */
    @Override
    public Shape getResultShape() {
        return resultShape;
    }


    /**
     * Checks whether {@code shape} is exactly the shape produced by this slice.
     *
     * @param shape Candidate result shape.
     * @return {@code true} if {@code shape} is exactly this slice's result
     *         shape; {@code false} otherwise.
     */
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