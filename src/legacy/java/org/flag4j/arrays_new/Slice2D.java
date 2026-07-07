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

import static org.flag4j.util.SliceSupport.invalidAxis;
import static org.flag4j.util.SliceSupport.sliceLength;


/**
 * Represents a fully specified two-dimensional strided array slice.
 *
 * <p>The row and column axes are each described by a
 * {@code (start, end, stride)} triple. Starts are inclusive and ends are
 * exclusive. Endpoint exclusivity is interpreted in the direction of
 * traversal:
 *
 * <ul>
 *     <li>For a positive stride, selected indices are
 *     {@code start, start + stride, ...}, while {@code index < end}.</li>
 *     <li>For a negative stride, selected indices are
 *     {@code start, start + stride, ...}, while {@code index > end}.</li>
 * </ul>
 *
 * <p>For negative strides, {@link #REVERSE_END} may be used as the exclusive
 * endpoint immediately before index {@code 0}. For example,
 * {@code (4, REVERSE_END, -1)} selects indices {@code 4, 3, 2, 1, 0}.
 *
 * <p>Axis {@code 0} represents rows and axis {@code 1} represents columns.
 * Use {@link #canSlice(Shape)} to verify that this slice can be applied to a
 * particular matrix shape.
 *
 * <p>Instances of this class are immutable and thread-safe.
 *
 * @see NDArraySlice
 * @see Slice
 * @see Slice1D
 */
public final class Slice2D implements NDArraySlice {

    /**
     * Row-axis index.
     */
    public static final int ROW_AXIS = 0;

    /**
     * Column-axis index.
     */
    public static final int COL_AXIS = 1;

    /**
     * Exclusive endpoint used with negative strides to include index {@code 0}.
     */
    public static final int REVERSE_END = NDArraySlice.REVERSE_END;

    /**
     * Inclusive row start index.
     */
    private final int rowStart;

    /**
     * Exclusive row end index.
     */
    private final int rowEnd;

    /**
     * Inclusive column start index.
     */
    private final int colStart;

    /**
     * Exclusive column end index.
     */
    private final int colEnd;

    /**
     * Row stride.
     */
    private final int rowStride;

    /**
     * Column stride.
     */
    private final int colStride;

    /**
     * Shape produced by this slice.
     */
    private final Shape resultShape;


    /**
     * Constructs a two-dimensional strided slice.
     *
     * @param rowStart Inclusive row start index.
     * @param rowEnd Exclusive row end index. For negative row strides,
     *               {@link #REVERSE_END} may be used to include row {@code 0}.
     * @param colStart Inclusive column start index.
     * @param colEnd Exclusive column end index. For negative column strides,
     *               {@link #REVERSE_END} may be used to include column {@code 0}.
     * @param rowStride Row stride.
     * @param colStride Column stride.
     *
     * @throws IllegalArgumentException If a start is negative, a stride is
     *         zero, an endpoint is invalid for the stride direction, or a
     *         start/end ordering is inconsistent with its stride.
     */
    public Slice2D(
            int rowStart,
            int rowEnd,
            int colStart,
            int colEnd,
            int rowStride,
            int colStride
    ) {
        SliceSupport.validateAxis(rowStart, rowEnd, rowStride, ROW_AXIS);
        SliceSupport.validateAxis(colStart, colEnd, colStride, COL_AXIS);

        this.rowStart = rowStart;
        this.rowEnd = rowEnd;
        this.colStart = colStart;
        this.colEnd = colEnd;
        this.rowStride = rowStride;
        this.colStride = colStride;

        this.resultShape = new Shape(
                sliceLength(rowStart, rowEnd, rowStride),
                sliceLength(colStart, colEnd, colStride)
        );
    }


    /**
     * Gets the start of this slice along the specified axis.
     *
     * @param axis The axis. {@code 0} is rows and {@code 1} is columns.
     * @return The inclusive start index along {@code axis}.
     *
     * @throws IndexOutOfBoundsException If {@code axis} is not {@code 0} or
     *         {@code 1}.
     */
    @Override
    public int getStart(int axis) {
        return switch (axis) {
            case ROW_AXIS -> rowStart;
            case COL_AXIS -> colStart;
            default -> throw invalidAxis(axis, 2);
        };
    }


    /**
     * Gets the end of this slice along the specified axis.
     *
     * @param axis The axis. {@code 0} is rows and {@code 1} is columns.
     * @return The exclusive end index along {@code axis}.
     *
     * @throws IndexOutOfBoundsException If {@code axis} is not {@code 0} or
     *         {@code 1}.
     */
    @Override
    public int getEnd(int axis) {
        return switch (axis) {
            case ROW_AXIS -> rowEnd;
            case COL_AXIS -> colEnd;
            default -> throw invalidAxis(axis, 2);
        };
    }


    /**
     * Gets the stride of this slice along the specified axis.
     *
     * @param axis The axis. {@code 0} is rows and {@code 1} is columns.
     * @return The stride along {@code axis}.
     *
     * @throws IndexOutOfBoundsException If {@code axis} is not {@code 0} or
     *         {@code 1}.
     */
    @Override
    public int getStride(int axis) {
        return switch (axis) {
            case ROW_AXIS -> rowStride;
            case COL_AXIS -> colStride;
            default -> throw invalidAxis(axis, 2);
        };
    }


    /**
     * Gets the start, end, and stride for the specified axis.
     *
     * @param axis The axis. {@code 0} is rows and {@code 1} is columns.
     * @return The start, end, and stride along {@code axis}.
     *
     * @throws IndexOutOfBoundsException If {@code axis} is not {@code 0} or
     *         {@code 1}.
     */
    @Override
    public IntTriple getStartEndStride(int axis) {
        return switch (axis) {
            case ROW_AXIS -> new IntTriple(rowStart, rowEnd, rowStride);
            case COL_AXIS -> new IntTriple(colStart, colEnd, colStride);
            default -> throw invalidAxis(axis, 2);
        };
    }


    /**
     * Checks whether this slice can be applied to the specified source shape.
     *
     * @param shape The source shape to slice.
     * @return {@code true} if this slice can be applied to {@code shape};
     *         {@code false} otherwise.
     */
    @Override
    public boolean canSlice(Shape shape) {
        Objects.requireNonNull(shape, "shape cannot be null.");

        return shape.getRank() == 2
                && SliceSupport.canSliceAxis(rowStart, rowEnd, rowStride, shape.getSize(ROW_AXIS))
                && SliceSupport.canSliceAxis(colStart, colEnd, colStride, shape.getSize(COL_AXIS));
    }


    /**
     * Gets the shape produced by applying this slice.
     *
     * <p>This does not verify that the slice is valid for a particular source
     * shape. Call {@link #canSlice(Shape)} when source-shape bounds validation
     * is required.
     *
     * @return The shape produced by this slice.
     */
    @Override
    public Shape getResultShape() {
        return resultShape;
    }


    /**
     * Checks whether {@code shape} exactly matches the shape produced by this
     * slice.
     *
     * <p>This compares the row and column extents independently, rather than
     * comparing only the total number of elements.
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
