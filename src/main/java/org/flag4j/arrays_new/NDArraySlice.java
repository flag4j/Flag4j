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


/**
 * Interface for nD array slices.
 */
public interface NDArraySlice {

    /**
     * Exclusive endpoint used with negative strides to include index {@code 0}.
     *
     * <p>For example, {@code (4, REVERSE_END, -1)} selects indices
     * {@code 4, 3, 2, 1, 0}.
     */
    public static final int REVERSE_END = -1;

    /**
     * Get the start of the nD slice along the given dimension.
     * @param axis The dimension to get the slice start of.
     * @return The start of the nD slice along {@code axis}.
     */
    public int getStart(int axis);

    /**
     * Get the end of the nD slice along the given dimension.
     * @param axis The dimension to get the slice end of.
     * @return The end of the nD slice along {@code axis}.
     */
    public int getEnd(int axis);


    /**
     * Get the stride of the nD slice along the given dimension.
     * @param axis The dimension to get the stride of.
     * @return The stride of the nD slice along {@code axis}.
     */
    public int getStride(int axis);

    /**
     * Checks if this slice can be applied to the given shape.
     * @param shape The shape to be sliced.
     * @return The result of the slice operation.
     */
    public boolean canSlice(Shape shape);


    /**
     * Gets the start, end, and stride of the slice along the given dimension.
     * @param axis The dimension to get the slice start, end, and stride of.
     * @return The start, end, and stride of the nD slice along {@code axis}.
     */
    public IntTriple getStartEndStride(int axis);

    /**
     * Gets the resulting shape produced by this slice.
     * @return The shape produced by this slice.
     */
    public Shape getResultShape();

    /**
     * Checks whether {@code shape} is exactly the shape produced by this slice.
     *
     * <p>This checks each axis independently. It does not merely compare total
     * element counts, since shapes such as {@code [2, 6]} and {@code [3, 4]}
     * both contain twelve entries but are not equivalent slice results.
     *
     * @param shape Candidate result shape.
     * @return {@code true} if {@code shape} is exactly the result shape of this slice.
     */
    public boolean isResultShape(Shape shape);


    /**
     * Gets the rank of the resulting shape produced by this slice.
     * @return THe rank of the resulting shape produced by this slice.
     */
    public int getResultRank();
}
