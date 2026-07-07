/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.util;


import org.flag4j.arrays.Shape;

/**
 * A static utility class containing methods useful for manipulating {@link org.flag4j.arrays.Shape Shapes}, nD indices, and nD axes.
 */
public final class ShapeUtils {

    private ShapeUtils() {
        // Hide constructor for utility class.
    }


    /**
     * <p>Normalizes the specified list of {@code axes}, converting any negative indices
     * to their positive equivalents and verifying that each axis is valid and
     * appears at most once.
     *
     * <p>This method is intended to be used to verify axes and convert any negative axes
     * to positive axis by: {@code -a ~= shape.getRank() - a}.
     *
     * @param axes The axes to normalize (done in-place); negative values
     * are interpreted as {@code -a ~= rank - a}. For instance, {@code -1} marks the last axes, {@code -2}
     * marks the second-to-last axes, and so on.
     * @param shape The shape of the nD array to normalize {@code axes} for.
     * @return An array of length {@code axes.length} containing the normalized axes.
     *
     * @throws IllegalArgumentException If any axis is out of the range {@code [0, shape.getRank())},
     * or if the same axis is listed more than once or if {@code axes.length > shape.getRank()}.
     * 
     * @see #normalizeAxes(int[], int)
     */
    public static void normalizeAxes(int[] axes, Shape shape) {
        validateAndNormalize(axes, shape.getRank(), false);
    }


    /**
     * <p>Normalizes the specified list of {@code axes}, converting any negative indices
     * to their positive equivalents and verifying that each axis is valid and
     * appears at most once.
     *
     * <p>This method is intended to be used to verify axes and convert any negative axes
     * to positive axis by: {@code -a ~= rank - a}.
     *
     * @param axes The axes to normalize (done in-place); negative values
     * are interpreted as {@code -a ~= rank - a}. For instance, {@code -1} marks the last axes, {@code -2}
     * marks the second-to-last axes, and so on.
     * @param rank Rank of nD array to normalize axes for.
     *
     * @throws IllegalArgumentException If any axis is out of the range {@code [0, rank)},
     * or if the same axis is listed more than once or if {@code axes.length > rank}.
     *
     * @see #normalizeAxes(int[], Shape)
     */
    public static void normalizeAxes(int[] axes, int rank) {
        validateAndNormalize(axes, rank, false);
    }


    /**
     * <p>Computes a boolean array marking the normalized axes from a list of axes.
     *
     * <p>This method is intended to be used to verify axes and convert any negative axes
     * to positive axis by: {@code -a ~= rank - a}. These axes will be marked by {@code true} values in the
     * array returned. If index {@code i} in the returned array, then axes {@code i} was specified in {@code axes}
     * after being normalized.
     *
     * @param axes The axes to normalize; negative values
     * are interpreted as {@code -a ~= rank - a}. For instance, {@code -1} marks the last axes, {@code -2}
     * marks the second-to-last axes, and so on.
     * @param rank The rank of the nD array to normalize axes for.
     * @return A boolean array of length {@code rank} where the index of each normalized axis in {@code axes} is marked by
     * {@code true}.
     */
    public static boolean[] getNormalizedFlags(int[] axes, Shape shape) {
        return validateAndNormalize(axes, shape.getRank(), true);
    }


    /**
     * <p>Computes a boolean array marking the normalized axes from a list of axes.
     *
     * <p>This method is intended to be used to verify axes and convert any negative axes
     * to positive axis by: {@code -a ~= rank - a}. These axes will be marked by {@code true} values in the
     * array returned. If index {@code i} in the returned array, then axes {@code i} was specified in {@code axes}
     * after being normalized.
     *
     * @param axes The axes to normalize; negative values
     * are interpreted as {@code -a ~= rank - a}. For instance, {@code -1} marks the last axes, {@code -2}
     * marks the second-to-last axes, and so on.
     * @param rank The rank of the nD array to normalize axes for.
     * @return A boolean array of length {@code rank} where the index of each normalized axis in {@code axes} is marked by
     * {@code true}.
     */
    public static boolean[] getNormalizedFlags(int[] axes, int rank) {
        return validateAndNormalize(axes, rank, true);
    }


    /**
     * Normalizes the specified axes.
     * @param axes The axes to normalize. If {@code getFlags == false}, then the normalization will be done in-place in this array.
     * Otherwise, this array will remain unmodified.
     * @param rank The rank of the shape the axes are being normalized for.
     * @param getFlags Indicates if an array of boolean flags should be returned marking the normalized axes.
     * @return If {@code getFlags == true}, then a boolean array of length {@code rank} marking all the normalized axes will be
     * returned. If {@code getFlags == false}, then this method will return {@code null}.
     */
    private static boolean[] validateAndNormalize(int[] axes, int rank, boolean getFlags) {
        if(axes.length > rank)
            throw new IllegalArgumentException("The number of axes cannot be larger than the rank.");

        boolean[] normalizedFlags = getFlags ? new boolean[rank] : null;
        int[] normalizedAxes = getFlags ? null : new int[axes.length];

        for (int i=0, size=axes.length; i<size; i++) {
            int ax = axes[i];
            ax = (ax < 0) ? ax + rank : ax;  // Convert negative indices if needed.

            if (ax >= rank)
                throw new IllegalArgumentException(
                        "Axis " + axes[i] + " (normalized to " + ax + ") is out of bounds for array of rank " + rank + ".");
            if (normalizedFlags[ax])
                throw new IllegalArgumentException("Axis " + ax + " repeated.");

            if(getFlags) normalizedFlags[ax] = true;
            else axes[i] = ax;
        }

        return normalizedFlags;
    }
}
