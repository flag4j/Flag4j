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
import org.flag4jv3.util.ValidateParameters;
import org.flag4jv3.util.arrays.ArrayUtils;

import java.io.Serial;
import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.StringJoiner;

/**
 * Represents the shape of an nD array (e.g., tensor, matrix, vector, etc.) specifying its dimensions and provides
 * utilities for shape-related operations.
 *
 * <p>The {@code Shape} class is immutable with respect to its dimensions, ensuring thread safety and consistency.
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * Shape shape = new Shape();  // Creates a shape for a scalar value.
 * shape = new Shape(3, 4, 5);  // Creates a shape for a 3x4x5 nD array.
 * int rank = shape.getRank();  // Gets the rank (number of dimensions).
 * }</pre>
 *
 * @see NDArrayBase
 * @see Layout
 */
public class Shape implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * The rank of an nD array with this shape.
     */
    public final int rank;
    /**
     * An array containing the size of each dimension in this shape.
     */
    final int[] dims; // To ensure shape immutability: do not modify here or anywhere else; do not expose this publicly.
    /**
     * Total number of entries of this shape. This is only computed on demand by {@link #numel()}.
     */
    private BigInteger numElements = null;
    /**
     * Stores the total number of entries in this shape as an exact integer if possible.
     * This is only computed on demand by {@link #numelIntValueExact()}.
     */
    private int numElementsIntExact = -1;

    /// Stores the strides of a contiguous array with this shape. Lazily evaluated.
    private int[] contiguousStrides = null;


    /**
     * Constructs a shape object from specified dimensions.
     *
     * @param dims The dimension measurements for the shape object. All items must be non-negative.
     * @throws IllegalArgumentException If any dimension is negative.
     */
    public Shape(int... dims) {
        // Ensure all dimensions for the shape object are non-negative.
        ValidateParameters.ensureNonNegative(dims);
        this.dims = dims;
        rank = dims.length;
    }


    /**
     * Gets the shape of an nD array as an array of dimensions.
     *
     * @return Shape of an nD array as an integer array.
     */
    public int[] dims() {
        return dims.clone();
    }


    /**
     * Get the size of the shape object in the specified dimension.
     *
     * @param i Dimension to get the size of.
     * @return The size of this shape object in the specified dimension.
     */
    public int getSize(int i) {
        return dims[i];
    }


    /**
     * Returns a slice of this shape starting from the specified index to the end of this shape's dimensions.
     *
     * @param startIdx The starting index for slicing (inclusive).
     * @return A new {@code Shape} object containing the dimensions from {@code startIdx} to the end dimension.
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this shape's rank.
     */
    public Shape slice(int startIdx) {
        return slice(startIdx, dims.length);
    }


    /**
     * Returns a slice of this shape from the specified start index to the stop index of this shape's dimensions.
     *
     * @param startIdx The starting index for slicing (inclusive).
     * @param stopIdx The stopping index for slicing (exclusive).
     * @return A new {@code Shape} object containing the dimensions from {@code startIdx} to {@code stopIdx}.
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} or {@code stopIdx} is out of bounds.
     * @throws IllegalArgumentException  If {@code startIdx > stopIdx}.
     */
    public Shape slice(int startIdx, int stopIdx) {
        return new Shape(Arrays.copyOfRange(dims, startIdx, stopIdx));
    }


    // TODO NOW: DOCS + VERIFY
    public Shape slice(int startIdx, int stopIdx, int stride) {
        return new Shape(ArrayUtils.copyOfStridedRange(dims, startIdx, stopIdx, stride));
    }


    /// TODO NOW: DOCS + VERIFY (also, is this the best name for this method?)
    public Shape slice(int[] dimIdxs) {
        int[] newDims = new int[dimIdxs.length];

        var c = 0;
        for (int idx : dimIdxs) {
            newDims[c++] = dims[idx];
        }

        return new Shape(newDims);
    }


    /// Squeezes this shape. That is, removes axes with size one.
    ///
    /// @return If `this` is already squeezed, then `this` is returned. Otherwise, a new shape which is
    ///                                                                 a squeezed copy of `this` is returned.
    ///
    /// @see #squeeze(int)
    /// @see #squeeze(int...)
    public Shape squeeze() {
        int squeezedRank = 0;

        for (int dim : dims) {
            if (dim != 1) squeezedRank++;
        }

        if (squeezedRank == rank) return this;

        int[] newDims = new int[squeezedRank];
        int dest = 0;

        for (int dim : dims) {
            if (dim != 1) newDims[dest++] = dim;
        }

        return new Shape(newDims);
    }


    /// Squeezes this shape on a single axis. That is, removes the axis if it has size one.
    ///
    /// @param axis The axis to squeeze.
    /// @return If `axis` of this` is already squeezed, then `this` is returned. Otherwise, a new shape which is
    /// a squeezed copy of `this` on `axis` is returned.
    ///
    /// @throws IllegalArgumentException If `axis` is negative or greater than this shapes rank-1.
    /// @see #squeeze()
    /// @see #squeeze(int...)
    public Shape squeeze(int axis) {
        ValidateParameters.ensureValidAxes(this, axis);

        if (dims[axis] != 1) {
            return this;
        }

        int[] newDims = new int[rank - 1];

        for (int src = 0, dest = 0; src < rank; src++) {
            if (src != axis) {
                newDims[dest++] = dims[src];
            }
        }

        return new Shape(newDims);
    }


    /// Squeezes this shape on a set of `axes`. That is, removes any `axes` that has size one.
    ///
    /// @param axex The axes to squeeze.
    /// @return If all `axes` of this` are already squeezed, then `this` is returned. Otherwise, a new shape which is
    /// a squeezed copy of `this` on `axes` is returned.
    ///
    /// @throws IllegalArgumentException If any axis in `axes` is negative or greater than this shapes rank-1.
    /// @see #squeeze()
    /// @see #squeeze(int)
    public Shape squeeze(int... axes) {
        ValidateParameters.ensureValidAxes(this, axes);

        boolean[] squeezeAxes = new boolean[rank];
        for (int axis : axes) {
            squeezeAxes[axis] = true;
        }

        int squeezedRank = rank;
        for (int axis : axes) {
            if (dims[axis] == 1) {
                squeezedRank--;
            }
        }

        if (squeezedRank == rank) {
            return this;
        }

        int[] newDims = new int[squeezedRank];
        int dest = 0;

        for (int axis = 0; axis < rank; axis++) {
            if (!squeezeAxes[axis] || dims[axis] != 1) {
                newDims[dest++] = dims[axis];
            }
        }

        return new Shape(newDims);
    }


    /**
     * Flattens this shape to a rank-1 shape with a dimension equal to the product of all dimensions in this shape.
     *
     * @return A rank-1 shape with a dimension equal to the product of all dimensions in this shape.
     *
     * @throws ArithmeticException If the product of this shape's dimensions is too large to be stored in a 32-bit integer.
     */
    public Shape flatten() {
        return new Shape(numelIntValueExact());
    }


    /**
     * Swaps two axes of this shape. If this shape has had its strides computed, then new strides will also be computed for the
     * resulting shape.
     *
     * @param axis1 First axis to swap.
     * @param axis2 Second axis to swap.
     * @return A copy of this shape with the specified axis swapped.
     *
     * @throws ArrayIndexOutOfBoundsException If either axis is not within [0, {@link #rank}-1].
     * @see #permuteAxes(int...)
     * @see #unsafePermuteAxes(int...)
     */
    public Shape swapAxes(int axis1, int axis2) {
        int[] newDims = dims.clone();
        ArrayUtils.swap(newDims, axis1, axis2);

        return new Shape(newDims);
    }


    /**
     * Permutes the axes of this shape.
     *
     * @param axes New axes permutation for the shape. This must be a permutation of {@code {1, 2, 3, ... N}} where
     * {@code N} is the rank of this shape.
     * @return Returns this shape.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N}}.
     * @see #swapAxes(int, int) (int...)
     * @see #unsafePermuteAxes(int...)
     */
    public Shape permuteAxes(int... axes) {
        ValidateParameters.ensureAllEqual(rank, axes.length);
        ValidateParameters.ensurePermutation(axes);

        int[] permutedDims = new int[dims.length];

        var i = 0;
        for (int axis : axes)  // Permute axes.
            permutedDims[i++] = dims[axis];

        return new Shape(permutedDims);
    }


    /**
     * <p>Permutes the axes of this shape.
     *
     * <p>Warning: Unlike {@link #permuteAxes(int...)}, this method does not perform bounds checking on {@code axes} or ensure that
     * {@code axes} is a permutation of {@code {1, 2, 3, ... n}}. This may result in unexpected behavior if {@code tempDims} is
     * malformed.
     *
     * @param axes New axes permutation for the shape. This must be a permutation of {@code {1, 2, 3, ... n}} where
     * {@code n} is the rank of this shape.
     * @return Returns this shape.
     *
     * @see #permuteAxes(int...)
     * @see #swapAxes(int, int)
     */
    public Shape unsafePermuteAxes(int... axes) {
        int[] permutedDims = new int[dims.length];

        var i = 0;
        for (int axis : axes)  // Permute axes.
            permutedDims[i++] = dims[axis];

        return new Shape(permutedDims);
    }


    /**
     * Gets the total number of elements for an nD array with this shape.
     *
     * @return The total number of elements for an nD array with this shape.
     *
     * @see #numelIntValueExact()
     * @see #numelLongValueExact()
     */
    public BigInteger numel() {
        // Check if total elements have already been computed for this shape.
        if (numElements != null) return numElements;

        // Otherwise, the total items needs to be computed.
        BigInteger product = BigInteger.ONE;  // We can start at one because scalar tensors have a single entry.
        for (int dim : dims)
            product = product.multiply(BigInteger.valueOf(dim));
        numElements = product;

        return product;
    }


    /**
     * <p>Gets the total number of elements for an nD array with this shape.
     * If the total number of elements exceeds {@link Integer#MAX_VALUE}, an exception is thrown.
     *
     * <p>This method is likely to be more efficient than {@link #numel()} if a primitive int value is desired.
     *
     * @return The total number of items for an nD array with this shape.
     *
     * @throws ArithmeticException If the total number of items overflows a primitive int.
     * @see #numel()
     * @see #numelLongValueExact()
     */
    public int numelIntValueExact() {
        if (numElementsIntExact >= 0) {
            return numElementsIntExact; // Already computed.
        }

        int product = 1; // Rank-0 scalar has one element.

        for (int dim : dims) {
            product = Math.multiplyExact(product, dim);
        }

        numElementsIntExact = product;
        return product;
    }


    /**
     * <p>Gets the total number of elements for an nD array with this shape as a {@code long}.
     * If the total number of elements exceeds {@link Long#MAX_VALUE}, an exception is thrown.
     *
     * @return The total number of elements for an nD array with this shape.
     *
     * @throws ArithmeticException If the total number of items overflows a primitive int.
     * @see #numelIntValueExact()
     * @see #numel()
     */
    public long numelLongValueExact() {
        if (numElementsIntExact >= 0) {
            return numElementsIntExact; // Already computed.
        }

        long product = 1; // Rank-0 scalar has one element.

        for (int dim : dims) {
            product = Math.multiplyExact(product, dim);
        }

        if (fitsInInt(product)) {
            numElementsIntExact = (int) product;
        }

        return product;
    }


    /**
     * Checks if the total number of elements represented by this shape can be represented as a 32-bit integer without overflowing.
     *
     * @return {@code true} if the total number of elements represented by this shape can be represented as a
     * 32-bit integer without overflowing; {@code false} if it would overflow.
     */
    public boolean isIntSized() {
        // Early out if we have already computed this.s
        if (numElementsIntExact >= 0) return true;

        try {
            numelIntValueExact();
        } catch (ArithmeticException e) {
            return false; // Could not compute the int value exactly...
        }

        return true;
    }


    /**
     * Gets the strides of a contiguous nD array with this shape.
     *
     * @return The strides of a contiguous nD array with this shape.
     */
    public int[] getContiguousStrides() {
        ensureStridesComputed();
        return contiguousStrides.clone();
    }


    /**
     * Gets the strides of a contiguous nD array with this shape.
     * <p><b>Warning:</b> This method returns a reference to the internal array of strides. Modifying this array may lead to
     * unexpected behavior. Do <em>not</em> modify this array directly or leak it to other objects where it can be modified.
     *
     * @return The strides of a contiguous nD array with this shape.
     */
    int[] getContiguousStridesUnsafe() {
        ensureStridesComputed();
        return contiguousStrides;
    }


    /**
     * Computes the index of the 1D items array for a dense nD array from nD indices for an nD array with this shape.
     *
     * @param nDIndex nD index within an nD array with this shape.
     * @return The 1D index of the element at the specified nD index in the 1D items array of a dense nD array.
     *
     * @throws IllegalArgumentException  If the number of indices does not match the rank of this shape.
     * @throws IndexOutOfBoundsException If any index does not fit within an nD array with this shape.
     * @see #to1DIndexUnsafe(int...)
     */
    public int to1DIndex(int... nDIndex) {
        if (nDIndex.length != rank) {
            throw new IllegalArgumentException("Indices rank " + nDIndex.length + " does not match shape with rank " + rank);
        }

        ensureStridesComputed();

        int index = 0;
        for (int axis = 0; axis < rank; axis++) {
            int idx = nDIndex[axis];
            if (idx < 0 || idx >= dims[axis]) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for axis " + axis +
                        " of shape " + this);
            }

            index += idx*contiguousStrides[axis];
        }

        return index;
    }


    /**
     * <p>Computes the index of the 1D items array, for a dense nD array, from nD indices for an nD array with this shape.
     * <p><b>Warning</b>: Unlike {@link #to1DIndex(int...)}, this method does not perform bounds checking on indices. This can lead
     * to exceptions or undefined behavior if {@code ndIndex} is not a valid nDIndex for this shape. This method is intended
     * to be used internally. Only use this method if you <em>absolutly</em> know what you are doing.
     *
     * @param nDIndex Indices of nD array with this shape.
     * @return The index of the element at the specified indices in the 1D items array of a dense nD array.
     *
     * @throws IllegalArgumentException  If the number of indices does not match the rank of this shape.
     * @throws IndexOutOfBoundsException If any index does not fit within an nD array with this shape.
     * @see #to1DIndex(int...)
     */
    public int to1DIndexUnsafe(int... nDIndex) {
        ensureStridesComputed(); // Computes strides if not previously computed.

        int index = 0;
        for (int i = 0, stop = rank; i < stop; i++)
            index += nDIndex[i]*contiguousStrides[i];

        return index;
    }


    /**
     * Efficiently computes the nD array index based on a 1D index from the internal 1D items array.
     *
     * @param index Index of the internal 1D items array.
     * @return The multidimensional indices corresponding to the 1D items array index. This will be an array of integers
     * with length equal to the {@link #rank() rank} of this shape.
     *
     * @see #toNDIndices(int...)
     * @see #to1DIndex(int...)
     */
    public int[] toNDIndex(int index) {
        ensureStridesComputed(); // Ensure strides are initialized if not already.

        int numElements = numelIntValueExact();
        if (index < 0 || index >= numElements) {
            throw new IndexOutOfBoundsException(
                    "1D index " + index + " out of bounds for shape " + this);
        }

        int[] indices = new int[rank];
        int remaining = index;

        for (int axis = 0; axis < rank; axis++) {
            indices[axis] = remaining/contiguousStrides[axis];
            remaining %= contiguousStrides[axis];
        }

        return indices;
    }


    /**
     * Efficiently computes the nD array indices from multiple 1D indices from the internal 1D items array.
     *
     * @param indices Array of 1D indices.
     * @return The multidimensional indices corresponding to the 1D items array index. This will be an array of integers
     * with length equal to the {@link #rank} of this shape.
     *
     * @see #toNDIndex(int)
     * @see #to1DIndex(int...)
     */
    public int[][] toNDIndices(int... indices) {
        ensureStridesComputed();

        int numElements = numelIntValueExact();
        int[][] nDIndices = new int[indices.length][rank];

        for (int i = 0; i < indices.length; i++) {
            int flatIndex = indices[i];

            if (flatIndex < 0 || flatIndex >= numElements) {
                throw new IndexOutOfBoundsException(
                        "1D index " + flatIndex + " out of bounds for shape " + this);
            }

            int remaining = flatIndex;
            int[] ndIndex = nDIndices[i];

            for (int axis = 0; axis < rank; axis++) {
                ndIndex[axis] = remaining/contiguousStrides[axis];
                remaining %= contiguousStrides[axis];
            }
        }

        return nDIndices;
    }


    /**
     * Checks if this shape is square. That is, if <em>all</em> dimensions of this shape are equal.
     *
     * @return {@code true} if all dimensions of this shape are equal; {@code false} otherwise.
     */
    public boolean isSquare() {
        if (dims.length <= 1) return true;
        int refDim = dims[0];

        for (int i = 1; i < rank; i++)
            if (dims[i] != refDim) return false;

        return true;
    }


    /**
     * Checks if an object is equal to this shape.
     *
     * @param b Object to compare with this shape.
     * @return True if d is a Shape object and equal to this shape.
     */
    @Override
    public boolean equals(Object b) {
        // Check for early returns.
        if (this == b) return true;
        if (b == null) return false;
        if (b.getClass() != getClass()) return false;

        return Arrays.equals(dims, ((Shape) b).dims);
    }


    /**
     * Generates the hashcode for this shape object. This is computed by passing the dims array of this shape object to
     * {@link java.util.Arrays#hashCode(int[])}.
     *
     * @return The hashcode for this array object.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }


    /**
     * Converts this Shape object to a string format.
     *
     * @return The string representation for this Shape object.
     */
    public String toString() {
        StringJoiner joiner = new StringJoiner(", ", "(", ")");

        for (int d : dims)
            joiner.add(Integer.toString(d));

        return joiner.toString();
    }


    /**
     * Checks if a value fits in an int.
     *
     * @param value The value to check.
     * @return {@code true} if {@code value} fits in an int; {@code false} otherwise.
     */
    static boolean fitsInInt(long value) {
        return value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE;
    }


    /// Computes the strides of a contiguous nD array with this shape. This fills out and sets
    /// [#contiguousStrides] and [#numElementsIntExact].
    ///
    /// @throws ArithmeticException If the product of all dimensions of this shape *do not* fit in an integer.
    private void ensureStridesComputed() {
        if (contiguousStrides == null) return; // Already computed; nothing to do.

        // Check for some trivial cases.
        if (rank == 0) {
            contiguousStrides = new int[0];
            numElementsIntExact = 1; // Rank-0 tensors are scalars.
            return;
        } else if (rank == 1) {
            contiguousStrides = new int[]{1};
            numElementsIntExact = dims[0];
        }

        int[] strides = new int[rank];
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            strides[i] = Math.multiplyExact(dims[i + 1], strides[i + 1]);
        }

        // Verify that the total element count fits in an int.
        numElementsIntExact = Math.multiplyExact(dims[0], strides[0]);
        contiguousStrides = strides; // We defer setting until here. If an exception is thrown, this will correctly stay null.
    }
}
