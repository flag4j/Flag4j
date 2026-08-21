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

import org.flag4jv3.exceptions.NDArrayShapeException;
import org.flag4jv3.ndarrays.base.NDArrayBase;
import org.flag4jv3.util.ValidateParameters;
import org.flag4jv3.util.arrays.ArrayUtils;
import org.flag4jv3.util.tuples.IntTuple;

import java.io.Serial;
import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.StringJoiner;

/// Represents the shape of an [nD-array][NDArrayBase] (e.g., tensor, matrix, vector, etc.) specifying its dimensions and provides
/// utilities for shape-related operations.
///
/// The `Shape` class is immutable with respect to its dimensions, ensuring thread safety and consistency.
///
/// The [maximum rank][#MAX_RANK] is limited to `32` to provided a realistic upper-bound on the rank of an [nD-array][NDArrayBase].
/// This allows for some optimizations with fixed size "scratch" arrays.
///
/// ## Example usage:
/// {@snippet :
/// Shape shape = new Shape();  // Creates a shape for a scalar value.
/// shape = new Shape(3, 4, 5);  // Creates a shape for a 3x4x5 nD-array.
/// int rank = shape.rank();  // Gets the rank (number of dimensions).
///}
///
/// @see NDArrayBase
/// @see Layout
public class Shape implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /// The maximum allowed rank for a `Shape` object.
    public static final int MAX_RANK = 32;

    /// The rank of this shape.
    final int rank;
    /// An array containing the size of each dimension in this shape.
    final int[] dims; // To ensure shape immutability: do not modify here or anywhere else; do not expose this publicly.
    /// Total number of entries of this shape. Computed lazily.
    private BigInteger numElements = null;
    /// The total number of entries in this shape as an exact integer if possible. Computed lazily.
    private int numElementsIntExact = -1;


    /// Constructs a shape object from the specified dimensions.
    ///
    /// @param dims dimensions of the shape object. Must be less than [#MAX_RANK] in length.
    /// @throws IllegalArgumentException If `dims.length > MAX_RANK`.
    /// @throws IllegalArgumentException If any dimension is negative.
    public Shape(int... dims) {
        this(dims, true);
    }


    public Shape(IntTuple dims) {
        this(dims.items(), false); // No need to clone dims as `items()` already does so.
    }


    /// Constructs a shape with option to [clone][Object#clone()] the `dims`.
    ///
    /// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> {@code dims} must *not* be able to be modified outside this class. Setting
    ///     {@code cloneDims} to `false` is dangerous. Only do this if you are sure the `dims` instance prvably cannot
    ///     be referenced outside this class.
    /// </blockquote>
    ///
    /// @param dims
    /// @param cloneDims
    private Shape(int[] dims, boolean cloneDims) {
        if (dims.length > MAX_RANK) {
            throw new IllegalArgumentException(
                    "rank cannot exceed " + MAX_RANK + " but got " + dims.length + ".");
        }

        // Ensure all dimensions for the shape object are non-negative.
        ValidateParameters.ensureNonNegative(dims);
        this.dims = cloneDims ? dims.clone() : dims;
        rank = dims.length;
    }


    /**
     * Gets the shape of an nD-array as an array of dimensions.
     *
     * @return Shape of an nD-array as an integer array.
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
    /// @throws IllegalArgumentException If `axis` is negative or greater than this shape's rank-1.
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
    /// @param axes The axes to squeeze.
    /// @return If all `axes` of this` are already squeezed, then `this` is returned. Otherwise, a new shape which is
    /// a squeezed copy of `this` on `axes` is returned.
    ///
    /// @throws IllegalArgumentException If any axis in `axes` is negative or greater than this shape's rank-1.
    /// @see #squeeze()
    /// @see #squeeze(int)
    public Shape squeeze(int... axes) {
        ValidateParameters.ensureValidAxes(this, axes);

        boolean[] squeezeAxes = new boolean[rank];
        int squeezedRank = rank;
        for (int axis : axes) {
            if (!squeezeAxes[axis] && dims[axis] == 1) squeezedRank--;
            squeezeAxes[axis] = true;
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
     * Gets the total number of elements for an nD-array with this shape.
     *
     * @return The total number of elements for an nD-array with this shape.
     *
     * @see #numelIntValueExact()
     * @see #numelLongValueExact()
     */
    public BigInteger numel() {
        // Check if total elements have already been computed for this shape.
        if (numElements != null) return numElements;

        // Otherwise, it needs to be computed.
        BigInteger product = BigInteger.ONE;  // We can start at one because scalar tensors have a single entry.
        for (int dim : dims)
            product = product.multiply(BigInteger.valueOf(dim));
        numElements = product;

        return product;
    }


    /**
     * <p>Gets the total number of elements for an nD-array with this shape.
     * If the total number of elements exceeds {@link Integer#MAX_VALUE}, an exception is thrown.
     *
     * <p>This method is likely to be more efficient than {@link #numel()} if a primitive int value is desired.
     *
     * @return The total number of items for an nD-array with this shape.
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
     * <p>Gets the total number of elements for an nD-array with this shape as a {@code long}.
     * If the total number of elements exceeds {@link Long#MAX_VALUE}, an exception is thrown.
     *
     * @return The total number of elements for an nD-array with this shape.
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


    /// Gets the conical contiguous strides of an nD-array with this shape and specified [order][ContiguousOrder].
    ///
    /// @param order The [order][ContiguousOrder] to get the strides of. Must be [C][ContiguousOrder#C]
    /// or [F][ContiguousOrder#F].
    /// @return The conical contiguous strides of an nD-array with this shape and [order][ContiguousOrder].
    ///
    /// @throws IllegalArgumentException If `order` is not [C][ContiguousOrder#C] or [F][ContiguousOrder#F].
    /// @throws ArithmeticException      If all strides are *not* int sized.
    /// @see #getCContiguousStrides()
    /// @see #getFContiguousStrides()
    public int[] getContiguousStrides(ContiguousOrder order) {
        return switch (order) {
            case C -> getCContiguousStrides();
            case F -> getFContiguousStrides();
            default -> throw new IllegalStateException("order must be C of F but got " + order);
        };
    }


    /// Gets the conical [C-contiguous][ContiguousOrder#C] strides of an nD-array with this shape.
    ///
    /// @return The conical [C-contiguous][ContiguousOrder#C] strides of an nD-array with this shape.
    ///
    /// @throws ArithmeticException If all strides are *not* int sized.
    /// @see #getContiguousStrides(ContiguousOrder)
    /// @see #getFContiguousStrides()
    public int[] getCContiguousStrides() {
        int[] strides = new int[rank];
        if (rank == 0) return strides;

        strides[rank - 1] = 1;

        for (int i = rank - 2; i >= 0; i--) {
            strides[i] = Math.multiplyExact(dims[i + 1], strides[i + 1]);
        }

        return strides;
    }


    /// Gets the conical [F-contiguous][ContiguousOrder#F] strides of an nD-array with this shape.
    ///
    /// @return The conical [F-contiguous][ContiguousOrder#F] strides of an nD-array with this shape.
    ///
    /// @throws ArithmeticException If all strides are *not* int sized.
    /// @see #getContiguousStrides(ContiguousOrder)
    /// @see #getCContiguousStrides()
    public int[] getFContiguousStrides() {
        int[] strides = new int[rank];
        if (rank == 0) return strides;

        strides[0] = 1;
        for (int i = 1; i < rank; i++) {
            strides[i] = Math.multiplyExact(dims[i - 1], strides[i - 1]);
        }

        return strides;
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
     * Checks if a value fits in an int.
     *
     * @param value The value to check.
     * @return {@code true} if {@code value} fits in an int; {@code false} otherwise.
     */
    static boolean fitsInInt(long value) {
        return value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE;
    }


    /**
     * Gets the rank of this shape (e.g., the number of dimensions this shape represents).
     *
     * @return The rank of this shape.
     */
    /// Gets the rank of this shape. (e.g., the number of dimensions of this shape.)
    ///
    /// @return
    public int rank() {
        return rank;
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


    /// Ensures that another shape is [equal][#equals(java.lang.Object)] to `this` shape.
    ///
    /// @param other The other shape to check equality with.
    /// @return A reference to `this` shape.
    ///
    /// @throws NDArrayShapeException If `other` is *not* [equal][#equals(java.lang.Object)] to `this`.
    public Shape requireEqual(Shape other) {
        if (!equals(other)) {
            throw new NDArrayShapeException("Expecting equal shapes but got " + this + " and " + other);
        }

        return this;
    }
}
