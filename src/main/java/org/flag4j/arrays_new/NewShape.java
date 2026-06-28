package org.flag4j.arrays_new;

import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

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
 *
 * <pre>{@code
 * Shape shape = new Shape();  // Creates a shape for a scalar value.
 * shape = new Shape(3, 4, 5);  // Creates a shape for a 3x4x5 nD array.
 * int rank = shape.getRank();  // Gets the rank (number of dimensions).
 * }</pre>
 *
 * @see org.flag4j.arrays.backend.AbstractNDArray
 */
public class NewShape implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * The rank of an nD array with this shape.
     */
    private final int rank;
    /**
     * An array containing the size of each dimension in this shape.
     */
    private final int[] dims;
    /**
     * Total number of entries of this shape. This is only computed on demand by {@link #numel()}.
     */
    private BigInteger numElements = null;
    /**
     * Stores the total number of entries in this shape as an exact integer if possible.
     * This is only computed on demand by {@link #numelIntValueExact()}.
     */
    private int numElementsIntExact = -1;


    /**
     * Constructs a shape object from specified dimensions.
     * @param dims A list of the dimension measurements for this shape object. All data must be non-negative.
     * @throws IllegalArgumentException If any dimension is negative.
     */
    public NewShape(int... dims) {
        // Ensure all dimensions for the shape object are non-negative.
        ValidateParameters.ensureNonNegative(dims);
        this.dims = dims;
        rank = dims.length;
    }


    /**
     * Gets the rank of an nD array with this shape.
     * @return The rank for an nD array with this shape.
     */
    public int getRank() {
        return dims.length;
    }


    /**
     * Gets the shape of an nD array as an array of dimensions.
     * @return Shape of an nD array as an integer array.
     */
    public int[] getDims() {
        return dims;
    }


    /**
     * Get the size of the shape object in the specified dimension.
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
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this shape's rank.
     */
    public NewShape slice(int startIdx) {
        return slice(startIdx, dims.length);
    }


    /**
     * Returns a slice of this shape from the specified start index to the stop index of this shape's dimensions.
     *
     * @param startIdx The starting index for slicing (inclusive).
     * @param stopIdx The stopping index for slicing (exclusive).
     * @return A new {@code Shape} object containing the dimensions from {@code startIdx} to {@code stopIdx}.
     * @throws IndexOutOfBoundsException If {@code startIdx} or {@code stopIdx} is out of bounds.
     * @throws IllegalArgumentException If {@code startIdx > stopIdx}.
     */
    public NewShape slice(int startIdx, int stopIdx) {
        return new NewShape(Arrays.copyOfRange(dims, startIdx, stopIdx));
    }

    // TODO NOW: DOCS.
    public NewShape slice(int startIdx, int stopIdx, int stride) {
        return new NewShape(ArrayUtils.copyOfStridedRange(dims, startIdx, stopIdx, stride));
    }


    public NewShape slice(int[] dimIdxs) {
        int[] newDims = new int[dimIdxs.length];

        var c = 0;
        for (int idx : dimIdxs) {
            newDims[c++] = dims[idx];
        }

        return new NewShape(newDims);
    }


    /**
     * Flattens this shape to a rank-1 shape with a dimension equal to the product of all dimensions in this shape.
     * @return A rank-1 shape with a dimension equal to the product of all dimensions in this shape.
     * @throws ArithmeticException If the product of this shape's dimensions is too large to be stored in a 32-bit integer.
     */
    public NewShape flatten() {
        return new NewShape(numelIntValueExact());
    }


    /**
     * Swaps two axes of this shape. If this shape has had its strides computed, then new strides will also be computed for the
     * resulting shape.
     * @param axis1 First axis to swap.
     * @param axis2 Second axis to swap.
     * @return A copy of this shape with the specified axis swapped.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within [0, {@link #getRank() rank}-1].
     * @see #permuteAxes(int...)
     * @see #unsafePermuteAxes(int...)
     */
    public NewShape swapAxes(int axis1, int axis2) {
        int[] newDims = dims.clone();
        ArrayUtils.swap(newDims, axis1, axis2);

        return new NewShape(newDims);
    }


    /**
     * Permutes the axes of this shape.
     * @param axes New axes permutation for the shape. This must be a permutation of {@code {1, 2, 3, ... N}} where
     *             {@code N} is the rank of this shape.
     * @return Returns this shape.
     * @throws ArrayIndexOutOfBoundsException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N}}.
     * @see #swapAxes(int, int) (int...)
     * @see #unsafePermuteAxes(int...)
     */
    public NewShape permuteAxes(int... axes) {
        ValidateParameters.ensureAllEqual(getRank(), axes.length);
        ValidateParameters.ensurePermutation(axes);

        int[] permutedDims = new int[dims.length];

        var i=0;
        for(int axis : axes)  // Permute axes.
            permutedDims[i++] = dims[axis];

        return new NewShape(permutedDims);
    }


    /**
     * <p>Permutes the axes of this shape.
     *
     * <p>Warning: Unlike {@link #permuteAxes(int...)}, this method does not perform bounds checking on {@code axes} or ensure that
     * {@code axes} is a permutation of {@code {1, 2, 3, ... n}}. This may result in unexpected behavior if {@code tempDims} is
     * malformed.
     *
     * @param axes New axes permutation for the shape. This must be a permutation of {@code {1, 2, 3, ... n}} where
     *             {@code n} is the rank of this shape.
     * @return Returns this shape.
     * @see #permuteAxes(int...)
     * @see #swapAxes(int, int)
     */
    public NewShape unsafePermuteAxes(int... axes) {
        int[] permutedDims = new int[dims.length];

        var i=0;
        for(int axis : axes)  // Permute axes.
            permutedDims[i++] = dims[axis];

        return new NewShape(permutedDims);
    }


    /**
     * Gets the total number of elements for an nD array with this shape.
     * @return The total number of elements for an nD array with this shape.
     * @see #numelIntValueExact()
     * @see #numelLongValueExact()
     */
    public BigInteger numel() {
        // Check if total elements have already been computed for this shape.
        if(numElements !=null) return numElements;

        // Otherwise, the total data needs to be computed.
        BigInteger product = BigInteger.ONE;  // We can start at one because scalar tensors have a single entry.
        for(int dim : dims)
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
     * @return The total number of data for an nD array with this shape.
     * @throws ArithmeticException If the total number of data overflows a primitive int.
     * @see #numel()
     * @see #numelLongValueExact()
     */
    public int numelIntValueExact() {
        if(numElementsIntExact >= 0) return numElementsIntExact;  // The Value has already been computed.
        long product = 1;
        numElementsIntExact = 1;  // We can start at one because scalar tensors have a single entry.

        for (int dim : dims) {
            product *= dim;

            // If product > Long.MAX_VALUE, then product*value would overflow since all dims are positive.
            if (product > Integer.MAX_VALUE)
                throw new ArithmeticException("Integer overflow while computing total data in shape: " + this);
        }

        numElementsIntExact = (int) product;

        return numElementsIntExact;
    }


    /**
     * <p>Gets the total number of elements for an nD array with this shape as a {@code long}.
     * If the total number of elements exceeds {@link Long#MAX_VALUE}, an exception is thrown.
     *
     * @return The total number of elements for an nD array with this shape.
     * @throws ArithmeticException If the total number of data overflows a primitive int.
     * @see #numelIntValueExact()
     * @see #numel()
     */
    public long numelLongValueExact() {
        if(numElementsIntExact >= 0) return numElementsIntExact; // The Value has already been computed as an integer.

        long product = 1;
        for (long value : dims) {
            if (value == 0 || product == 0) {
                product = 0;
                continue;
            }

            // If product > Long.MAX_VALUE / value, then product*value would overflow since all dims are positive.
            if (product > Long.MAX_VALUE / value)
                throw new ArithmeticException("Long overflow while computing total data in the shape.");

            product *= value;
        }

        // If we can safely cast to an integer, update the cached integer value.
        if (product < Integer.MAX_VALUE)
            numElementsIntExact = (int) product;

        return product;
    }


    /**
     * Checks if the total number of elements represented by this shape can be represented as a 32-bit integer without overflowing.
     * @return {@code true} if the total number of elements represented by this shape can be represented as a
     * 32-bit integer without overflowing; {@code false} if it would overflow.
     */
    public boolean isIntSized() {
        // Early out if we have already computed this.s
        if(numElementsIntExact >= 0) return true;

        try {
            numelLongValueExact();
        } catch (ArithmeticException e) {
            return false; // Could not compute the int value exactly...
        }

        return true;
    }


    /**
     * Checks if this shape is square. That is, if <em>all</em> dimensions of this shape are equal.
     * @return {@code true} if all dimensions of this shape are equal; {@code false} otherwise.
     */
    public boolean isSquare() {
        if(dims.length <= 1) return true;
        int refDim = dims[0];

        for(int i=1; i<rank; i++)
            if(dims[i] != refDim) return false;

        return true;
    }


    /**
     * Checks if an object is equal to this shape.
     * @param b Object to compare with this shape.
     * @return True if d is a Shape object and equal to this shape.
     */
    @Override
    public boolean equals(Object b) {
        // Check for early returns.
        if(this == b) return true;
        if(b==null) return false;
        if(b.getClass() != getClass()) return false;

        return Arrays.equals(dims, ((NewShape) b).dims);
    }


    /**
     * Generates the hashcode for this shape object. This is computed by passing the dims array of this shape object to
     * {@link java.util.Arrays#hashCode(int[])}.
     * @return The hashcode for this array object.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }


    /**
     * Converts this Shape object to a string format.
     * @return The string representation for this Shape object.
     */
    public String toString() {
        StringJoiner joiner = new StringJoiner(", ", "(", ")");

        for(int d : dims)
            joiner.add(Integer.toString(d));

        return joiner.toString();
    }
}
