/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.arrays;

import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.StringJoiner;

/**
 * Represents the shape of an nD array (e.g., tensor, matrix, vector, etc.) specifying its dimensions and provides
 * utilities for shape-related operations.
 *
 * <p>A shape is defined by an array of dimensions, where each dimension specifies the size of the nD array along a particular axis.
 * {@link #getStrides() Strides} can also be computed for the shape that specifies the number of elements to step in each dimension of
 * the shape when traversing an array with the given shape. Strides will always be row-major contiguous and allow for efficient
 * array traversal and mapping of nD indices to 1D contiguous indices.
 *
 * <p>This class also supports converting between multidimensional and flat indices, computing the shape's rank (i.e., number of
 * dimensions), computing the total number of data in an nD array with the given shape and manipulating dimensions through swaps or
 * permutations.
 *
 * <p>The {@code Shape} class is immutable with respect to its dimensions, ensuring thread safety and consistency. Strides
 * are computed lazily only when needed to minimize overhead.
 *
 * <h2>Example usage:</h2>
 *
 * <pre>{@code
 * Shape shape = new Shape();  // Creates a shape for a scalar value.
 * shape = new Shape(3, 4, 5);  // Creates a shape for a 3x4x5 nD array.
 * int rank = shape.getRank();  // Gets the rank (number of dimensions).
 * int[] strides = shape.getStrides();  // Retrieves the strides for this shape.
 * int flatIndex = getFlatIndex(2, 1, 4);  // Converts multidimensional indices to a flat index.
 * int[] multiDimIndex = shape.getNdIndices(56);  // Flat index to nD index: {2, 3, 1}.
 * }</pre>
 *
 * @see org.flag4j.arrays.backend.AbstractNDArray
 */
public class Shape implements Serializable {
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
     * An array containing the strides of all dimensions within this shape. This is computed lazily and cached.
     */
    private int[] strides;
    /**
     * Flag indicating if strides have been computed for this shape instance or not.
     */
    private boolean hasStrides;
    /**
     * Total number of entries of this shape. This is only computed on demand by {@link #totalEntries()}.
     */
    private BigInteger totalEntries = null;
    /**
     * Stores the total number of entries in this shape as an exact integer if possible.
     * This is only computed on demand by {@link #totalEntriesIntValueExact()}.
     */
    private int totalEntriesIntExact = -1;


    /**
     * Constructs a shape object from specified dimensions.
     * @param dims A list of the dimension measurements for this shape object. All data must be non-negative.
     * @throws IllegalArgumentException If any dimension is negative.
     */
    public Shape(int... dims) {
        // Ensure all dimensions for the shape object are non-negative.
        ValidateParameters.ensureNonNegative(dims);
        this.dims = dims;
        strides = new int[dims.length];  // Initialize. Will be filled lazily.
        hasStrides = false; // Indicate strides have not been computed.
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
     * Gets the strides of this shape as an array. Strides are the step sizes needed to move from one
     * element to another along each axis in the nD array.
     * @return The strides of this shape as an integer array.
     */
    public int[] getStrides() {
        makeStrides();
        return strides;
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
    public Shape slice(int startIdx) {
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
    public Shape slice(int startIdx, int stopIdx) {
        return new Shape(Arrays.copyOfRange(dims, startIdx, stopIdx));
    }


    /**
     * Flattens this shape to a rank-1 shape with a dimension equal to the product of all dimensions in this shape.
     * @return A rank-1 shape with a dimension equal to the product of all dimensions in this shape.
     * @throws ArithmeticException If the product of this shape's dimensions is too large to be stored in a 32-bit integer.
     */
    public Shape flatten() {
        return new Shape(totalEntriesIntValueExact());
    }


    /**
     * Constructs strides for each dimension of this shape as if for a newly constructed nD array.
     * Strides will be a monotonically decreasing sequence with the last stride being 1.
     * @return The strides for all dimensions of a newly constructed nD array with this shape.
     */
    private synchronized void createNewStrides() {
        if(strides.length>0) {
            strides[strides.length-1] = 1; // Set the last stride to 1.

            for(int i=strides.length-2; i>=0; i--)
                strides[i] = dims[i + 1]*strides[i + 1];
        }
    }


    /**
     * If strides have not already been computed for this shape, create them. Otherwise, do nothing.
     */
    private void makeStrides() {
        if(!hasStrides) {
            createNewStrides();
            hasStrides = true;
        }
    }


    /**
     * Computes the index of the 1D data array for a dense nD array from nD indices for an nD array with this shape.
     * @param nDIndex nD index within an nD array with this shape.
     * @return The 1D index of the element at the specified nD index in the 1D data array of a dense nD array.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this shape.
     * @throws IndexOutOfBoundsException If any index does not fit within an nD array with this shape.
     * @see #unsafeGet1DIndex(int...)
     */
    public int get1DIndex(int... nDIndex) {
        if(nDIndex.length != dims.length)
            throw new IllegalArgumentException("Indices rank " + nDIndex.length + " does not match nD array rank " + dims.length);

        makeStrides(); // Computes strides if not previously computed.

        int index = 0;
        for(int i=0, stop=nDIndex.length; i<stop; i++) {
            int idx = nDIndex[i];
            if(idx < 0 || idx >= dims[i]) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for axis " + i +
                        " of nD array with shape " + this);
            }

            index += idx*strides[i];
        }

        return index;
    }


    /**
     * <p>Computes the index of the 1D data array, for a dense nD array, from nD indices for an nD array with this shape.
     * <p>Warning: Unlike {@link #get1DIndex(int...)}, this method does not perform bounds checking on indices. This can lead
     * to exceptions being thrown or possibly no exception but incorrect results if {@code indices} are not valid indices.
     * @param nDIndex Indices of nD array with this shape.
     * @return The index of the element at the specified indices in the 1D data array of a dense nD array.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this shape.
     * @throws IndexOutOfBoundsException If any index does not fit within an nD array with this shape.
     * @see #get1DIndex(int...)
     */
    public int unsafeGet1DIndex(int... nDIndex) {
        makeStrides(); // Computes strides if not previously computed.

        int index = 0;
        for(int i = 0, stop = nDIndex.length; i < stop; i++)
            index += nDIndex[i]*strides[i];

        return index;
    }


    /**
     * Efficiently computes the nD array index based on a 1D index from the internal 1D data array.
     * @param index Index of the internal 1D data array.
     * @return The multidimensional indices corresponding to the 1D data array index. This will be an array of integers
     * with length equal to the {@link #getRank() rank} of this shape.
     * @see #getNdIndices(int...)
     * @see #get1DIndex(int...)
     */
    public int[] getNdIndices(int index) {
        makeStrides(); // Ensure strides are initialized if not already.
        int[] indices = new int[rank];

        for (int i = 0; i < strides.length; i++)
            indices[i] = (index / strides[i]) % dims[i];

        return indices;
    }


    /**
     * Efficiently computes the nD array indices from multiple 1D indices from the internal 1D data array.
     * @param indices Array of 1D indices.
     * @return The multidimensional indices corresponding to the 1D data array index. This will be an array of integers
     * with length equal to the {@link #getRank() rank} of this shape.
     * @see #getNdIndices(int)
     * @see #get1DIndex(int...)
     */
    public int[][] getNdIndices(int... indices) {
        makeStrides(); // Ensure strides are initialized if not already.
        int[][] nDIndices = new int[indices.length][rank];

        for(int i=0; i<indices.length; i++) {
            int index = indices[i];

            for (int j = 0; j < strides.length; j++)
                nDIndices[i][j] = (index / strides[j]) % dims[j];
        }

        return nDIndices;
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
    public Shape swapAxes(int axis1, int axis2) {
        int[] newDims = dims.clone();
        ArrayUtils.swap(newDims, axis1, axis2);

        return new Shape(newDims);
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
    public Shape permuteAxes(int... axes) {
        ValidateParameters.ensureAllEqual(getRank(), axes.length);
        ValidateParameters.ensurePermutation(axes);

        int[] permutedDims = new int[dims.length];

        int i=0;
        for(int axis : axes)  // Permute axes.
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
     *             {@code n} is the rank of this shape.
     * @return Returns this shape.
     * @see #permuteAxes(int...) 
     * @see #swapAxes(int, int) 
     */
    public Shape unsafePermuteAxes(int... axes) {
        int[] permutedDims = new int[dims.length];

        int i=0;
        for(int axis : axes)  // Permute axes.
            permutedDims[i++] = dims[axis];

        return new Shape(permutedDims);
    }


    /**
     * Gets the total number of data for an nD array with this shape.
     * @return The total number of data for an nD array with this shape.
     * @see #totalEntriesIntValueExact()
     * @see #totalEntriesLongValueExact()
     */
    public BigInteger totalEntries() {
        // Check if totalEntries has already been computed for this shape.
        if(totalEntries!=null) return totalEntries;

        // Otherwise, the total data needs to be computed.
        BigInteger product = BigInteger.ONE;  // We can start at one because scalar tensors have a single entry.
        for(int dim : dims)
            product = product.multiply(BigInteger.valueOf(dim));
        totalEntries = product;

        return product;
    }


    /**
     * <p>Gets the total number of elements for an nD array with this shape.
     * If the total number of elements exceeds {@link Integer#MAX_VALUE}, an exception is thrown.
     *
     * <p>This method is likely to be more efficient than {@link #totalEntries()} if a primitive int value is desired.
     *
     * @return The total number of data for an nD array with this shape.
     * @throws ArithmeticException If the total number of data overflows a primitive int.
     * @see #totalEntries()
     * @see #totalEntriesLongValueExact()
     */
    public int totalEntriesIntValueExact() {
        if(totalEntriesIntExact >= 0) return totalEntriesIntExact;  // The Value has already been computed.
        long product = 1;
        totalEntriesIntExact = 1;  // We can start at one because scalar tensors have a single entry.

        for (int dim : dims) {
            product *= dim;

            // If product > Long.MAX_VALUE, then product*value would overflow since all dims are positive.
            if (product > Integer.MAX_VALUE)
                throw new ArithmeticException("Integer overflow while computing total data in shape: " + this);
        }

        totalEntriesIntExact = (int) product;

        return totalEntriesIntExact;
    }


    /**
     * <p>Gets the total number of elements for an nD array with this shape as a {@code long}.
     * If the total number of elements exceeds {@link Long#MAX_VALUE}, an exception is thrown.
     *
     * @return The total number of elements for an nD array with this shape.
     * @throws ArithmeticException If the total number of data overflows a primitive int.
     * @see #totalEntriesIntValueExact()
     * @see #totalEntries()
     */
    public long totalEntriesLongValueExact() {
        if(totalEntriesIntExact >= 0) return totalEntriesIntExact; // The Value has already been computed as an integer.

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
            totalEntriesIntExact = (int) product;

        return product;
    }


    /**
     * Checks if the total number of elements represented by this shape can be represented as a 32-bit integer without overflowing.
     * @return {@code true} if the total number of elements represented by this shape can be represented as a
     * 32-bit integer without overflowing; {@code false} if it would overflow.
     */
    public boolean isIntSized() {
        if(totalEntriesIntExact >= 0) return true; // Value is already known to be computable as an integer.
        long product = 1;

        for (int value : dims) {
            product *= value;

            if (product > Integer.MAX_VALUE)
                return false;  // Integer overflow detected.
        }

        // Update the cached value since we already computed it.
        totalEntriesIntExact = (int) product;  // This cast is safe since it will only execute if product <= Integer.MAX_VALUE.

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

        return Arrays.equals(dims, ((Shape) b).dims);
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
