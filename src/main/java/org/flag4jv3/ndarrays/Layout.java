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

import org.flag4jv3.util.ValidateParameters;
import org.flag4jv3.util.tuples.Pair;

import java.util.Arrays;
import java.util.Objects;

// TODO NOW: Finish docs.
public class Layout {

    public Shape shape;

    public final int offset;
    final int[] strides;

    public final boolean isContiguous;


    public Layout(Shape shape, int offset, int[] strides) {
        // TODO NOW: Should we also verify strides/offset against shape? We could also have a
        //  makeLayoutUnsafe that bypasses that for internal use...
        Objects.requireNonNull(shape, "shape must not be null");
        Objects.requireNonNull(strides, "strides cannot be null.");
        ValidateParameters.ensureNonNegative(offset, "offset cannot be negative.");
        ValidateParameters.ensureNonNegative(strides);

        this.shape = shape;
        this.offset = offset;
        this.strides = strides.clone();

        isContiguous = offset == 0 && Arrays.equals(strides, shape.getContiguousStridesUnsafe());
    }


    /// Factory for constructing a [Layout] of a contiguous nD array with the specified [shape][Shape].
    ///
    /// @param shape The [shape][Shape] of the nD array to get contiguous [Layout] for.
    public static Layout contiguous(Shape shape) {
        return new Layout(shape, 0, shape.getContiguousStridesUnsafe());
    }


    /// Squeezes this layout. That is, gets a [Layout] that is equivalent to this layout but with all dimensions of size 1 removed.
    ///
    /// @return If this layout is already a squeezed layout, then `this` is returned; Otherwise,
    ///                         a new layout that is a squeezed copy of this layout is constructed and returned.
    public Layout squeeze() {
        int squeezedRank = 0;

        for (int dim : shape.dims) {
            if (dim != 1) {
                squeezedRank++;
            }
        }

        if (squeezedRank == shape.rank) {
            return this;
        }

        int[] newDims = new int[squeezedRank];
        int[] newStrides = new int[squeezedRank];

        for (int src = 0, dest = 0; src < shape.rank; src++) {
            if (shape.dims[src] != 1) {
                newDims[dest] = shape.dims[src];
                newStrides[dest] = strides[src];
                dest++;
            }
        }

        return new Layout(new Shape(newDims), offset, newStrides);
    }


    /// Squeezes this layout along a single axis by removing that axis if its size is one.
    ///
    /// @param axis The axis to squeeze.
    /// @return This layout if the specified axis does not have size one; otherwise,
    /// a new equivalent layout with the specified singleton axis removed.
    ///
    /// @throws IllegalArgumentException If {@code axis} is invalid for this layout's shape.
    /// @see #squeeze()
    /// @see #squeeze(int...)
    public Layout squeeze(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);

        if (shape.dims[axis] != 1) {
            return this;
        }

        int[] newDims = new int[shape.rank - 1];
        int[] newStrides = new int[shape.rank - 1];

        for (int src = 0, dest = 0; src < shape.rank; src++) {
            if (src != axis) {
                newDims[dest] = shape.dims[src];
                newStrides[dest] = strides[src];
                dest++;
            }
        }

        return new Layout(new Shape(newDims), offset, newStrides);
    }


    /// Squeezes this layout along the specified axes, removing each requested axis
    /// whose size is one.
    ///
    /// @param axes The axes to squeeze.
    /// @return This layout if none of the specified axes has size one; otherwise,
    /// a new equivalent layout with the requested singleton axes removed.
    ///
    /// @throws IllegalArgumentException If any axis is invalid for this layout's shape.
    /// @see #squeeze()
    /// @see #squeeze(int)
    public Layout squeeze(int... axes) {
        ValidateParameters.ensureValidAxes(shape, axes);

        boolean[] squeezeAxes = new boolean[shape.rank];
        int squeezedRank = shape.rank;

        for (int axis : axes) {
            if (!squeezeAxes[axis]) {
                squeezeAxes[axis] = true;

                if (shape.dims[axis] == 1) {
                    squeezedRank--;
                }
            }
        }

        if (squeezedRank == shape.rank) {
            return this;
        }

        int[] newDims = new int[squeezedRank];
        int[] newStrides = new int[squeezedRank];

        for (int src = 0, dest = 0; src < shape.rank; src++) {
            if (!squeezeAxes[src] || shape.dims[src] != 1) {
                newDims[dest] = shape.dims[src];
                newStrides[dest] = strides[src];
                dest++;
            }
        }

        return new Layout(new Shape(newDims), offset, newStrides);
    }


    /// Gets the strides of this [Layout].
    ///
    /// This will be a copy of the strides so modifying
    /// the returned strides will have no effect on this layout.
    ///
    /// @return The strides of this [Layout].
    public int[] strides() {
        return strides.clone();
    }


    /// Gets the initial offset of this [Layout].
    ///
    /// @return The initial offset of this [Layout].
    public int offset() {
        return offset;
    }


    /**
     * Converts an nD index to the 1D index of the items buffer of an nD array with this layout.
     *
     * @param idxND The ND index to convert.
     * @return The 1D index of a nD array's items buffer corresponding to the nD index {idxND}.
     */
    public int toBufferIndex(int... idxND) {
        int p = offset;
        for (int a = 0; a < idxND.length; a++) {
            p += idxND[a]*strides[a];
        }
        return p;
    }


    /// Broadcasts a pair of layouts together into a new pair of layouts with the same shape.
    ///
    /// The returned layouts share each operand's original offset. Strides for missing
    /// leading axes are expanded and set to 1.
    ///
    /// Two layouts can be broadcast together if starting from the rightLayout-most dimension of each layout's shape,
    /// 1. they are equal, or
    /// 2. one of them is 1.
    ///
    /// @param left The layout of the leftLayout operand.
    /// @param right The layout of the rightLayout operand.
    /// @return A [Pair] containing the broadcasted and layouts of `left` and `right`.
    ///
    /// @throws IllegalArgumentException If `leftShape` and `rightShape` cannot be broadcast together.
    public static Pair<Layout, Layout> broadcast(Layout left, Layout right) {
        if (left.shape.equals(right.shape)) {
            return new Pair<>(left, right);
        }

        // TODO NOW: add a broadcast to the Shape class that just broadcasts two shapes.
        //  This would be useful because it is a bit simpler and would be a nice feature to have
        //  for users to understand broadcasting without the layout semantics.

        int broadcastRank = Math.max(left.shape.rank, right.shape.rank);
        int[] broadcastDims = new int[broadcastRank];
        int[] newLeftStrides = new int[broadcastRank];
        int[] newRightStrides = new int[broadcastRank];

        int leftAxis = left.shape.rank - 1;
        int rightAxis = right.shape.rank - 1;

        for (int broadcastAxis = broadcastRank - 1; broadcastAxis >= 0;
             broadcastAxis--, leftAxis--, rightAxis--) {
            int leftDim = (leftAxis >= 0) ? left.shape.dims[leftAxis] : 1;
            int rightDim = (rightAxis >= 0) ? right.shape.dims[rightAxis] : 1;

            int leftStride = leftAxis >= 0 ? left.strides[leftAxis] : 0;
            int rightStride = rightAxis >= 0 ? right.strides[rightAxis] : 0;

            if (leftDim == rightDim) {
                broadcastDims[broadcastAxis] = leftDim;
                newLeftStrides[broadcastAxis] = leftStride;
                newRightStrides[broadcastAxis] = rightStride;
            } else if (leftDim == 1) {
                broadcastDims[broadcastAxis] = rightDim;
                newLeftStrides[broadcastAxis] = 0; // Zero-stride means repeat value across axis.
                newRightStrides[broadcastAxis] = rightStride;
            } else if (rightDim == 1) {
                broadcastDims[broadcastAxis] = leftDim;
                newLeftStrides[broadcastAxis] = leftStride;
                newRightStrides[broadcastAxis] = 0; // Zero-stride means repeat value across axis.
            } else {
                throw new IllegalArgumentException("Cannot broadcast shapes " + left.shape + " and " + right.shape);
            }
        }

        Shape broadcastShape = new Shape(broadcastDims);

        return new Pair<>(
                new Layout(broadcastShape, left.offset, newLeftStrides),
                new Layout(broadcastShape, right.offset, newRightStrides)
        );
    }


    @Override
    public boolean equals(Object obj) {
        return obj instanceof Layout other
                && shape.equals(other.shape)
                && offset == other.offset
                && Arrays.equals(strides, other.strides);
    }


    @Override
    public int hashCode() {
        int hash = 31*Integer.hashCode(offset) + Arrays.hashCode(strides);
        hash = 31*hash + shape.hashCode();
        return hash;
    }
}
