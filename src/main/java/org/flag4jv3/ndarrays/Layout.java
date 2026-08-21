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

import org.flag4jv3.util.NewValidateParameters;
import org.flag4jv3.util.ValidateParameters;
import org.flag4jv3.util.tuples.IntPair;
import org.flag4jv3.util.tuples.Pair;

import java.io.Serial;
import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Objects;
import java.util.StringJoiner;

import static org.flag4jv3.util.NewValidateParameters.Sign;


/// Describes the memory layout of a [dense nD-array][org.flag4jv3.ndarrays.dense.DenseNDArrayBase].
/// Specifically, how the logical elements of a [dense nD-array][org.flag4jv3.ndarrays.dense.DenseNDArrayBase] map onto
/// positions in its 1D backing data buffer.
///
/// A layout is defined by:
/// - [shape][#shape()]: the size of the array along each axis.
/// - [strides][#strides()]: the step, per axis, between consecutive logical elements.
/// - [offset][#offset()]: the logical-element slice of the first element.
/// - [itemSize][#itemSize()]: the number of buffer positions a single logical element occupies.
///
/// ## Element units vs. buffer positions
/// Both `strides` and `offset` are expressed in **logical elements**, *not* raw buffer positions. This keeps the layout
/// independent of how an element is stored: a real, complex, or quaternion array with the same shape shares identical
/// strides and differs only in [itemSize][#itemSize()] (`1`, `2`, and `4` respectively). The `itemSize` factor is
/// applied only when resolving a buffer position via [#toBufferIndex(int, int...)]; the logical-element slice alone is
/// given by [#toLinearElementIndex(int...)].
///
/// For example, a complex array stores each element as interleaved real/imaginary doubles ({@code itemSize == 2}), so
/// logical element `i` occupies buffer positions `2i` (real) and `2i + 1` (imaginary).
///
/// ## Views
/// A layout need not solely own its buffer or represent the full buffer. A non-zero [offset][#offset()], non-canonical
/// [strides][#strides()] (e.g., zero strides, negative strides, etc.) let a layout describe a *view* into a larger or shared buffer.
/// Use [#contiguousOrder()] to test for a dense, canonically strided layout.
///
/// ### Non-Writable Views
/// Views *cannot* be [writable][#isWritable()]. Layouts that are not writable indicate that an nD-array with that layout
/// *cannot* be safely used as the output of an operation. For example, a layout with a zero stride means that all indices along
/// the associated axis point to the same slice in the buffer (may happen when extending, repeating, or broadcasting an nD-array).
/// Thus, attempting to write the outputs of an operation into such a layout would result in values clobbering past results.
///
/// The [#isWritable()] property of a layout is determined at construction time using [#mayBeInjective()] to conservatively test
/// whether distinct logical elements may alias the same buffer slice.
///
/// Instances of [Layout] are immutable and thread-safe.
public class Layout implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /// The shape of this layout.
    final Shape shape;

    /// The initial offset of this layout. This is in terms of logical elements
    /// (i.e., individual elements of [itemSize]) and *not* in terms of buffer indices.
    final int offset;

    /// The strides of this layout. This is in terms of logical elements
    /// (i.e., individual elements of [itemSize]) and *not* in terms of buffer indices.
    final int[] strides;

    /// Flag indicating if this layout is contiguous in memory (`true`) or not (`false`).
    final ContiguousOrder contiguousOrder;

    // TODO NOW: The item size will also matter for sparse arrays. `Layout`s are only intended for dense arrays.
    //  It is needed here, but it should also be in the base nD-array object (similar to shape).
    /// The size of an individual item of this layout. That is, the number of buffer positions needed to store a single element.
    /// E.g., a complex number with real/imaginay components stored interleaved in a `double[]` buffer would have `itemSize = 2`.
    final int itemSize;


    /// Flag indicating if this layout is writable or not. A layout is writable when it can be verified that it is injective.
    /// This means all indices map to distinct elements within the buffer. This makes it safe to write values to the nD-array.
    ///
    /// This is a "soft" property derived from [StrideInjectivity#mayBeInjective(org.flag4jv3.ndarrays.Layout)]. It may be
    /// `false` even if the layout *is* injective in reality. However, checking if an arbitrary layout is injective is worst-case
    /// exponential so an exhaustive check is not performed. However, this will *never* be erroneously be `true`. If this
    /// is `true`, the layout is provably injective and thus safe to write to.
    final boolean isWritable;


    /// Creates a [Layout] of an nD-array.
    ///
    /// To create a contiguous layout, use [#contiguous(Shape, int)].
    ///
    /// @param shape The shape of the layout. If `shape.numel()` *must* fit in a primitive `int` (see [Shape#numelIntValueExact()]).
    /// @param offset The initial offset of the first item in the layout. Must be non-negative.
    /// @param strides The strides of the layout. That is, the number of buffer positions needed to store a single element.
    /// E.g., a complex number with real/imaginay components stored interleaved in a `double[]` buffer would have `itemSize = 2`.
    /// Must be positive.
    /// @param itemSize The size of an individual item of this layout.
    /// @return A [Layout] of an nD-array with the specified properties.
    ///
    /// @throws NullPointerException     If `shape` or `strides` is `null`.
    /// @throws IllegalArgumentException If `offset` is negative or if `itemSize` is *not* positive.
    /// @throws ArithmeticException      If `shape.numel()` is *not* int sized.
    /// @see #contiguous(Shape, int)
    public Layout(Shape shape, int offset, int[] strides, int itemSize) {
        // TODO NOW: Should we also verify strides/offset against shape? We could also have a
        //  makeLayoutUnsafe that bypasses that for internal use...
        Objects.requireNonNull(shape, "shape must not be null");
        Objects.requireNonNull(strides, "strides cannot be null.");
        NewValidateParameters.ensureSign(offset, Sign.NON_NEGATIVE, "offset cannot be negative.");
        NewValidateParameters.ensureSign(itemSize, Sign.POSITIVE,
                "item size must be positive but got " + itemSize + "."
        );
        NewValidateParameters.ensureEqual(strides.length, shape.rank, "Stride rank must match shape rank.");

        // Layouts require that all elements must fit inside a standard array.
        shape.numelIntValueExact(); // Throws ArithmeticExceptio if the shape can't represent it's number of elements as an int.

        this.shape = shape;
        this.offset = offset;
        this.strides = strides.clone();
        this.itemSize = itemSize;

        contiguousOrder = getContiguousOrder(this.shape, this.strides);

        // This will miss some cases, but we accept that for safety.
        isWritable = StrideInjectivity.mayBeInjective(this);
    }


    /// Constructs a layout where the [#contiguousOrder] and [#isWritable] flags are specified.
    ///
    /// <blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    ///     <strong>Warning:</strong> Using this constructor is dangerous. *only* use this constructor if you are <em>certain</em>
    ///     about the correctness of the two flags. Further, <em>no</em> input validation is performed. Unlike the public constructor,
    ///     this constructor <em>does not</em> clone the {@code strides} which is dangerous.
    /// </blockquote>
    ///
    /// @param shape The shape of the layout.
    /// @param offset The initial offset of the first item in the layout. Must be non-negative.
    /// @param strides The strides of the layout. That is, the number of buffer positions needed to store a single element.
    /// @param itemSize The size of an individual item of this layout (in terms of buffer indices).
    /// For example, a complex number with real/imaginay components stored interleaved in a `double[]` buffer would have `itemSize = 2`.
    /// Must be positive.
    /// @param knownContiguous Indicates if the layout is already known to be contiguous and what its contiguous ordering is.
    /// `null` indicates the contiguous ordering is not known and so it will be computed.
    /// @param isWritable Flag indicating if the layout [is writable][#isWritable()] or not.
    /// @return A [Layout] of an [nD-array][org.flag4jv3.ndarrays.dense.DenseNDArrayBase] with the specified attributes
    /// and properties.
    private Layout(Shape shape, int offset, int[] strides, int itemSize, ContiguousOrder knownContiguous, boolean isWritable) {
        if (strides.length != shape.rank) {
            throw new IllegalArgumentException(
                    "Stride rank must match shape rank.");
        }

        this.shape = shape;
        this.offset = offset;
        this.strides = strides;
        this.itemSize = itemSize;
        this.contiguousOrder = (knownContiguous != null) ? knownContiguous : getContiguousOrder(shape, strides);
        this.isWritable = isWritable;
    }


    /// Factory for constructing a [Layout] of a [C-contiguous][ContiguousOrder#C] nD-array with the specified [shape][Shape].
    ///
    /// To construct a contiguous array with [F-ordering][ContiguousOrder#F], use [#contiguous(Shape, int, ContiguousOrder)]
    ///
    /// @param shape The [shape][Shape] of the nD-array to get contiguous [Layout] for.
    /// @param itemSize The item size of the [Layout]. This is the number of positions in the buffer that an individual item
    /// takes up. E.g., for a complex number whose real/imaginary components are stored interleaved in a `double[]` buffer, this
    /// would be `2`.
    /// @return A layout that is contiguous with the specified `shape` and `itemSize`.
    ///
    /// @see #contiguous(Shape, int, ContiguousOrder)
    public static Layout contiguous(Shape shape, int itemSize) {
        return contiguous(shape, itemSize, ContiguousOrder.C);
    }


    /// Factory for constructing a [Layout] of a contiguous nD-array with the specified [shape][Shape].
    ///
    /// @param shape The [shape][Shape] of the nD-array to get contiguous [Layout] for.
    /// @param itemSize The item size of the [Layout]. This is the number of positions in the buffer that an individual item
    /// takes up. E.g., for a complex number whose real/imaginary components are stored interleaved in a `double[]` buffer, this
    /// would be `2`.
    /// @param order The ordering of the contiguous array. Must be either [ContiguousOrder#C] or [ContiguousOrder#F]
    /// @see #contiguous(Shape, int)
    public static Layout contiguous(Shape shape, int itemSize, ContiguousOrder order) {
        ContiguousOrder.ensureCorFExact(order);

        // Despite an order being specified, we still want to check if it is BOTH so pass `null` for `knownContiguous`
        return new Layout(shape, 0, shape.getContiguousStrides(order), itemSize, null, true);
    }


    /// Gets a C-contiguous [Layout] which has the same [shape][#shape()] and [item size][#itemSize()] as `this` [Layout].
    ///
    /// To specify an F-contiguous layout, use [#asContiguous()].
    ///
    /// @return A C-contiguous [Layout] that is equivalent to `this` [Layout].
    ///
    /// @see #asContiguous(ContiguousOrder)
    public Layout asContiguous() {
        return contiguous(shape, itemSize);
    }


    /// Gets a contiguous [Layout] with the specified `order` and which has the same
    ///  [shape][#shape()] and [item size][#itemSize()] as `this` [Layout].
    ///
    /// @return A contiguous [Layout] with the specified `order` that is equivalent to `this` [Layout].
    ///
    /// @see #asContiguous()
    public Layout asContiguous(ContiguousOrder order) {
        return contiguous(shape, itemSize, order);
    }


    /// Gets the full buffer size required to store all elements of an nD-array with `this` layout.
    ///
    /// @return The full buffer size required to store all elements of an nD-array with `this` layout.
    ///
    /// @see #bufferSizeIntValueExact()
    public BigInteger bufferSize() {
        return shape.numel().multiply(BigInteger.valueOf(itemSize));
    }


    /// Gets the full buffer size (as an `int`) required to store all elements of an nD-array with `this` layout.
    ///
    /// @return The full buffer size (as an `int`) required to store all elements of an nD-array with `this` layout.
    ///
    /// @throws ArithmeticException If the number of an nD-array with `this` layout does not exactly fit in an `int`.
    /// @see #bufferSize()
    public int bufferSizeIntValueExact() {
        return Math.multiplyExact(shape.numel().intValueExact(), itemSize);
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

        if (squeezedRank == shape.rank()) {
            return this;
        }

        int[] newDims = new int[squeezedRank];
        int[] newStrides = new int[squeezedRank];

        for (int src = 0, dest = 0; src < shape.rank(); src++) {
            if (shape.dims[src] != 1) {
                newDims[dest] = shape.dims[src];
                newStrides[dest] = strides[src];
                dest++;
            }
        }

        return new Layout(new Shape(newDims), offset, newStrides, itemSize, null, isWritable);
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

        int[] newDims = new int[shape.rank() - 1];
        int[] newStrides = new int[shape.rank() - 1];

        for (int src = 0, dest = 0; src < shape.rank(); src++) {
            if (src != axis) {
                newDims[dest] = shape.dims[src];
                newStrides[dest] = strides[src];
                dest++;
            }
        }

        return new Layout(new Shape(newDims), offset, newStrides, itemSize, null, isWritable);
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

        boolean[] squeezeAxes = new boolean[shape.rank()];
        int squeezedRank = shape.rank();

        for (int axis : axes) {
            if (!squeezeAxes[axis]) {
                squeezeAxes[axis] = true;

                if (shape.dims[axis] == 1) {
                    squeezedRank--;
                }
            }
        }

        if (squeezedRank == shape.rank()) {
            return this;
        }

        int[] newDims = new int[squeezedRank];
        int[] newStrides = new int[squeezedRank];

        for (int src = 0, dest = 0; src < shape.rank(); src++) {
            if (!squeezeAxes[src] || shape.dims[src] != 1) {
                newDims[dest] = shape.dims[src];
                newStrides[dest] = strides[src];
                dest++;
            }
        }

        return new Layout(new Shape(newDims), offset, newStrides, itemSize, null, isWritable);
    }


    /// Gets the strides of this [Layout].
    ///
    /// This will be a copy of the strides. Modifying
    /// the returned strides will have *no effect* on this layout.
    /// To avoid internal copying of strides, use [#stride(int)].
    ///
    /// @return The strides of this [Layout].
    ///
    /// @see #stride(int)
    public final int[] strides() {
        return strides.clone();
    }


    /// Gets the stride of this [Layout] at the specified dimension.
    ///
    /// @param dim The dimension to get the stride of.
    /// @return The stride of this layout at `dim`.
    ///
    /// @see #strides()
    public final int stride(int dim) {
        return strides[dim];
    }


    /// Gets the size of a dimension of this layout. This is the number of logical elements along a specific dimension.
    /// Equivalent to `layout.shape().getSize(dim)`.
    ///
    /// @param dim The dimension to get the extent of. Must be non-negative.
    /// @return The extent of this layout at `dim`.
    public final int getSize(int dim) {
        return shape.dims[dim];
    }


    /// Gets the rank of this layout.
    ///
    /// Equivalent to `layout.shape().rank()`.
    ///
    /// @return The rank of this layout.
    public final int rank() {
        return shape.rank;
    }


    /// Gets the initial offset of this [Layout].
    ///
    /// @return The initial offset of this [Layout].
    public final int offset() {
        return offset;
    }


    /// Gets the shape of this [Layout].
    ///
    /// @return The shape of this [Layout].
    public final Shape shape() {
        return shape;
    }


    /// The size of an individual item of this [Layout].
    /// That is, the number of buffer positions needed to store a single element.
    public final int itemSize() {
        return itemSize;
    }


    /// Checks if two layouts are *both* contiguous *and* their [ordering][#contiguousOrder()] match.
    ///
    /// @param layout1 The first layout to compare.
    /// @param layout2 The second layout to compare.
    /// @return `true` if `layout1` and `layout2` are *both* contiguous and their [ordering][#contiguousOrder()] match.
    public static final boolean areContiguousAndMatchOrder(Layout layout1, Layout layout2) {
        return layout1.isContiguous() && layout2.isContiguous()
                && layout1.contiguousOrder() == layout2.contiguousOrder();
    }


    /// Gets the [contiguous ordering][ContiguousOrder] of this layout.
    ///
    /// @return The [contiguous ordering][ContiguousOrder] of this layout.
    ///
    /// @see #isContiguous()
    /// @see #isFContiguous()
    /// @see #isContiguous()
    public final ContiguousOrder contiguousOrder() {
        return contiguousOrder;
    }


    /// Checks if this layout is [C-contiguous][ContiguousOrder#C]
    ///
    /// @return `true` if this layout is [C-contiguous][ContiguousOrder#C]; otherwise `false`.
    ///
    /// @see #contiguousOrder()
    /// @see #isFContiguous()
    /// @see #isContiguous()
    public final boolean isCContiguous() {
        return contiguousOrder == ContiguousOrder.C
                || contiguousOrder == ContiguousOrder.BOTH;
    }


    /// Checks if this layout is [F-contiguous][ContiguousOrder#F]
    ///
    /// @return `true` if this layout is [F-contiguous][ContiguousOrder#F]; otherwise `false`.
    ///
    /// @see #contiguousOrder()
    /// @see #isCContiguous()
    /// @see #isContiguous()
    public final boolean isFContiguous() {
        return contiguousOrder == ContiguousOrder.F
                || contiguousOrder == ContiguousOrder.BOTH;
    }


    /// Checks if this layout is [contiguous][ContiguousOrder] (either [C][ContiguousOrder#C], [F][ContiguousOrder#F],
    /// or [both][ContiguousOrder#BOTH])
    ///
    /// @return `true` if this layout is [contiguous][ContiguousOrder]; otherwise `false`.
    ///
    /// @see #contiguousOrder()
    /// @see #isFContiguous()
    /// @see #isCContiguous()
    public final boolean isContiguous() {
        return contiguousOrder != ContiguousOrder.NONE;
    }


    /// Checks if this layout is contiguous *and* has the specified `order`.
    ///
    /// @param order The order to check for. Must be [C][ContiguousOrder#C] or [C][ContiguousOrder#F].
    /// @return `true` if this layout is contiguous *and* has the specified `order`.
    public final boolean isContiguous(ContiguousOrder order) {
        ContiguousOrder.ensureCorFExact(order);
        return isContiguous() && order == contiguousOrder;
    }


    /// Checks if this layout may be written to.
    ///
    /// @return `true` if this layout may be written to; otherwise, `false`.
    public final boolean isWritable() {
        return isWritable;
    }


    /// Helper for initial check for contiguous layout.
    ///
    /// @param shape The shape of the layout.
    /// @param strides The strides of the layout.
    /// @return The order of the contiguity (possibly [none][ContiguousOrder#NONE]).
    private static ContiguousOrder getContiguousOrder(Shape shape, int[] strides) {
        int rank = shape.rank();

        if (rank == 0) return ContiguousOrder.BOTH; // Scalars are trivially both C and F contiguous.

        // Do a quick scan for empty array.
        for (int ax = 0; ax < rank; ax++)
            if (shape.getSize(ax) == 0) return ContiguousOrder.BOTH; // Empty array trivially both C and F contiguous.

        boolean cContiguous = true;
        boolean fContiguous = true;

        long expectedCStride = 1;
        long expectedFStride = 1;

        for (int i = 0; i < rank; i++) {
            int fAxis = i;
            int cAxis = rank - 1 - i;

            int fSize = shape.getSize(fAxis);
            int cSize = shape.getSize(cAxis);

            if (fContiguous && fSize > 1) {
                if (strides[fAxis] != expectedFStride) {
                    fContiguous = false;
                } else {
                    expectedFStride *= fSize;
                }
            }

            if (cContiguous && cSize > 1) {
                if (strides[cAxis] != expectedCStride) {
                    cContiguous = false;
                } else {
                    expectedCStride *= cSize;
                }
            }

            if (!cContiguous && !fContiguous) {
                return ContiguousOrder.NONE;
            }
        }

        if (cContiguous && fContiguous) {
            return ContiguousOrder.BOTH;
        } else if (cContiguous) {
            return ContiguousOrder.C;
        } else {
            return ContiguousOrder.F;
        }
    }


    /// Checks if this layout possibly represents overlapping locations in memory (e.g, zero strides).
    ///
    /// <blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    ///     <strong>Note:</strong> This method does not exhaustive check for all possible overlaps as that would have worst-case
    ///     exponential time. As such, this method <em>may</em> erroneously return {@code true} even if there is no overlap.
    ///     However, it will <em>never</em> mistakenly return {@code false}.
    /// </blockquote>
    ///
    /// @return `false` if it can be determined that *absolutely no* elements overlap in memory; otherwise, `true`.
    public boolean mayBeInjective() {
        return StrideInjectivity.mayBeInjective(this);
    }


    /// Converts an nD slice and a component slice to a position in the raw data buffer of an nD-array with `this` layout.
    ///
    /// A single logical element occupies [itemSize()][#itemSize()] consecutive buffer positions. The `componentIdx`
    /// selects one of them: `toBufferIndex(0, idxND)` is the first position of the element at `idxND`, and
    /// `toBufferIndex(itemSize() - 1, idxND)` is its last. For example, a complex element stored as interleaved
    /// real/imaginary doubles (`itemSize == 2`) has its real part at component slice `0` and imaginary part at
    /// component slice `1`.
    ///
    /// @param componentIdx The component within the logical element. Must be in range `[0, itemSize())`.
    /// @param idxND The nD slice of the logical element.
    /// @return The buffer position of the specified component of the element at `idxND`.
    ///
    /// @see #toLinearElementIndex(int...)
    public int toBufferIndex(int componentIdx, int... idxND) {
        Objects.checkIndex(componentIdx, itemSize);
        return itemSize*toLinearElementIndex(idxND) + componentIdx;
    }


    /// Converts an nD slice to a flat logical-element slice (in element units, ignoring [itemSize()][#itemSize()]).
    ///
    /// This is the element position; use [#toBufferIndex(int, int...)] to get a position within the raw data buffer.
    ///
    /// @param idxND The nD slice of the logical element.
    /// @return The flat logical-element slice of the element at `idxND`.
    ///
    /// @see #toBufferIndex(int, int...)
    public int toLinearElementIndex(int... idxND) {
        if (idxND.length != shape.rank) {
            throw new IllegalArgumentException("idxND must have equal length to the rank of the layouts shape but got:" +
                    "length=" + idxND.length + " for rank" + shape.rank + ".");
        }

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
    /// Two layouts can be broadcast together if starting from the right-most dimension of each layout's shape,
    /// 1. they are equal, or
    /// 2. one of them is 1.
    ///
    /// @param left The layout of the leftLayout operand.
    /// @param right The layout of the rightLayout operand.
    /// @return A [Pair] containing the broadcasted and layouts of `left` and `right`.
    ///
    /// @throws IllegalArgumentException If `left.shape()` and `right.shape()` cannot be broadcast together.
    public static Pair<Layout, Layout> broadcast(Layout left, Layout right) {
        if (left.shape.equals(right.shape)) {
            return new Pair<>(left, right);
        }

        // TODO NOW: add a broadcast to the Shape class that just broadcasts two shapes.
        //  This would be useful because it is a bit simpler and would be a nice feature to have
        //  for users to understand broadcasting without the layout semantics.

        int broadcastRank = Math.max(left.shape.rank(), right.shape.rank());
        int[] broadcastDims = new int[broadcastRank];
        int[] newLeftStrides = new int[broadcastRank];
        int[] newRightStrides = new int[broadcastRank];

        int leftAxis = left.shape.rank() - 1;
        int rightAxis = right.shape.rank() - 1;

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
                new Layout(broadcastShape, left.offset, newLeftStrides, left.itemSize()),
                new Layout(broadcastShape, right.offset, newRightStrides, right.itemSize())
        );
    }


    /// Computes the minimum and maximum reachable indices in a data buffer with this layout.
    ///
    /// @return A [pair][IntPair] containing, in order, the minimum and maximum reachable indices in a data buffer with
    /// this layout.
    public IntPair minMaxIndex() {
        int min = offset;
        int max = min;

        for (int axis = 0, rank = shape.rank; axis < rank; axis++) {
            int extent = strides[axis]*(shape.dims[axis] - 1);

            if (extent < 0) {
                min += extent;
            } else {
                max += extent;
            }
        }

        // itemSize*max will be starting position of final item; itemSize*(max + 1) will be one after last slice, hence
        // itemSize*(max + 1) - 1 will be the final slice in the buffer.
        return new IntPair(min*itemSize, itemSize*(max + 1) - 1);
    }


    /// Checks if this layout is equal to another object. A layout is equal to another object if *all* the following are true:
    /// - The object is an instance of the [Layout] class (referred to as "other layout").
    /// - The other layout has the same [shape][#shape()].
    /// - The other layout has the same [offset][#offset()].
    /// - The other layout has the same [strides][#strides()].
    /// - The other layout has the same [item size][#itemSize()].
    ///
    /// @param object The reference object with which to compare.
    /// @return `true` if `object` is equal to this layout as defined above; otherwise `false`.
    @Override
    public boolean equals(Object object) {
        if (object == this) return true;

        return object instanceof Layout other
                && itemSize == other.itemSize
                && shape.equals(other.shape)
                && offset == other.offset
                && Arrays.equals(strides, other.strides);
    }


    /// Computes the hashcode for this layout. This satisfies the contract with [#equals]. That is, any two [Layout]s that
    /// are equal, according to [#equals], will also have the same hash code.
    ///
    /// @return A hashcode for this layout that is *very* unlikely to be the same for any other layout *not*
    ///  [equal][#equals] to this layout.
    @Override
    public int hashCode() {
        int hash = 31*Integer.hashCode(offset) + Arrays.hashCode(strides);
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Integer.hashCode(itemSize);
        return hash;
    }


    /**
     * Converts this [Layout] object to a human-readable string format.
     *
     * @return The string representation for this [Layout] object.
     */
    public String toString() {
        StringJoiner joiner = new StringJoiner("\n\t", "Layout: (\n\t", "\n)");

        joiner.add("Shape: " + shape);
        joiner.add("strides: " + Arrays.toString(strides));
        joiner.add("offset: " + offset);

        return joiner.toString();
    }


    /// Ensures that `this` layout is [non-overlapping][Layout#mayBeInjective()].
    ///
    /// @return A reference to `this` layout.
    ///
    /// @throws IllegalArgumentException If it could not be determined that `this` layout is non-overlapping.
    public Layout requireNonOverlapping() {
        if (mayBeInjective()) {
            throw new IllegalArgumentException("layout may have overlap in memory. Try making layout contiguous first.");
        }

        return this;
    }


    /// Derives a view of this layout by applying a multi-axis slicing operation, i.e., an index expression.
    ///
    /// The entries of `indexExpr` are matched left to right against the axes of this layout. Each entry either consumes a source
    /// axis, emits an output axis, or both:
    ///
    /// | Variant          | Source axes consumed | Out axes emitted | Effect                                                                             |
    /// |------------------|----------------------|------------------|------------------------------------------------------------------------------------|
    /// | [Slice.Point]    | &#x2714;             | &#x2718;         | Selects one position; the axis is dropped.                                         |
    /// | [Slice.Range]    | &#x2714;             | &#x2714;         | Strided sub-range; the axis is kept, possibly smaller.                             |
    /// | [Slice#ALL]      | &#x2714;             | &#x2714;         | The whole axis; equivalent to [range(0, n, 1)][Slice#range(Integer, Integer, int)].|
    /// | [Slice#NEW_AXIS] | &#x2718;             | &#x2714;         | Inserts a new axis of extent `1`.                                                  |
    /// | [Slice#ELLIPSIS] | *k*                  | *k*              | Expands to as many [Slice#ALL] as needed to cover remaining axes.                  |
    ///
    /// Source axes not consumed by `indexExpr` are implicitly [Slice#ALL] and are appended to the result.
    /// E.g., `indexExpr = [slice(point(i))]` on layout with  shape `(4, 5, 6)` yields a layout with shape `(5, 6)`.
    ///
    ///
    /// <blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
    ///     <strong>Examples:</strong> The resulting layout shape is shown in the comment on the right hand side.
    /// {@snippet :
    /// Layout a = Layout.contiguous(new Shape(4, 5, 6), 1);
    ///
    /// a.slice(point(1));                        // (5, 6)
    /// a.slice(ALL, point(0));                   // (4, 6)
    /// a.slice(range(1, 3));                     // (2, 5, 6)
    /// a.slice(ELLIPSIS, point(0));              // (4, 5)
    /// a.slice(NEW_AXIS);                        // (1, 4, 5, 6)
    /// a.slice(range(3, 0, -1), ALL, point(2));  // (3, 5)
    ///}
    /// </blockquote>
    ///
    /// @param indexExpr The index expression to apply. May be empty, in which case a layout equal
    /// to this one is returned.
    /// @return A layout describing the specified sliced view of this layout.
    ///
    /// @throws NullPointerException      If `indexExpr` or any of its entries is `null`.
    /// @throws IllegalArgumentException  If `indexExpr` contains more than one [Slice#ELLIPSIS], or
    /// consumes more axes than this layout has.
    /// @throws IndexOutOfBoundsException If a [Slice.Point] does not resolve into its axis, or a
    /// [Slice.AxisRange] start does not resolve into its axis.
    /// @see Slice
    public Layout slice(Slice... indexExpr) {
        int rank = rank();
        var arity = SliceSupport.arity(indexExpr, rank);
        int[] outDims = new int[arity.outRank()];
        int[] outStrides = new int[arity.outRank()];
        int outOffset = offset();
        int srcAx = 0, outAx = 0; // source and output axes.

        for (Slice ix : indexExpr) {
            switch (ix) {
                case Slice.Point p -> {
                    outOffset += p.resolve(getSize(srcAx))*strides[srcAx];
                    srcAx++;
                }
                case Slice.AxisRange ar -> {
                    var rr = ar.resolve(getSize(srcAx));
                    outOffset += rr.start()*strides[srcAx];
                    outDims[outAx] = rr.extent();
                    outStrides[outAx] = rr.step()*strides[srcAx];
                    srcAx++;
                    outAx++;
                }
                case Slice.Marker.ALL -> {
                    outDims[outAx] = getSize(srcAx);
                    outStrides[outAx] = strides[srcAx];
                    srcAx++;
                    outAx++;
                }
                case Slice.Marker.NEW_AXIS -> {
                    outDims[outAx] = 1;
                    /* Extent-1 axes are traversed once, so stride is irrelevant to traversal,
                    contiguity classification, and overlap analysis. 1 is chosen arbitrarily. */
                    outStrides[outAx] = 1;
                    outAx++;
                }
                case Slice.Marker.ELLIPSIS -> {
                    for (int k = 0; k < arity.uncovered(); k++) {
                        outDims[outAx] = getSize(srcAx);
                        outStrides[outAx] = strides[srcAx];
                        srcAx++;
                        outAx++;
                    }
                }
            }
        }

        // Tail pass for unspecified axes. Missing axes are implicitly `ALL`. (i.e., a trailing `ELLIPSIS`) is tacked on.
        while (srcAx < rank) {
            outDims[outAx] = getSize(srcAx);
            outStrides[outAx] = strides[srcAx];
            srcAx++;
            outAx++;
        }

        return new Layout(new Shape(outDims), outOffset, outStrides, itemSize(), null, isWritable);
    }
}
