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

package org.flag4jv3.linalg.internal.kernels.dense;

import org.flag4jv3.linalg.internal.kernels.DenseKernelSupport;
import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.dense.DenseData;
import org.flag4jv3.ndarrays.dense.DenseDoubleData;
import org.flag4jv3.util.arrays.ArrayBuilder;

import static java.lang.System.arraycopy;
import static org.flag4jv3.linalg.internal.kernels.DenseKernelSupport.coalesceCOrder;

/// Utility class for copying data from one dense data container to another.
public final class DenseCopy {
    private DenseCopy() {
        // Hide default constructor for utility class.
    }


    /// Copies the logical elements of one dense data container to another.
    ///
    /// Both `src` and `out` must have the same [item size][Layout#itemSize()], have the same [shape][Layout#shape()].
    /// `out` must be [writable][Layout#isWritable()].
    ///
    /// If `src` and `out` have the same [Layout] and point to the same buffer (in terms of identity equality), then
    /// no work will be performed.
    ///
    /// @param src The source data container to copy elements from.
    /// @param out The designation to copy elements to.
    public static void copyTo(DenseData src, DenseData out) {
        if (src.buffer() == out.buffer() && src.layout().equals(out.layout())) return; // Equivalent, no work to do.

        var srcShape = src.layout().shape();
        var outShape = out.layout().shape();

        srcShape.requireEqual(outShape);
        DenseKernelSupport.requireItemSize(src.layout().itemSize(), out.layout().itemSize());
        DenseKernelSupport.requireWritable(out.layout());

        if (Layout.areContiguousAndMatchOrder(src.layout(), out.layout())) {
            copyToContiguous(src.layout(), src.buffer(), out.layout(), out.buffer());
        } else {
            copyToStrided(src.layout(), src.buffer(), out.layout(), out.buffer());
        }
    }


    /// Copies the logical elements of one dense data container to another.
    ///
    /// Both `src` and `out` must have the same [item size][Layout#itemSize()], have the same [shape][Layout#shape()].
    /// `out` must be [writable][Layout#isWritable()].
    ///
    /// If `src` and `out` have the same [Layout] and point to the same buffer (in terms of identity equality), then
    /// no work will be performed.
    ///
    /// @param src The source data container to copy elements from.
    /// @param out The designation to copy elements to.
    public static void copyTo(DenseDoubleData src, DenseDoubleData out) {
        if (src.buffer() == out.buffer() && src.layout().equals(out.layout())) return; // Equivalent, no work to do.

        DenseKernelSupport.requireItemSize(src.layout().itemSize(), out.layout().itemSize());
        DenseKernelSupport.requireSameShape(src.layout(), out.layout());
        DenseKernelSupport.requireWritable(out.layout());

        if (Layout.areContiguousAndMatchOrder(src.layout(), out.layout())) {
            copyToContiguous(src.layout(), src.buffer(), out.layout(), out.buffer());
        } else {
            copyToStrided(src.layout(), src.buffer(), out.layout(), out.buffer());
        }
    }


    /// Copies the logical elements from a dense nD-array to another where at lease one of the arrays is strided.
    ///
    /// @param srcLayout The layout of the source nD-array to copy from.
    /// @param srcBuffer The data buffer of the source nD-array to copy from.
    /// @param outLayout The layout of the destination nD-array to copy to.
    /// @param outBuffer The data buffer of the source nD-array to copy to.
    private static <T> void copyToStrided(Layout srcLayout, T[] srcBuffer, Layout outLayout, T[] outBuffer) {
        int itemSize = srcLayout.itemSize();

        int rank = srcLayout.rank();
        if (rank == 0) { // scalar container
            arraycopy(srcBuffer, srcLayout.offset(), outBuffer, outLayout.offset(), itemSize);
            return;
        }

        int[] extent = new int[rank];
        int[] srcStride = new int[rank];
        int[] destStride = new int[rank];
        for (int ax = 0; ax < rank; ax++) {
            extent[ax] = srcLayout.getSize(ax);
            if (extent[ax] == 0) return; // empty: nothing to copy
            srcStride[ax] = srcLayout.stride(ax);
            destStride[ax] = outLayout.stride(ax);
        }

        int last = rank - 1;
        int innerN = extent[last], innerSrcS = srcStride[last], innerDestS = destStride[last];
        int[] idx = new int[rank];
        int srcOff = srcLayout.offset();
        int destOff = outLayout.offset();

        // Fast path: innermost axis is contiguous on BOTH sides. Then one run of
        // innerN elements is innerN*itemSize contiguous slots — a single arraycopy.
        boolean innerContig = (innerSrcS == itemSize) && (innerDestS == itemSize);

        while (true) {
            if (innerContig) {
                arraycopy(srcBuffer, srcOff, outBuffer, destOff, innerN*itemSize);
            } else {
                int s = srcOff, d = destOff;
                for (int i = 0; i < innerN; i++, s += innerSrcS, d += innerDestS)
                    arraycopy(srcBuffer, s, outBuffer, d, itemSize);
            }

            // Carry over the outer axes. srcOff/destOff always sit at (...,idx[last-1],0).
            int ax = last - 1;
            for (; ax >= 0; ax--) {
                if (++idx[ax] < extent[ax]) {
                    srcOff += srcStride[ax];
                    destOff += destStride[ax];
                    break;
                }
                idx[ax] = 0;
                srcOff -= (extent[ax] - 1)*srcStride[ax];   // back-stride
                destOff -= (extent[ax] - 1)*destStride[ax];
            }

            if (ax < 0) break;
        }
    }


    /// Copies the logical elements from a dense nD-array to another where at lease one of the arrays is strided.
    ///
    /// @param srcLayout The layout of the source nD-array to copy from.
    /// @param srcBuffer The data buffer of the source nD-array to copy from.
    /// @param outLayout The layout of the destination nD-array to copy to.
    /// @param outBuffer The data buffer of the source nD-array to copy to.
    private static void copyToStrided(Layout srcLayout, double[] srcBuffer, Layout outLayout, double[] outBuffer) {
        int itemSize = srcLayout.itemSize();

        int rank = srcLayout.rank();
        if (rank == 0) { // scalar container
            arraycopy(srcBuffer, srcLayout.offset(), outBuffer, outLayout.offset(), itemSize);
            return;
        }

        int[] extent = new int[rank];
        int[] srcStride = new int[rank];
        int[] destStride = new int[rank];
        for (int ax = 0; ax < rank; ax++) {
            extent[ax] = srcLayout.getSize(ax);
            if (extent[ax] == 0) return; // empty: nothing to copy
            srcStride[ax] = srcLayout.stride(ax);
            destStride[ax] = outLayout.stride(ax);
        }

        int last = rank - 1;
        int innerN = extent[last], innerSrcS = srcStride[last], innerDestS = destStride[last];
        int[] idx = new int[rank];
        int srcOff = srcLayout.offset();
        int destOff = outLayout.offset();

        // Fast path: innermost axis is contiguous on BOTH sides.
        boolean innerContig = (innerSrcS == 1) && (innerDestS == 1);

        while (true) {
            if (innerContig) {
                arraycopy(srcBuffer, srcOff*itemSize, outBuffer, destOff*itemSize, innerN*itemSize);
            } else {
                int s = srcOff, d = destOff;
                for (int i = 0; i < innerN; i++, s += innerSrcS, d += innerDestS)
                    arraycopy(srcBuffer, s*itemSize, outBuffer, d*itemSize, itemSize);
            }

            // Carry over the outer axes. srcOff/destOff always sit at (...,idx[last-1],0).
            int ax = last - 1;
            for (; ax >= 0; ax--) {
                if (++idx[ax] < extent[ax]) {
                    srcOff += srcStride[ax];
                    destOff += destStride[ax];
                    break;
                }
                idx[ax] = 0;
                srcOff -= (extent[ax] - 1)*srcStride[ax];   // back-stride
                destOff -= (extent[ax] - 1)*destStride[ax];
            }

            if (ax < 0) break;
        }
    }


    /// Copies the logical elements from a dense nD-array to another where *both* arrays are [contiguous][Layout#contiguousOrder()]
    /// in memory.
    ///
    /// **Note**: This method does *not* verify that `srcLayout` and `outLayout` are contiguous.
    ///
    /// @param srcLayout The layout of the source nD-array to copy from.
    /// @param srcBuffer The data buffer of the source nD-array to copy from.
    /// @param outLayout The layout of the destination nD-array to copy to.
    /// @param outBuffer The data buffer of the source nD-array to copy to.
    private static <T> void copyToContiguous(Layout srcLayout, T[] srcBuffer, Layout outLayout, T[] outBuffer) {
        int n = srcLayout.shape().numelIntValueExact()*srcLayout.itemSize();
        arraycopy(srcBuffer, srcLayout.offset(), outBuffer, outLayout.offset(), n);
    }


    /// Copies the logical elements from a dense nD-array to another where *both* arrays are [contiguous][Layout#contiguousOrder()]
    /// in memory.
    ///
    /// **Note**: This method does *not* verify that `srcLayout` and `outLayout` are contiguous.
    ///
    /// @param srcLayout The layout of the source nD-array to copy from.
    /// @param srcBuffer The data buffer of the source nD-array to copy from.
    /// @param outLayout The layout of the destination nD-array to copy to.
    /// @param outBuffer The data buffer of the source nD-array to copy to.
    private static void copyToContiguous(Layout srcLayout, double[] srcBuffer, Layout outLayout, double[] outBuffer) {
        int n = srcLayout.shape().numelIntValueExact()*srcLayout.itemSize();
        arraycopy(srcBuffer, srcLayout.offset(), outBuffer, outLayout.offset(), n);
    }


    public static <T> DenseData<T> compactCopy(DenseData<T> srcData) {
        Layout srcL = srcData.layout();
        int itemSize = srcL.itemSize();
        Layout dstL = Layout.contiguous(srcL.shape(), itemSize);
        T[] src = srcData.buffer();
        T[] dst = ArrayBuilder.newArrayLike(src, dstL.bufferSizeIntValueExact());

        int rank = srcL.rank();
        int scratch = Math.max(rank, 1);
        int[] cDims = new int[scratch];
        int[] cStrides = new int[scratch];
        int cRank = coalesceCOrder(srcL.shape().dims(), srcL.strides(), rank, cDims, cStrides);

        int n0 = cDims[cRank - 1];
        int s0 = cStrides[cRank - 1];
        int runSlots = n0*itemSize;
        boolean denseInner = (s0 == 1);

        long outerIters = 1;
        for (int i = 0; i < cRank - 1; i++) outerIters *= cDims[i];

        int[] odo = new int[Math.max(cRank - 1, 0)];
        int srcPos = srcL.offset();  // element units; scaled by itemSize only at the buffer touch
        int dstPos = 0;              // slot units; dst is contiguous with offset 0

        for (long it = 0; it < outerIters; it++) {
            if (denseInner) {
                arraycopy(src, srcPos*itemSize, dst, dstPos, runSlots);
            } else {
                for (int j = 0, p = srcPos, d = dstPos; j < n0; j++, p += s0, d += itemSize)
                    arraycopy(src, p*itemSize, dst, d, itemSize);
            }
            dstPos += runSlots;

            // Advance the odometer over the outer axes, incrementally maintaining srcPos.
            for (int axis = cRank - 2; axis >= 0; axis--) {
                srcPos += cStrides[axis];
                if (++odo[axis] < cDims[axis]) break;
                srcPos -= cStrides[axis]*cDims[axis];  // rewind this axis before carrying
                odo[axis] = 0;
            }
        }
        return new DenseData<>(dstL, dst);
    }


    public static DenseDoubleData compactCopy(DenseDoubleData srcData) {
        Layout srcL = srcData.layout();
        int itemSize = srcL.itemSize();
        Layout dstL = Layout.contiguous(srcL.shape(), itemSize);
        double[] src = srcData.buffer();
        double[] dst = new double[dstL.bufferSizeIntValueExact()];

        int rank = srcL.rank();
        int scratch = Math.max(rank, 1);
        int[] cDims = new int[scratch];
        int[] cStrides = new int[scratch];
        int cRank = coalesceCOrder(srcL.shape().dims(), srcL.strides(), rank, cDims, cStrides);

        int n0 = cDims[cRank - 1];
        int s0 = cStrides[cRank - 1];
        int s0Run = s0*itemSize;
        int runSlots = n0*itemSize;
        boolean denseInner = (s0 == 1);

        long outerIters = 1;
        for (int i = 0; i < cRank - 1; i++) outerIters *= cDims[i];

        int[] odo = new int[Math.max(cRank - 1, 0)];
        int srcPos = srcL.offset();  // element units; scaled by itemSize only at the buffer touch
        int dstPos = 0;              // slot units; dst is contiguous with offset 0

        for (long it = 0; it < outerIters; it++) {
            if (denseInner) {
                arraycopy(src, srcPos*itemSize, dst, dstPos, runSlots);
            } else {
                if (itemSize == 1) {
                    for (int j = 0, p = srcPos, d = dstPos; j < n0; j++, p += s0Run, d += itemSize) {
                        dst[d] = src[p];
                    }
                } else {
                    for (int j = 0, p = srcPos, d = dstPos; j < n0; j++, p += s0, d += itemSize)
                        arraycopy(src, p*itemSize, dst, d, itemSize);
                }
            }
            dstPos += runSlots;

            // Advance the odometer over the outer axes, incrementally maintaining srcPos.
            for (int axis = cRank - 2; axis >= 0; axis--) {
                srcPos += cStrides[axis];
                if (++odo[axis] < cDims[axis]) break;
                srcPos -= cStrides[axis]*cDims[axis];  // rewind this axis before carrying
                odo[axis] = 0;
            }
        }
        return new DenseDoubleData(dstL, dst);
    }
}
