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

import org.flag4jv3.ndarrays.Layout;

import static org.flag4jv3.linalg.internal.kernels.DenseKernelSupport.coalesceCOrder2;

/// A cursor for iterating over a source [layout][Layout] and storing results in another output [layout][Layout].
///
/// <blockquote style="color: #9da7c2; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
///     <strong>Note:</strong> Note: traversal order using a cursor is unspecified beyond visiting each logical index exactly once.
/// </blockquote>
public final class StridedRunCursor2 {
    public int aPos;
    public int bPos;
    public final int aInnerBufStride;
    public final int bInnerBufStride;
    public final int innerN;
    private final int aBase;
    private final int bBase;
    private final int[] dims;
    private final int[] aBufStrides;
    private final int[] bBufStrides;
    private final int[] odo;
    private final int outerRank;
    private final boolean empty;
    private final int totalRuns;


    public StridedRunCursor2(Layout a, Layout b) {
        final int rank = a.rank();
        final int aIS = a.itemSize(), bIS = b.itemSize();

        aBase = a.offset()*aIS;
        bBase = b.offset()*bIS;
        final int numel = a.shape().numelIntValueExact();
        empty = numel == 0;

        // Emptiness is a property of the shape alone: any zero extent => no work.
        if (empty) {
            innerN = 0;
            aInnerBufStride = bInnerBufStride = 0;
            dims = aBufStrides = bBufStrides = odo = null;
            outerRank = 0;
            totalRuns = 0;
            aPos = bPos = 0;
            return;
        }

        final int scratch = Math.max(rank, 1);
        int[] d = new int[scratch], sa = new int[scratch], sb = new int[scratch];
        for (int ax = 0; ax < rank; ax++) {
            d[ax] = a.getSize(ax);
            sa[ax] = a.stride(ax)*aIS;   // slot units before coalescing
            sb[ax] = b.stride(ax)*bIS;
        }

        // Reorder axes so out (b) strides are |descending| => b's smallest-stride
        // axis becomes innermost. Legal for element-wise traversal because the same
        // permutation is applied to both operands. Insertion sort: rank is tiny,
        // and strict '<' keeps it stable (ties preserve original C order).
        for (int i = 1; i < rank; i++) {
            int di = d[i], sai = sa[i], sbi = sb[i];
            int key = Math.abs(sbi);
            int j = i - 1;
            while (j >= 0 && Math.abs(sb[j]) < key) {
                d[j + 1] = d[j];
                sa[j + 1] = sa[j];
                sb[j + 1] = sb[j];
                j--;
            }
            d[j + 1] = di;
            sa[j + 1] = sai;
            sb[j + 1] = sbi;
        }

        int cRank = (rank == 0) ? 0 : coalesceCOrder2(d, sa, sb, rank, d, sa, sb);

        // Reachable for rank-0, or if coalescing drops extent-1 axes and every axis was
        // extent-1. Both imply numel == 1 (we already know numel != 0), so a single
        // unit-length run is the correct normalization.
        if (cRank == 0) {
            cRank = 1;
            d[0] = 1;
            sa[0] = 0;
            sb[0] = 0;
        }

        innerN = d[cRank - 1];
        aInnerBufStride = sa[cRank - 1];
        bInnerBufStride = sb[cRank - 1];
        outerRank = cRank - 1;
        dims = d;
        aBufStrides = sa;
        bBufStrides = sb;
        odo = new int[Math.max(outerRank, 1)];
        aPos = aBase;
        bPos = bBase;

        int r = 1;
        for (int ax = 0; ax < outerRank; ax++) r *= dims[ax];
        totalRuns = r;

        if (totalRuns*innerN != numel) {
            throw new AssertionError("Coalescing did not preserve element count: totalRuns="
                    + totalRuns + ", innerN=" + innerN + ", numel=" + numel + ".");
        }
    }


    public void positionAtRun(long runIdx) {
        int aP = aBase, bP = bBase;            // pre-scaled slot offsets saved at construction
        for (int ax = outerRank - 1; ax >= 0; ax--) {
            int digit = (int) (runIdx%dims[ax]);
            runIdx /= dims[ax];
            odo[ax] = digit;
            aP += digit*aBufStrides[ax];
            bP += digit*bBufStrides[ax];
        }
        aPos = aP;
        bPos = bP;
    }


    public boolean isEmpty() {
        return empty;
    }


    /// The total number of inner runs within this cursor.
    public int totalRuns() {
        return totalRuns;
    }


    /// Produces the next odometer index.
    public boolean next() {
        for (int ax = outerRank - 1; ax >= 0; ax--) {
            aPos += aBufStrides[ax];
            bPos += bBufStrides[ax];
            if (++odo[ax] < dims[ax]) return true;
            aPos -= aBufStrides[ax]*dims[ax];   // rewind before carrying
            bPos -= bBufStrides[ax]*dims[ax];
            odo[ax] = 0;
        }

        return false;
    }
}
