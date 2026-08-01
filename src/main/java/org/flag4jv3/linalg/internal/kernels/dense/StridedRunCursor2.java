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

public final class StridedRunCursor2 {
    public int aPos;
    public int bPos;
    public final int aInnerStride;
    public final int bInnerStride;
    public final int innerN;
    private final int aBase;
    private final int bBase;
    private final int[] dims;
    private final int[] aStrides;
    private final int[] bStrides;
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
            aInnerStride = bInnerStride = 0;
            dims = aStrides = bStrides = odo = null;
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
        aInnerStride = sa[cRank - 1];
        bInnerStride = sb[cRank - 1];
        outerRank = cRank - 1;
        dims = d;
        aStrides = sa;
        bStrides = sb;
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
            aP += digit*aStrides[ax];
            bP += digit*bStrides[ax];
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
            aPos += aStrides[ax];
            bPos += bStrides[ax];
            if (++odo[ax] < dims[ax]) return true;
            aPos -= aStrides[ax]*dims[ax];   // rewind before carrying
            bPos -= bStrides[ax]*dims[ax];
            odo[ax] = 0;
        }

        return false;
    }
}
