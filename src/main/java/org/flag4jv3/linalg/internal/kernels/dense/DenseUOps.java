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

import org.flag4jv3.concurrency.Configurations;
import org.flag4jv3.concurrency.ThreadManager;
import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.dense.DenseData;
import org.flag4jv3.ndarrays.dense.DenseDoubleData;

import java.util.Objects;

import static org.flag4jv3.linalg.internal.kernels.DenseKernelSupport.*;

/// TODO NOW DOCS + Look over implementations + consider if common code can be extracted to private methods between generic and
public final class DenseUOps {

    private DenseUOps() {
        // Hide default constructor for utility class.
    }


    public static <T> DenseData<T> applyUOp(
            DenseData<T> src,
            DenseData<T> out,
            StridedUnaryLoop loop
    ) {
        Objects.requireNonNull(src, "src must not be null");
        Objects.requireNonNull(loop, "loop must not be null");
        var srcL = src.layout();
        var srcB = src.buffer();

        final var fOut = resolveOut(srcB, srcL.shape(), srcL.itemSize(), out);
        final var fOutL = fOut.layout();
        final var fOutB = fOut.buffer();

        requireSameShape(srcL, fOutL);
        requireWritable(fOutL);

        final var fSrc = resolveSrc(src, fOut);
        final var fSrcL = fSrc.layout();
        final var fSrcB = fSrc.buffer();

        int numel = fSrc.layout().shape().numelIntValueExact();
        // TODO NOW: This is just a placeholder. This should be dependent on operation being applied.
        //  Also may want an ExecutionPlan object rather than a single value for numel. How exactly that is implemented
        //  is up in the air...
        int parallelThreshold = 8192;

        // Fast path when both arrays are contiguous and their ordering match.
        // The cursor actually handles this path well and degenerates to a single loop run,
        // but having an explicit fast path here saves on the cursor overhead.
        if (Layout.areContiguousAndMatchOrder(fSrcL, fOutL)) {
            final int sIS = fSrcL.itemSize(), oIS = fOutL.itemSize();
            final int sOff = fSrcL.offset()*sIS, oOff = fOutL.offset()*oIS;

            if (numel < parallelThreshold) {
                loop.apply(fSrcB, sOff, sIS, fOutB, oOff, oIS, numel);
            } else {
                final int bandBlock = Configurations.getMemBlockSize(oIS*Configurations.ByteSize.DOUBLE.getNumBytes());
                ThreadManager.concurrentBlockedKernel(numel, bandBlock, (gLo, gHi) ->
                        loop.apply(fSrcB, sOff + gLo*sIS, sIS,
                                fOutB, oOff + gLo*oIS, oIS, gHi - gLo));
            }
            return fOut;
        }

        var cursor = new StridedRunCursor2(fSrcL, fOutL);
        if (cursor.isEmpty()) return fOut; // Nothing to do.

        if (numel < parallelThreshold) {
            runSequential(fSrcB, fOutB, loop, cursor);
        } else {
            runConcurrent(fSrcL, fSrcB, fOutL, fOutB, loop, cursor);
        }

        return fOut;
    }


    /// Applies the `loop` unary operation sequentially on all runs of the `cursor`.
    ///
    /// @param srcB The source buffer providing inputs to the unary operation.
    /// @param outB The output buffer storing results of the unary operation.
    /// @param loop The unary operation to apply to a contiguous run of `srcB`.
    /// @param cursor The cursor defining runs to apply `loop` to.
    private static <T> void runSequential(
            T[] srcB, T[] outB,
            StridedUnaryLoop loop, StridedRunCursor2 cursor
    ) {
        if (cursor.isEmpty()) return; // Nothing to do.

        do {
            loop.apply(srcB, cursor.aPos, cursor.aInnerBufStride, outB, cursor.bPos, cursor.bInnerBufStride, cursor.innerN);
        } while (cursor.next());
    }


    /// Applies the `loop` unary operation concurrently on all runs of the `cursor`.
    ///
    /// @param fSrcL The layout of the source nD-array.
    /// @param fSrcB The source buffer providing inputs to the unary operation.
    /// @param fOutL The layout of the output nD-array.
    /// @param fOutB The output buffer storing results of the unary operation.
    /// @param loop The unary operation to apply to a contiguous run of `srcB`.
    /// @param cursor The cursor defining runs to apply `loop` to.
    private static <T> void runConcurrent(
            Layout fSrcL, T[] fSrcB,
            Layout fOutL, T[] fOutB,
            StridedUnaryLoop loop, StridedRunCursor2 cursor
    ) {
        final int numel = fSrcL.shape().numelIntValueExact();
        if (numel == 0) return; // Nothing to do.
        final int innerN = cursor.innerN;
        // Reference elements (generic object references) are typically 4 or 8 bytes depending on configuration
        final int bandBlock = Configurations.getMemBlockSize(fOutL.itemSize()*4);

        ThreadManager.concurrentBlockedKernel(numel, bandBlock, (gLo, gHi) -> {
            var c = new StridedRunCursor2(fSrcL, fOutL); // per-worker cursor
            c.positionAtRun(gLo/innerN);
            int off = gLo%innerN; // nonzero only for the first run
            int g = gLo;
            while (g < gHi) {
                int n = Math.min(innerN - off, gHi - g);   // clamp: partial run at either end
                loop.apply(fSrcB, c.aPos + off*c.aInnerBufStride, c.aInnerBufStride,
                        fOutB, c.bPos + off*c.bInnerBufStride, c.bInnerBufStride, n);
                g += n;
                off = 0;
                c.next();
            }
        });
    }


    public static DenseDoubleData applyUOp(
            DenseDoubleData src,
            DenseDoubleData out,
            StridedDoubleUnaryLoop loop
    ) {
        Objects.requireNonNull(src, "src must not be null");
        Objects.requireNonNull(loop, "loop must not be null");
        var srcL = src.layout();

        final var fOut = resolveOut(srcL.shape(), srcL.itemSize(), out);
        final var fOutL = fOut.layout();
        final var fOutB = fOut.buffer();

        requireSameShape(srcL, fOutL);
        requireWritable(fOutL);

        final var fSrc = resolveSrc(src, fOut);
        final var fSrcL = fSrc.layout();
        final var fSrcB = fSrc.buffer();

        int numel = fSrc.layout().shape().numelIntValueExact();
        int parallelThreshold = 8192; // TODO NOW: This is just a place holder...

        // Fast path when both arrays are contiguous and their ordering match.
        // The cursor actually handles this path well and degenerates to a single loop run,
        // but having an explicit fast path here saves on the cursor overhead.
        if (Layout.areContiguousAndMatchOrder(fSrcL, fOutL)) {
            final int sIS = fSrcL.itemSize(), oIS = fOutL.itemSize();
            final int sOff = fSrcL.offset()*sIS, oOff = fOutL.offset()*oIS;

            if (numel < parallelThreshold) {
                loop.apply(fSrcB, sOff, sIS, fOutB, oOff, oIS, numel);
            } else {
                final int bandBlock = Configurations.getMemBlockSize(oIS*Configurations.ByteSize.DOUBLE.getNumBytes());
                ThreadManager.concurrentBlockedKernel(numel, bandBlock, (gLo, gHi) ->
                        loop.apply(fSrcB, sOff + gLo*sIS, sIS,
                                fOutB, oOff + gLo*oIS, oIS, gHi - gLo));
            }
            return fOut;
        }

        var cursor = new StridedRunCursor2(fSrcL, fOutL);
        if (cursor.isEmpty()) return fOut; // Nothing to do.

        if (numel < parallelThreshold) {
            runSequential(fSrcB, fOutB, loop, cursor);
        } else {
            runConcurrent(fSrcL, fSrcB, fOutL, fOutB, loop, cursor);
        }

        return fOut;
    }


    /// Applies the `loop` unary operation sequentially on all runs of the `cursor`.
    ///
    /// @param srcB The source buffer providing inputs to the unary operation.
    /// @param outB The output buffer storing results of the unary operation.
    /// @param loop The unary operation to apply to a contiguous run of `srcB`.
    /// @param cursor The cursor defining runs to apply `loop` to.
    private static void runSequential(
            double[] srcB, double[] outB,
            StridedDoubleUnaryLoop loop, StridedRunCursor2 cursor
    ) {
        if (cursor.isEmpty()) return; // Nothing to do.

        do {
            loop.apply(srcB, cursor.aPos, cursor.aInnerBufStride, outB, cursor.bPos, cursor.bInnerBufStride, cursor.innerN);
        } while (cursor.next());
    }


    /// Applies the `loop` unary operation concurrently on all runs of the `cursor`.
    ///
    /// @param fSrcL The layout of the source nD-array.
    /// @param fSrcB The source buffer providing inputs to the unary operation.
    /// @param fOutL The layout of the output nD-array.
    /// @param fOutB The output buffer storing results of the unary operation.
    /// @param loop The unary operation to apply to a contiguous run of `srcB`.
    /// @param cursor The cursor defining runs to apply `loop` to.
    private static void runConcurrent(
            Layout fSrcL, double[] fSrcB,
            Layout fOutL, double[] fOutB,
            StridedDoubleUnaryLoop loop, StridedRunCursor2 cursor
    ) {
        final int numel = fSrcL.shape().numelIntValueExact();
        if (numel == 0) return; // Nothing to do.
        final int innerN = cursor.innerN;
        final int bandBlock = Configurations.getMemBlockSize(fOutL.itemSize()*Configurations.ByteSize.DOUBLE.getNumBytes());

        ThreadManager.concurrentBlockedKernel(numel, bandBlock, (gLo, gHi) -> {
            var c = new StridedRunCursor2(fSrcL, fOutL); // per-worker cursor
            c.positionAtRun(gLo/innerN);
            int off = gLo%innerN; // nonzero only for the first run
            int g = gLo;
            while (g < gHi) {
                int n = Math.min(innerN - off, gHi - g);   // clamp: partial run at either end
                loop.apply(fSrcB, c.aPos + off*c.aInnerBufStride, c.aInnerBufStride,
                        fOutB, c.bPos + off*c.bInnerBufStride, c.bInnerBufStride, n);
                g += n;
                off = 0;
                c.next();
            }
        });
    }
}
