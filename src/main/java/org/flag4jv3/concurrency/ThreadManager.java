/*
 * MIT License
 *
 * Copyright (c) 2022-2026. Jacob Watters
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

package org.flag4jv3.concurrency;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

import static org.flag4jv3.concurrency.Configurations.DEFAULT_PARALLELISM;

/**
 * <p>Manages a global thread pool and provides utility methods for executing parallel tensor kernels in Flag4j.
 *
 * <h2>Usage:</h2>
 * <ul>
 *   <li>The size of this pool can be set via {@link #setParallelism(int)} and queried via {@link #getParallelismLevel()}.</li>
 *   <li>
 *       {@link #concurrentKernel(int, TensorKernel)} divides an slice range into contiguous chunks for
 *       general parallel kernels. {@link #concurrentBlockedKernel(int, int, TensorKernel)} divides it into
 *       block-aligned bands for kernels that are internally cache-blocked.
 *    </li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <ul>
 *   <li>
 *      This class is thread-safe: concurrent calls to {@link #setParallelism(int)} and the concurrent
 *      kernels do not corrupt internal state.
 *   </li>
 *   <li>
 *       Both {@link #concurrentKernel(int, TensorKernel)} and {@link #concurrentBlockedKernel(int, int, TensorKernel)}
 *       assume the supplied {@link TensorKernel} is thread safe. Generally this would be achieved by arranging
 *       each chunk or band write to a disjoint region of the output. It is the responsibility of users of this class to ensure
 *       that this assumption is respected. Any section in a kernel that is not thread safe must be guarded with a
 *       {@code synchronized} block by the caller in the kernel implementation. Failure to respect this assumption may lead to
 *       undefined behavior.
 *       </li>
 * </ul>
 */
public final class ThreadManager {

    private ThreadManager() {
        // Hide the default constructor for the utility class.
    }


    /**
     * Marker type for threads belonging to this manager's pool. Allows O(1) detection of
     * re-entrant kernel dispatch (see {@link #isWorkerThread()}).
     */
    /// Marker type for threads belonging to this manage's pool. Allows O(1) detection of re-entrant kernel dispatch
    /// (see [#isWorkerThread()]) to prevent deadlocking.
    private static final class Flag4jWorker extends Thread {
        Flag4jWorker(Runnable target) {
            super(target);
            setDaemon(true);  // Do not block JVM shutdown.
        }
    }

    private static final ThreadFactory DAEMON_FACTORY = Flag4jWorker::new;


    /**
     * Checks if the calling thread is a worker thread of this manager's pool. Kernels dispatched
     * from a worker thread are executed inline rather than re-partitioned; see
     * {@link #concurrentKernel(int, TensorKernel)}.
     *
     * @return {@code true} if the calling thread belongs to this manager's pool.
     */
    public static boolean isWorkerThread() {
        return Thread.currentThread() instanceof Flag4jWorker;
    }


    /**
     * The parallelism level for the thread manager. That is, the number of threads in the thread pool.
     */
    private static volatile int parallelism = DEFAULT_PARALLELISM;

    /**
     * Thread pool for executing concurrent kernels.
     */
    private static ThreadPoolExecutor threadPool =
            (ThreadPoolExecutor) Executors.newFixedThreadPool(parallelism, DAEMON_FACTORY);

    /**
     * Lock object for synchronizing changes to the thread pool and parallelismLevel.
     */
    private static final Object POOL_LOCK = new Object();


    /**
     * Sets the number of threads to use in the thread pool.
     *
     * @param level The parallelism level to use in the thread pool.
     * <ul>
     *     <li>If {@code level > 0}: The parallelism level is used as is.</li>
     *     <li>If {@code level <= 0}: The parallelism level will be set to
     *     {@code Math.max(Configurations.DEFAULT_PARALLELISM + level, 1)}. Such values may be interpreted as
     *     'x' less than the number of available processors. To set the parallelism level to 2 less than the number of available
     *     processors, do {@code setParallelism(-2)}.</li>
     * </ul>
     */
    protected static void setParallelism(int level) {
        synchronized (POOL_LOCK) {
            if (level <= 0) level = DEFAULT_PARALLELISM + level;
            level = Math.max(level, 1);

            if (level > threadPool.getCorePoolSize()) {
                threadPool.setMaximumPoolSize(level);
                threadPool.setCorePoolSize(level);
            } else {
                threadPool.setCorePoolSize(level);
                threadPool.setMaximumPoolSize(level);
            }

            parallelism = level;
        }
    }


    /**
     * Gets the current parallelism level for the ThreadManager. That is, the number of threads used in the thread pool.
     *
     * @return The current parallelism level for the ThreadManager.
     */
    public static int getParallelismLevel() {
        return parallelism;
    }


    /**
     * <p>Computes a tensor kernel concurrently by partitioning the outer-loop slice range
     * {@code [0, totalSize)} into contiguous, equal-sized chunks and dispatching one chunk per worker thread
     * (up to the current {@link #getParallelismLevel() parallelism level}).
     *
     * <p>For a kernel that is internally blocked, prefer
     * {@link #concurrentBlockedKernel(int, int, TensorKernel)}, which aligns chunk boundaries to block borders
     * so the kernel's cache tiling is preserved.
     *
     * <p><b>WARNING</b>: This method provides <em>no</em> guarantees of thread safety. It is the responsibility of the
     * caller to ensure {@code kernel} is thread safe (over partioning of the outer-loop).
     *
     * @param totalSize Total size of the outer loop. The range partitioned across threads is {@code [0, totalSize)}.
     * @param kernel The kernel to apply to each chunk, invoked as {@code kernel.apply(startIdx, endIdx)} with
     * {@code startIdx} inclusive and {@code endIdx} exclusive.
     * @see #concurrentBlockedKernel(int, int, TensorKernel)
     */
    public static void concurrentKernel(final int totalSize, final TensorKernel kernel) {
        if (totalSize <= 0) return;

        final int parallelism = getParallelismLevel();
        // Re-entrant dispatch (already on a pool thread) must not submit and block: the pool is
        // fixed-size, so a worker waiting on subtasks it cannot schedule would deadlock.
        if (parallelism == 1 || isWorkerThread()) {
            kernel.apply(0, totalSize);
            return;
        }

        final int numChunks = Math.min(parallelism, totalSize);
        List<Future<?>> futures = new ArrayList<>(numChunks - 1);

        // Submit all chunks but the last; the calling thread computes that one itself.
        for (int i = 0; i < numChunks - 1; i++) {
            final int start = (int) ((long) i*totalSize/numChunks);
            final int end = (int) ((long) (i + 1)*totalSize/numChunks);
            if (start < end) futures.add(threadPool.submit(() -> kernel.apply(start, end)));
        }

        final int lastStart = (int) ((long) (numChunks - 1)*totalSize/numChunks);
        runInlineAndAwait(kernel, lastStart, totalSize, futures);
    }


    /**
     * <p>Computes a blocked tensor kernel concurrently by partitioning the outer-loop slice range
     * {@code [0, totalSize)} into contiguous, block-aligned bands and dispatching one band per worker thread
     * (up to the current {@link #getParallelismLevel() parallelism level}).
     *
     * <p>Unlike {@link #concurrentKernel(int, TensorKernel)}, this method respects {@code blockSize}: every
     * band handed to {@code kernel} begins on a block boundary and spans a whole number of blocks (the final
     * band may be shorter). This keeps each worker's range aligned to the cache-tiling structure of a blocked
     * kernel so interior tiles stay full-sized and gives each worker a single contiguous band of the slice range
     * for predictable cache locality. Prefer this over {@link #concurrentKernel(int, TensorKernel)}
     * when {@code kernel} is internally blocked.
     *
     * <p><b>WARNING</b>: This method provides <em>no</em> guarantees of thread safety. It is the responsibility of the
     * caller to ensure {@code kernel} is thread safe (over partioning of the outer-loop).
     *
     * @param totalSize Total size of the outer loop. The range partitioned across threads is {@code [0, totalSize)}.
     * @param blockSize Block size of the blocked {@code kernel}. Band boundaries are aligned to multiples of this
     * value. Must be positive.
     * @param kernel Blocked kernel to apply to each band, invoked as {@code kerneln.apply(startIdx, endIdx)}
     * with {@code startIdx} inclusive and {@code endIdx} exclusive. {@code endIdx - startIdx} is a
     * multiple of {@code blockSize} for every band except possibly the last.
     * @see #concurrentKernel(int, TensorKernel)
     */
    public static void concurrentBlockedKernel(final int totalSize, final int blockSize, final TensorKernel kernel) {
        if (totalSize <= 0) return;
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be positive but got " + blockSize + ".");
        }

        final int parallelism = getParallelismLevel();
        if (parallelism == 1 || isWorkerThread()) {
            kernel.apply(0, totalSize);
            return;
        }

        final int numBlocks = (totalSize + blockSize - 1)/blockSize;
        final int numBands = Math.min(parallelism, numBlocks);
        List<Future<?>> futures = new ArrayList<>(numBands - 1);

        for (int i = 0; i < numBands - 1; i++) {
            final int start = (int) ((long) i*numBlocks/numBands)*blockSize;
            final int end = (int) ((long) (i + 1)*numBlocks/numBands)*blockSize;
            if (start < end) futures.add(threadPool.submit(() -> kernel.apply(start, end)));
        }

        final int lastStart = (int) ((long) (numBands - 1)*numBlocks/numBands)*blockSize;
        runInlineAndAwait(kernel, lastStart, totalSize, futures);
    }


    private static void runInlineAndAwait(TensorKernel kernel, int start, int end, List<Future<?>> futures) {
        Throwable inlineFailure = null;
        try {
            if (start < end) kernel.apply(start, end);
        } catch (RuntimeException | Error e) {
            inlineFailure = e;
        }

        try {
            awaitAll(futures);
        } catch (RuntimeException e) {
            if (inlineFailure != null) e.addSuppressed(inlineFailure);
            throw e;
        }

        if (inlineFailure instanceof RuntimeException e) throw e;
        if (inlineFailure instanceof Error e) throw e;
    }


    private static void awaitAll(List<Future<?>> futures) {
        RuntimeException failure = null;

        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                for (Future<?> f : futures) f.cancel(true);
                throw new RuntimeException("Concurrent kernel was interrupted.", e);
            } catch (ExecutionException e) {
                if (failure == null) {
                    failure = new RuntimeException("Concurrent kernel failed.", e.getCause());
                } else {
                    failure.addSuppressed(e.getCause());
                }
            }
        }

        if (failure != null) throw failure;
    }
}
