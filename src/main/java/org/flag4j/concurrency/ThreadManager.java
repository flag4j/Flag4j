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

package org.flag4j.concurrency;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.flag4j.concurrency.Configurations.DEFAULT_PARALLELISM;

/**
 * <p>Manages a global thread pool and utility methods for executing parallel operations in Flag4j.
 *
 * <h2>Usage:</h2>
 * <ul>
 *   <li>This class provides a shared, fixed-size thread pool to perform parallel operations.</li>
 *   <li>The size of this thread pool (i.e., the "parallelism level") can be set via
 *       {@link #setParallelismLevel(int)} and queried via {@link #getParallelismLevel()}.</li>
 *   <li>The pool uses daemon threads to avoid blocking JVM shutdown.</li>
 * </ul>
 *
 * <h2>Thread Safety & Design:</h2>
 * <ul>
 *   <li>This class is thread-safe in that calls to {@code setParallelismLevel} and concurrent operations
 *       do not corrupt internal state. However, changing the parallelism level while tasks are actively
 *       running may cause those tasks to be abruptly shut down. Use <em>extreme</em> caution if dynamic
 *       changes to parallelism are required.</li>
 *   <li>The methods {@link #concurrentOperation(int, TensorOperation)} and
 *       {@link #concurrentBlockedOperation(int, int, TensorOperation)} assume the provided
 *       {@link TensorOperation} is itself thread-safe. If there are blocks within the operation which are not thread-safe they
 *       should be wrapped in a {@code synchronized} block.</li>
 * </ul>
 */
public final class ThreadManager {
    private ThreadManager() {
        // Hide default constructor for utility class.
    }

    /**
     * Simple thread factory for creating basic daemon threads.
     */
    private static final ThreadFactory DAEMON_FACTORY = r -> {
        Thread thread = new Thread(r);
        thread.setDaemon(true); // Set the thread as a daemon thread to avoid blocking JVM shutdown.
        return thread;
    };


    /**
     * The parallelism level for the thread manager. That is, the number of threads to be used in the thread pool
     * when executing concurrent ops.
     */
    private static int parallelismLevel = DEFAULT_PARALLELISM;

    /**
     * Simple logger for when a thread throws an exception during execution.
     */
    private static final Logger THREAD_LOGGER = Logger.getLogger(ThreadManager.class.getName());

    /**
     * Thread pool for managing threads executing concurrent ops.
     */
    private static ThreadPoolExecutor threadPool =
            (ThreadPoolExecutor) Executors.newFixedThreadPool(parallelismLevel, DAEMON_FACTORY);

    /**
     * Lock object for synchronizing changes to the thread pool and parallelismLevel.
     */
    private static final Object POOL_LOCK = new Object();


    /**
     * Sets the number of threads to use in the thread pool.
     * @param parallelismLevel The parallelism level to use in the thread pool.
     * <ul>
     *     <li>If {@code parallelismLevel > 0}: The parallelism level is used as is.</li>
     *     <li>If {@code parallelismLevel <= 0}: The parallelism level will be set to
     *     {@code Math.max(Configurations.DEFAULT_PARALLELISM + parallelismLevel, 1)}. Such values may be interpreted as
     *     'x' less than the number of available processors. To set the parallelism level to 2 less than the number of available
     *     processors, do {@code setParallelismLevel(-2)}.</li>
     * </ul>
     */
    protected static void setParallelismLevel(int parallelismLevel) {
        synchronized (POOL_LOCK) {
            // Attempt to gracefully shut down the old pool.
            if(threadPool != null) {
                threadPool.shutdown(); // Disable new tasks from being submitted.
                try {
                    // Wait for existing tasks to terminate.
                    if(!threadPool.awaitTermination(60, TimeUnit.SECONDS)) {
                        threadPool.shutdownNow(); // Cancel currently executing tasks.

                        if(!threadPool.awaitTermination(60, TimeUnit.SECONDS))
                            THREAD_LOGGER.warning("ThreadPool did not terminate gracefully.");
                    }
                } catch(InterruptedException ie) {
                    THREAD_LOGGER.log(Level.WARNING, "Interrupted during thread pool shutdown.", ie);
                    threadPool.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }

            if (parallelismLevel <= 0)
                parallelismLevel = DEFAULT_PARALLELISM + parallelismLevel;

            parallelismLevel = Math.max(parallelismLevel, 1);  // Ensure the value is positive.
            threadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(parallelismLevel, DAEMON_FACTORY);
        }
    }


    /**
     * Gets the current parallelism level for the ThreadManager. That is, the number of threads used in the thread pool.
     * @return The current parallelism level for the ThreadManager.
     */
    public static int getParallelismLevel() {
        return parallelismLevel;
    }


    /**
     * <p>Computes a specified tensor operation concurrently by evenly dividing work among available threads (specified by
     * {@link Configurations#getParallelismLevel()}).
     *
     * <p>WARNING: This method provides <em>no</em> guarantees of thread safety. It is the responsibility of the caller to ensure that
     * {@code operation} is thread safe.
     *
     * @param totalSize Total size of the outer loop for the operation.
     * @param operation Operation to be computed.
     */
    public static void concurrentOperation(final int totalSize, final TensorOperation operation) {
        // Calculate chunk size.
        final int LOCAL_PARALLELISM = getParallelismLevel();
        int chunkSize = (totalSize + LOCAL_PARALLELISM - 1) / LOCAL_PARALLELISM;
        List<Future<?>> futures = new ArrayList<>(LOCAL_PARALLELISM);

        for(int threadIndex = 0; threadIndex < LOCAL_PARALLELISM; threadIndex++) {
            final int startIdx = threadIndex * chunkSize;
            final int endIdx = Math.min(startIdx + chunkSize, totalSize);

            futures.add(ThreadManager.threadPool.submit(() -> {
                operation.apply(startIdx, endIdx);
            }));
        }

        // Wait for all tasks to complete.
        for(Future<?> future : futures) {
            try {
                future.get(); // Ensure all tasks are complete.
            } catch (InterruptedException e) {
                // Log and preserve interrupt status.
                THREAD_LOGGER.log(Level.WARNING, "Interrupted while waiting for concurrent operation task.", e);
                Thread.currentThread().interrupt();
            } catch (ExecutionException e) {
                // The operation threw an exception.
                THREAD_LOGGER.log(Level.WARNING, "Error during concurrent operation task.", e);
            }
        }
    }


    /**
     * <p>Computes a specified blocked tensor operation concurrently by evenly dividing work among available threads (specified by
     * {@link Configurations#getParallelismLevel()}).
     *
     * <p>Unlike {@link #concurrentOperation(int, TensorOperation)} this method respects the block size of the blocked operation.
     * This means tasks split across threads will be aligned to block borders if possible which allows for the improved cache
     * performance benefits of blocked ops to be fully realized. For this reason, it is <em>not</em> recommended to use
     * {@link #concurrentOperation(int, TensorOperation)} to compute a blocked operation concurrently.
     *
     * <p>WARNING: This method provides <em>no</em> guarantees of thread safety. It is the responsibility of the caller to ensure that
     * {@code blockedOperation} is thread safe.
     *
     * @param totalSize Total size of the outer loop for the operation.
     * @param blockSize Size of the block used in the {@code blockedOperation}.
     * @param blockedOperation Operation to be computed.
     */
    public static void concurrentBlockedOperation(final int totalSize, final int blockSize, final TensorOperation blockedOperation) {
        // Calculate chunk size for blocks.
        final int LOCAL_PARALLELISM = getParallelismLevel();
        int numBlocks = (totalSize + blockSize - 1)/blockSize;
        List<Future<?>> futures = new ArrayList<>(LOCAL_PARALLELISM);

        for(int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
            final int startBlock = blockIndex*blockSize;
            final int endBlock = Math.min(startBlock + blockSize, totalSize);

            futures.add(threadPool.submit(() ->
                    blockedOperation.apply(startBlock, endBlock))
            );
        }

        // Wait for all tasks to complete.
        for(Future<?> future : futures) {
            try {
                future.get(); // Ensure all tasks are complete.
            } catch (InterruptedException e) {
                THREAD_LOGGER.log(Level.WARNING, "Interrupted while waiting for blocked operation task.", e);
                Thread.currentThread().interrupt();
            } catch (ExecutionException e) {
                THREAD_LOGGER.log(Level.WARNING, "Error during blocked operation task.", e);
            }
        }
    }
}
