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


import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.logging.Logger;


/**
 * <p>A utility class for configuring standard and concurrent operations.
 * <p>This class provides configurable settings for multithreaded and blocked computations.
 *
 * <p>If a configuration file {@code ./configs/GeneralConfigs.properties} is present, settings will be loaded
 * at class initialization. If not, default values will be used.
 *
 * <h2>Thread Safety:</h2>
 * This class is designed to be thread-safe.
 * <ul>
 *     <li>The {@code parallelism} setting is managed via {@link ThreadManager}, which ensures safe access.</li>
 *     <li>The {@code blockSize} setting is accessed and modified using synchronized methods.</li>
 * </ul>
 */
public final class Configurations {

    /**
     * The default parallelism (i.e., number of threads) to use for concurrent algorithms.
     */
    public static final int DEFAULT_PARALLELISM = Runtime.getRuntime().availableProcessors();
    /**
     * The default block size for blocked algorithms.
     */
    public static final int DEFAULT_BLOCK_SIZE = 64;
    /**
     * The block size to use in blocked algorithms.
     */
    private static volatile int blockSize = DEFAULT_BLOCK_SIZE;


    static {
        // Set up a simple logger in case of missing configs.
        Logger logger = Logger.getLogger(ThreadManager.class.getName());
        Path path = Paths.get("./configs/GeneralConfigs.properties");
        Properties properties = new Properties();

        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            properties.load(reader);
        } catch (IOException e) {
            logger.warning("Failed to load " + path.toAbsolutePath() + ": " + e.getMessage());
            logger.warning("Using default configuration");
        }

        int parallelism = DEFAULT_PARALLELISM;

        if (properties.containsKey("blockSize"))
            blockSize = Integer.parseInt(properties.getProperty("blockSize"));
        if (properties.containsKey("parallelism")) {
            setParallelismLevel(Integer.parseInt(properties.getProperty("parallelism")));
        }
    }


    private Configurations() {
        // Hide default constructor for utility class.
    }


    /**
     * Sets the parallelism level (i.e., number of threads) for use in concurrent ops as the number of processors available to the Java
     * virtual machine. Note that this value may change during runtime. This method will include logical cores so the value
     * returned may be higher than the number of physical cores on the machine if hyper-threading is enabled.
     * <br><br>
     * @implNote This is implemented as:
     * {@code parallelism = {@link Runtime#availableProcessors() Runtime.getRuntime().availableProcessors()};}
     * @return The new parallelism value, i.e., the number of available processors.
     */
    public static int setParallelismLevelAsAvailableProcessors() {
        ThreadManager.setParallelismLevel(Runtime.getRuntime().availableProcessors());
        return ThreadManager.getParallelismLevel();
    }


    /**
     * Gets the current parallelism (i.e., number of threads) to be used in concurrent algorithms.
     * @return Current parallelism (i.e., number of threads) to use in concurrent algorithms.
     */
    public static int getParallelismLevel() {
        return ThreadManager.getParallelismLevel();
    }


    /**
     * Sets the parallelism level (i.e., number of threads) to use in concurrent algorithms.
     * @param parallelismLevel The parallelism level (i.e., number of threads) to use in concurrent algorithms.
     * <ul>
     *     <li>If {@code parallelismLevel > 0}: The parallelism level is used as is.</li>
     *     <li>If {@code parallelismLevel <= 0}: The parallelism level will be set to
     *     {@code Math.max(Configurations.DEFAULT_PARALLELISM + parallelismLevel + 1, 1)}.</li>
     * </ul>
     */
    public static void setParallelismLevel(int parallelismLevel) {
        ThreadManager.setParallelismLevel(parallelismLevel);
    }


    /**
     * Gets the current block size used in blocked algorithms. If it has not been changed it will
     * {@link #DEFAULT_BLOCK_SIZE default to 64}.
     * @return Current block size to use in concurrent algorithms.
     */
    public static synchronized int getBlockSize() {
        return blockSize;
    }


    /**
     * Sets the current block size used in blocked algorithms.
     * @param blockSize Block size to be used in concurrent algorithms.
     */
    public static synchronized void setBlockSize(int blockSize) {
        Configurations.blockSize = Math.max(1, blockSize);
    }


    /**
     * Resets all configurations to their default values.
     */
    public static synchronized void resetAll() {
        ThreadManager.setParallelismLevel(DEFAULT_PARALLELISM);
        blockSize = DEFAULT_BLOCK_SIZE;
    }
}
