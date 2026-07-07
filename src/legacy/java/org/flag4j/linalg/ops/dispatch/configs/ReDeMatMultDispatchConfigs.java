/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.ops.dispatch.configs;

import org.flag4j.concurrency.ThreadManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.logging.Logger;

/**
 * <p>A utility class for configuring dynamic kernel dispatch thresholds for real dense matrix multiplication.
 *
 * <p>If a configuration file {@code ./configs/ReDeMatMult.properties} is present, settings will be loaded
 * at class initialization. If not, default values will be used. It is not generally recommended to set these values by hand.
 *
 * <h2>Thread Safety:</h2>
 * This class is designed to be thread-safe.
 */
public final class ReDeMatMultDispatchConfigs {

    /**
     * The cache size for caching kernels to use for a given pair of shapes.
     */
    private static double cacheSize = 64;

    /**
     * Threshold for considering a matrix "near-square". If the quotient of the maximum and minimum dimension of either matrix
     * is less than this value, the matrix will be considered "near-square".
     */
    private static double aspectThreshold = 2.0;

    /**
     * Threshold for the total number of entries in both matrices to consider the problem "small enough" to
     * default to the standard algorithm.
     */
    private static int smallThreshold = 2_450;

    /**
     * Threshold for using a concurrent standard kernel for "near square" matrices.
     */
    private static int squareMtStandardThreshold = 50;
    /**
     * Threshold for using a concurrent reordered kernel for "near square" matrices.
     */
    private static int squareMtReorderedThreshold = 2_048;

    /**
     * Thresholds for non-square wide matrices. i.e., {@code m = max(m, n, k)} and {@code max(m, n, k) / min(m, n, k) >  aspectThreshold}.
     */
    private static int wideMtReorderedThreshold = 25;
    /**
     * Threshold for using standard matrix-vector kernel when matrix is "near square".
     */
    private static int squareSequentialVecThreshold = 256;

    /**
     * Threshold for considering the minimum dimension small enough to fall back to sequential kernel.
     */
    private static int minDimSmallThreshold = 10;

    static {
        Logger logger = Logger.getLogger(ThreadManager.class.getName());
        Path path = Paths.get("./configs/ReDeMatMult.properties");
        Properties properties = new Properties();

        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            properties.load(reader);
        } catch (IOException e) {
            logger.info("No config found at the expected location" + path.toAbsolutePath());
            logger.info("Falling back to default configurations");
        }

        if (properties.containsKey("cacheSize"))
            cacheSize = Double.parseDouble(properties.getProperty("cacheSize"));
        if (properties.containsKey("aspectThreshold"))
            aspectThreshold = Double.parseDouble(properties.getProperty("aspectThreshold"));
        if (properties.containsKey("smallThreshold"))
            smallThreshold = Integer.parseInt(properties.getProperty("smallThreshold"));
        if (properties.containsKey("squareMtStandardThreshold"))
            squareMtStandardThreshold = Integer.parseInt(properties.getProperty("squareMtStandardThreshold"));
        if (properties.containsKey("squareMtReorderedThreshold"))
            squareMtReorderedThreshold = Integer.parseInt(properties.getProperty("squareMtReorderedThreshold"));
        if (properties.containsKey("wideMtReorderedThreshold"))
            wideMtReorderedThreshold = Integer.parseInt(properties.getProperty("wideMtReorderedThreshold"));
        if (properties.containsKey("squareSequentialVecThreshold"))
            squareSequentialVecThreshold = Integer.parseInt(properties.getProperty("squareSequentialVecThreshold"));
        if (properties.containsKey("minDimSmallThreshold"))
            minDimSmallThreshold = Integer.parseInt(properties.getProperty("minDimSmallThreshold"));
    }

    private ReDeMatMultDispatchConfigs() {
        // Hide default constructor for utility clas.
    }


    /**
     * Gets the "aspect ratio" threshold for real dense matrix multiplication problems.
     * @return The "aspect ratio" threshold for real dense matrix multiplication problems.
     */
    public static double getAspectThreshold() {
        return aspectThreshold;
    }


    /**
     * Gets the threshold for considering a real dense matrix multiplication problem to be "small".
     * @return The threshold for considering a real dense matrix multiplication problem to be "small".
     */
    public static int getSmallThreshold() {
        return smallThreshold;
    }


    /**
     * Gets the threshold for using a standard concurrent kernel for a "near square" real dense matrix multiplication problem.
     * @return The threshold for using a standard concurrent kernel for a "near square" real dense matrix multiplication problem.
     */
    public static int getSquareMtStandardThreshold() {
        return squareMtStandardThreshold;
    }


    /**
     * Gets the threshold for using a reordered concurrent kernel for a "near square" real dense matrix multiplication problem.
     * @return The threshold for using a reordered concurrent kernel for a "near square" real dense matrix multiplication problem.
     */
    public static int getSquareMtReorderedThreshold() {
        return squareMtReorderedThreshold;
    }


    /**
     * Gets the threshold for using a reordered concurrent kernel for a "wide" real dense matrix multiplication problem.
     * @return The threshold for using a reordered concurrent kernel for a "wide" real dense matrix multiplication problem.
     */
    public static int getWideMtReorderedThreshold() {
        return wideMtReorderedThreshold;
    }


    /**
     * Gets the threshold for using a sequential kernel for a "near square" real dense matrix-vector multiplication problem.
     * @return The Gets the threshold for using a sequential kernel for a "near square" real dense matrix-vector
     * multiplication problem.
     */
    public static int getSquareSequentialVecThreshold() {
        return squareSequentialVecThreshold;
    }


    /**
     * Gets the threshold for considering a real dense matrix or matrix-vector multiplication problem "small".
     * @return The threshold for considering a real dense matrix or matrix-vector multiplication problem "small".
     */
    public static int getMinDimSmallThreshold() {
        return minDimSmallThreshold;
    }


    /**
     * Gets the cache size to use in the dispatcher for real dense matrix multiplication problems.
     * @return The cache size to use in the dispatcher for real dense matrix multiplication problems.
     */
    public static int getCacheSize() {
        return minDimSmallThreshold;
    }
}
