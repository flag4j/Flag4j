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

package org.flag4j.rng;


import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Static class containing the global seed and default random number generator in Flag4j.
 */
public final class RandomState {
    /**
     * Global RNG seed for Flag4j.
     */
    private static final AtomicLong globalSeed = new AtomicLong(new Random().nextLong());
    /**
     * Random number generator instance for this thread.
     */
    private static final ThreadLocal<RandomComplex> threadLocalRandom = ThreadLocal.withInitial(() ->
            new RandomComplex(globalSeed.get()));

    private RandomState() {
        // Hide default constructor for static class.
    }


    /**
     * Sets the global seed for Flag4j.
     * @param seed The new value to use for the global seed.
     */
    public static void setGlobalSeed(long seed) {
        globalSeed.set(seed);
        threadLocalRandom.remove();
    }


    /**
     * Gets the global seed for Flag4j.
     * @return The current value of the global seed.
     */
    public static long getGlobalSeed() {
        return globalSeed.get();
    }


    /**
     * Gets a thread-local instance of ComplexRandom.
     * @return A Random instance.
     */
    public static RandomComplex getDefaultRng() {
        return threadLocalRandom.get();
    }
}
