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

package org.flag4j.linalg.ops.dispatch;

import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractNDArray;

import java.util.function.BiFunction;


/**
 * Base class for all dispatchers which apply a binary tensor operation.
 * @param <T> Type of left tensor in tensor operation.
 * @param <U> Type of right tensor in tensor operation.
 * @param <R> Type of tensor resulting from the tensor operation on types {@link T} and {@link U}.
 */
public abstract class BiTensorOpDispatcher<
        T extends AbstractNDArray<T, ?, ?>, U extends AbstractNDArray<U, ?, ?>, R extends AbstractNDArray<R, ?, ?>> {
    /**
     * Default cache size for dispatchers.
     */
    protected static final int DEFAULT_CACHE_SIZE = 64;

    /**
     * LRU Cache backing this matrix multiplication dispatcher
     */
    final LRUCache<Pair<Shape, Shape>, BiFunction<T, U, R>> cache;


    /**
     * Creates a matrix multiplication dispatcher with the specified cacheSize.
     * @param cacheSize The size of the cache for this matrix multiplication dispatcher.
     */
    protected BiTensorOpDispatcher(int cacheSize) {
        cache = new LRUCache<>(cacheSize);
    }


    /**
     * Dispatches the matrix multiplication problem to the proper function and updates the cache if needed.
     * @param a First matrix in the matrix multiplication problem.
     * @param b Second matrix multiplication problem
     * @return The result of the matrix multiplication problem.
     */
    protected final R dispatch_(T a, U b) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();

        var key = new Pair<>(aShape, bShape);
        var func = cache.get(key);

        // If the function is not in the cache, update it.
        if (func == null) {
            // Only verify the shapes if we have not encountered them before.
            validateShapes(aShape, bShape);
            func = getFunc(key.first(), key.second(), a.dataLength(), b.dataLength());
            cache.put(key, func);
        }

        return func.apply(a, b);
    }


    /**
     * Validates the shapes are valid for the operation.
     * @param aShape Shape of first tensor in the operation.
     * @param bShape Shape of second tensor in the operation.
     */
    protected abstract void validateShapes(Shape aShape, Shape bShape);


    /**
     * Computes the appropriate function to use when computing the tensor operation between two tensors.
     * @param aShape Shape of the first tensor in the operation.
     * @param bShape Shape of the second tensor in the operation.
     * @param data1Length Full length of the items array within the first tensor.
     * @param data2Length Full length of the items array within the second tensor.
     * @return The appropriate function to use when computing the tensor operation.
     */
    protected abstract BiFunction<T, U, R> getFunc(Shape aShape, Shape bShape, int data1Length, int data2Length);
}
