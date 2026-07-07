/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.arrays;

import java.util.List;

/**
 * <p>Data class for storing information for a sparse COO tensor.
 * <p>This record stores two ndarrays: the non-zero items, the non-zero indices.
 * @param shape Shape of the tensor.
 * @param data Non-zero items of the sparse tensor.
 * @param indices Non-zero indices of the sparse tensor.
 * @param <T> Type of the items of the tensor.
 */
public record SparseTensorData<T>(Shape shape, List<T> data, List<int[]> indices) {

    /**
     * Converts the indices of this sparse tensor items to a 2D primitive integer array.
     * @return A 2D primitive integer array containing the indices of this sparse tensor items.
     */
    public int[][] indicesToArray() {
        return indices.toArray(new int[indices.size()][]);
    }
}
