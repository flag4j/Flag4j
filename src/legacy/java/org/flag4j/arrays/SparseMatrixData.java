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

import org.flag4j.util.ArrayConversions;

import java.util.List;


/**
 * <p>Data class for storing information for a sparse (CSR or COO) matrix.
 * <p>This record stores three lists: the non-zero items, the row indices/pointers, and the column indices.
 * @param shape Shape of the matrix.
 * @param data Non-zero items of the sparse matrix.
 * @param rowData Non-zero row indices/pointers.
 * @param colData Non-zero column indices.
 * @param <T> Type of the items of the matrix.
 */
public record SparseMatrixData<T>(Shape shape, List<T> data, List<Integer> rowData, List<Integer> colData) {


    /**
     * Converts the row indices of this sparse matrix items to a primitive integer array.
     * @return A primitive integer array containing the row indices of this sparse matrix items.
     */
    public int[] rowIndicesToArray() {
        return ArrayConversions.fromIntegerList(rowData);
    }


    /**
     * Converts the column indices of this sparse matrix items to a primitive integer array.
     * @return A primitive integer array containing the column indices of this sparse matrix items.
     */
    public int[] colIndicesToArray() {
        return ArrayConversions.fromIntegerList(colData);
    }
}
