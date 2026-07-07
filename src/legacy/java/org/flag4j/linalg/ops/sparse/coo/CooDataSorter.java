/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo;

import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.util.ArrayConversions;

import java.util.*;

/**
 * <p>A class which can be used to sort the indices, along with items values, in lexicographical order.
 * <p>This class is <em>not</em> thread safe.
 * <p>Specifically, if a sparse tensor has shape (10, 15, 5, 2) and the following indices and non-zero values,
 * <pre>
 *      - Indices: {{4, 1, 2, 0}, {4, 0, 1, 2},
 *                  {1, 2, 3, 0}, {2, 3, 5, 1},
 *                  {9, 10, 4, 1}, {1, 2, 1, 1}}
 *      - Values: {1.1, 2.2, 3.3, 4.4, 5.5, 6.6}
 * </pre>
 * then the sorted indices and non-zero values will be,
 * <pre>
 *      - Indices: {{1, 2, 1, 1}, {1, 2, 3, 0},
 *                  {2, 3, 5, 1}, {4, 0, 1, 2},
 *                  {4, 1, 2, 0}, {9, 10, 4, 1}}
 *      - Values: {6.6, 3.3, 4.4, 2.2, 1.1, 5.5}
 * </pre>
 */
public class CooDataSorter<T> {

    /**
     * Non-zero values of the sparse tensor to sort.
     */
    private final List<T> values;
    /**
     * Stores wrapped indices. Each list is a single index for the {@code values} items. If the tensor is of rank {@code N}, then
     * each List in {@code keys} will have {@code N} items. If there are {@code M} non-zero items in the
     * sparse tensor, then there will be {@code M} lists in the array.
     */
    private final List<Integer>[] keys;


    /**
     * <p>Creates a sorter which can sort the indices and values of a COO matrix lexicographically.
     * <p>When {@link #sparseSort()} is called on an instance created with this constructor, the sorting will be done in-place.
     * @param values Non-zero values of the sparse COO matrix.
     * @param rowIndices Row indices of the sparse COO matrix.
     * @param colIndices Row indices of the sparse COO matrix.
     */
    public CooDataSorter(List<T> values, List<Integer> rowIndices, List<Integer> colIndices) {
        this.values = values;
        this.keys = new List[]{rowIndices, colIndices};
    }


    /**
     * Wraps the items, and it's associated indices, of a sparse tensor.
     * @param values Non-zero values of the sparse tensor.
     * @param indices Indices of non-zero values in the sparse tensor.
     * @param transpose Indicates if a transpose should be applied to the indices array.
     */
    private CooDataSorter(T[] values, int[][] indices, boolean transpose) {
        this.values = Arrays.asList(values);
        this.keys = transpose ? new List[indices[0].length] : new List[indices.length];

        int[][] indicesT = transpose ? RealDenseTranspose.blockedIntMatrix(indices) : indices;

        for(int i=0; i<indicesT.length; i++) {
            if(values.length != indicesT[i].length) {
                throw new IllegalArgumentException("All lists must have the same length.");
            }

            this.keys[i] = ArrayConversions.toArrayList(indicesT[i]);
        }
    }


    /**
     * Factory method which wraps items in an instance of {@link CooDataSorter} and returns that instance.
     * @param values Non-zero values of the sparse tensor.
     * @param indices Indices of non-zero values in the sparse tensor.
     * @return A new instance of {@link CooDataSorter} which wraps the specified {@code values} and {@code indices}.
     */
    public static CooDataSorter wrap(Object[] values, int[][] indices) {
        return new CooDataSorter(values, indices, true);
    }


    /**
     * Factory method which wraps items in an instance of {@link CooDataSorter} and returns that instance.
     * @param values Non-zero values of the sparse tensor.
     * @param indices Indices of non-zero values in the sparse tensor.
     * @return A new instance of {@link CooDataSorter} which wraps the specified {@code values} and {@code indices}.
     */
    public static CooDataSorter wrap(Object[] values, int[] indices) {
        return new CooDataSorter(values, new int[][]{indices}, false);
    }


    /**
     * Factory method which wraps items in an instance of {@link CooDataSorter} and returns that instance.
     * @param values Non-zero values of the sparse tensor.
     * @param indices Indices of non-zero values in the sparse tensor.
     * @return A new instance of {@link CooDataSorter} which wraps the specified {@code values} and {@code indices}.
     */
    public static CooDataSorter wrap(double[] values, int[][] indices) {
        // Wrap the primitive array.
        return new CooDataSorter(Arrays.stream(values).boxed().toArray(Double[]::new), indices, true);
    }


    /**
     * Factory method which wraps items in an instance of {@link CooDataSorter} and returns that instance.
     * @param values Non-zero values of the sparse tensor.
     * @param indices Indices of non-zero values in the sparse tensor.
     * @return A new instance of {@link CooDataSorter} which wraps the specified {@code values} and {@code indices}.
     */
    public static CooDataSorter wrap(double[] values, int[] indices) {
        // Wrap the primitive array.
        return new CooDataSorter(Arrays.stream(values).boxed().toArray(Double[]::new), new int[][]{indices}, false);
    }


    /**
     * Factory method which wraps items in an instance of {@link CooDataSorter} and returns that instance.
     * @param values Non-zero values of the sparse tensor.
     * @param rowIndices Row indices of non-zero items of the sparse tensor.
     * @param colIndices Column indices of non-zero items of the sparse tensor.
     * @return A new instance of {@link CooDataSorter} which wraps the specified {@code values} and {@code indices}.
     */
    public static CooDataSorter wrap(double[] values, int[] rowIndices, int[] colIndices) {
        // Wrap the primitive array.
        return new CooDataSorter(Arrays.stream(values).boxed().toArray(Double[]::new), new int[][]{rowIndices, colIndices}, false);
    }


    /**
     * Factory method which wraps items in an instance of {@link CooDataSorter} and returns that instance.
     * @param values Non-zero values of the sparse tensor.
     * @param rowIndices Row indices of non-zero items of the sparse tensor.
     * @param colIndices Column indices of non-zero items of the sparse tensor.
     * @return A new instance of {@link CooDataSorter} which wraps the specified {@code values} and {@code indices}.
     */
    public static CooDataSorter wrap(Object[] values, int[] rowIndices, int[] colIndices) {
        // Wrap the primitive array.
        return new CooDataSorter(values, new int[][]{rowIndices, colIndices}, false);
    }


    /**
     * Sorts the wrapped indices in lexicographical order while maintaining the order of the non-zero values so that
     * their order matches the order of the indices.
     * @return A reference to this sparse items wrapper.
     */
    public CooDataSorter sparseSort() {
        if(values.size() >= 2) {
            // Only need to sort list with more than 1 entry.
            sparseSortHelper(0, 0, values.size());
        }

        return this;
    }


    /**
     * Sorts the specified key list over a specified range. During sorting, when swaps are made in the key list, the
     * same indices are swapped in all other lists within the wrapper. This is done recursively top down.
     * @param keyIdx Index of key list within {@link #keys}.
     * @param start Staring index of range to sort the key list over (inclusive).
     * @param stop Stopping index of range to sort the key list over (exclusive).
     */
    private void sparseSortHelper(int keyIdx, int start, int stop) {
        if(start>=stop-1 || keyIdx>=keys.length || keyIdx<0) {
            return; // No sorting to do.
        }

        List<Integer> key = keys[keyIdx].subList(start, stop);

        // Create a List of indices.
        List<Integer> indices = new ArrayList<>();
        for(int i=0; i<key.size(); i++)
            indices.add(i);

        // Sort the indices list based on the key list.
        indices.sort(Comparator.comparingInt(key::get));

        Map<Integer, Integer> swapMap = new HashMap<>(indices.size());
        List<Integer> swapFrom = new ArrayList<>(indices.size()),
                swapTo = new ArrayList<>(indices.size());

        // Create map to facilitate reordering list to match sorted indices.
        for(int i=0; i<key.size(); i++) {
            int k = indices.get(i);

            while(i != k && swapMap.containsKey(k))
                k = swapMap.get(k);

            swapFrom.add(i);
            swapTo.add(k);
            swapMap.put(i, k);
        }

        // Use map to reorder values sub-list.
        List<T> valuesSubList = values.subList(start, stop);
        for(int i=0; i<key.size(); i++) {
            Collections.swap(valuesSubList, swapFrom.get(i), swapTo.get(i));
        }

        // Use map to reorder each key sub-list.
        for(List<Integer> list : keys) {
            List<Integer> subList = list.subList(start, stop);

            for (int i = 0; i < key.size(); i++)
                Collections.swap(subList, swapFrom.get(i), swapTo.get(i));
        }

        // Find ranges which have the same value in the sorted key list.
        int startCounter = 0;
        int endCounter = 1;
        while(endCounter<key.size()) {
            if(!key.get(startCounter).equals(key.get(endCounter))) {
                if(endCounter-startCounter > 1) {
                    // Sort the range of the next key list.
                    sparseSortHelper(keyIdx+1, startCounter+start, endCounter+start);
                }

                startCounter=endCounter;
            }

            endCounter++;
        }

        // Check for final range of same values.
        if(startCounter!=endCounter-1) {
            // Sort the range of the next key list.
            sparseSortHelper(keyIdx+1, startCounter+start, endCounter+start);
        }
    }


    /**
     * Unwraps sparse items values and indices.
     * @param values Storage for unwrapped values. Must have the same length as that passed to the constructor. Modified.
     * @param indices Storage for unwrapped indices. Must have the same shape as that passed to the constructor. Modified.
     */
    public void unwrap(Object[] values, int[][] indices) {
        // Copy over items values.
        for(int i=0; i<this.values.size(); i++)
            values[i] = this.values.get(i);

        // Copy over indices (must be transposed).
        for(int i=0; i<indices.length; i++) {
            for(int j=0; j<indices[0].length; j++) {
                indices[i][j] = this.keys[j].get(i);
            }
        }
    }


    /**
     * Unwraps sparse items values and indices.
     * @param values Storage for unwrapped values. Must have the same length as that passed to the constructor. Modified.
     * @param indices Storage for unwrapped indices. Must have the same shape as that passed to the constructor. Modified.
     */
    @SuppressWarnings("unused")
    public void unwrap(Object[] values, int[] indices) {
        // Copy over items values.
        for(int i=0; i<this.values.size(); i++)
            values[i] = this.values.get(i);

        // Copy over indices (must be transposed).
        for(int i=0; i<indices.length; i++)
            indices[i] = this.keys[0].get(i);
    }


    /**
     * Unwraps sparse items values and indices.
     * @param values Storage for unwrapped values. Must have the same length as that passed to the constructor. Modified.
     * @param indices Storage for unwrapped indices. Must have the same shape as that passed to the constructor. Modified.
     */
    public void unwrap(double[] values, int[][] indices) {
        // Copy over items values.
        for(int i=0; i<this.values.size(); i++)
            values[i] = (double) this.values.get(i);

        // Copy over indices (must be transposed).
        for(int i=0; i<indices.length; i++) {
            for(int j=0; j<indices[0].length; j++) {
                indices[i][j] = this.keys[j].get(i);
            }
        }
    }


    /**
     * Unwraps sparse items values and indices.
     * @param values Storage for unwrapped values. Must have the same length as that passed to the constructor. Modified.
     * @param rowIndices Storage for unwrapped row indices. Must have the same shape as that passed to the constructor. Modified.
     * @param colIndices Storage for unwrapped column indices. Must have the same shape as that passed to the constructor. Modified.
     */
    public void unwrap(double[] values, int[] rowIndices, int[] colIndices) {
        // Copy over items values and indices.
        for(int i=0; i<this.values.size(); i++) {
            values[i] = (double) this.values.get(i);
            rowIndices[i] = this.keys[0].get(i);
            colIndices[i] = this.keys[1].get(i);
        }
    }


    /**
     * Unwraps sparse items values and indices.
     * @param values Storage for unwrapped values. Must have the same length as that passed to the constructor. Modified.
     * @param rowIndices Storage for unwrapped row indices. Must have the same shape as that passed to the constructor. Modified.
     * @param colIndices Storage for unwrapped column indices. Must have the same shape as that passed to the constructor. Modified.
     */
    public void unwrap(Object[] values, int[] rowIndices, int[] colIndices) {
        // Copy over items values and indices.
        for(int i=0; i<this.values.size(); i++) {
            values[i] = this.values.get(i);
            rowIndices[i] = this.keys[0].get(i);
            colIndices[i] = this.keys[1].get(i);
        }
    }


    /**
     * Unwraps sparse items values and indices.
     * @param values Storage for unwrapped values. Must have the same length as that passed to the constructor. Modified.
     * @param indices Storage for unwrapped indices. Must have the same shape as that passed to the constructor. Modified.
     */
    public void unwrap(double[] values, int[] indices) {
        // Copy over items values and indices.
        for(int i=0; i<this.values.size(); i++) {
            values[i] = (double) this.values.get(i);
            indices[i] = this.keys[0].get(i);
        }
    }
}
