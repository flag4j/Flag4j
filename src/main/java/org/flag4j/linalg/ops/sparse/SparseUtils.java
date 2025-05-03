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

package org.flag4j.linalg.ops.sparse;

import org.flag4j.arrays.*;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCsrSemiringMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrFieldMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ValidateParameters;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.*;
import java.util.function.BinaryOperator;

/**
 * <p>Contains common utility functions for working with sparse matrices.
 *
 * <p>All methods in this class which accept non-zero data and non-zero indices, they are assumed to be sorted
 * lexicographically. Similarly, all ops which store results in user provided destination arrays will also produce in
 * lexicographically sorted arrays.
 */
public final class SparseUtils {

    public SparseUtils() {
        // Utility class cannot be instanced.
    }


    /**
     * Computes the sparsity of a sparse tensor with the specified shape and non-zero entries.
     * @param shape The shape of the sparse tensor.
     * @param nnz The number of non-zero entries in the sparse tensor.
     * @return The sparsity of a sparse tensor with the specified {@code shape} and {@code nnz}.
     */
    public static double computeSparsity(Shape shape, int nnz) {
        BigInteger totalEntries = shape.totalEntries();
        BigDecimal sparsity = new BigDecimal(totalEntries).subtract(BigDecimal.valueOf(nnz));
        sparsity = sparsity.divide(new BigDecimal(totalEntries), RoundingMode.HALF_UP);
        return sparsity.doubleValue();
    }


    /**
     * Flattens the non-zero indices of a sparse COO tensor. This reduced the rank of this tensor to rank 1.
     * @param Shape Shape of the sparse COO tensor.
     * @param indices Shape of the sparse COO tensor.
     * @return The flattened indices of the sparse COO tensor.
     * @see #cooFlattenIndices(Shape, int[][], int)
     */
    public static int[][] cooFlattenIndices(Shape shape, int[][] indices) {
        int[][] destIndices = new int[indices.length][1];

        for(int i=0, size=indices.length; i<size; i++)
            destIndices[i][0] = shape.get1DIndex(indices[i]);

        return destIndices;
    }


    /**
     * Flattens the non-zero indices of a sparse COO tensor along a specified axis. Unlike
     * {@link #cooFlattenIndices(Shape, int[][])} this method preserves the rank of the tensor.
     * @param shape Shape of the sparse COO tensor.
     * @param indices Shape of the sparse COO tensor.
     * @param axis Axis along which to flatten the tensor.
     * @return The non-zero indices flattened along the specified {@code axis}.
     */
    public static int[][] cooFlattenIndices(Shape shape, int[][] indices, int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[][] destIndices = new int[indices.length][indices[0].length];

        for(int i=0, size=indices.length; i<size; i++)
            destIndices[i][axis] = shape.get1DIndex(indices[i]);

        return destIndices;
    }


    /**
     * Computes new indices for the reshaping of a sparse COO tensor.
     * @param oldShape Current shape (before reshaping) of the sparse COO tensor.
     * @param newShape New shape of the tensor.
     * @param indices The non-zero indices of the tensor to be reshaped.
     * @return The non-zero indices corresponding to the reshaped tensor.
     */
    public static int[][] cooReshape(Shape oldShape, Shape newShape, int[][] indices) {
        ValidateParameters.ensureTotalEntriesEqual(oldShape, newShape);

        int rank = indices[0].length;
        int newRank = newShape.getRank();
        int nnz = indices.length;

        int[] oldStrides = oldShape.getStrides();
        int[] newStrides = newShape.getStrides();

        int[][] newIndices = new int[nnz][newRank];

        for(int i=0; i<nnz; i++) {
            int[] idxRow = indices[i];
            int[] newIdxRow = newIndices[i];

            int flatIndex = 0;
            for(int j=0; j < rank; j++) {
                flatIndex += idxRow[j] * oldStrides[j];
            }

            for(int j=0; j<newRank; j++) {
                newIdxRow[j] = flatIndex / newStrides[j];
                flatIndex %= newStrides[j];
            }
        }

        return newIndices;
    }


    /**
     * Creates a HashMap where the keys are row indices and the value is a list of all indices in src with that row
     * index.
     * @param nnz Number of non-zero data in the sparse matrix.
     * @param rowIndices Row indices of sparse matrix.
     * @return A HashMap where the keys are row indices and the value is a list of all indices in {@code src} with that row
     * index.
     */
    public static Map<Integer, List<Integer>> createMap(int nnz, int[] rowIndices) {
        Map<Integer, List<Integer>> map = new HashMap<>();

        for(int j=0; j<nnz; j++) {
            int r2 = rowIndices[j]; // = k
            map.computeIfAbsent(r2, x -> new ArrayList<>()).add(j);
        }

        return map;
    }


    /**
     * Sorts the non-zero data and column indices of a sparse CSR matrix lexicographically by row and column index. The row
     * pointers in the CSR matrix are assumed to be correct already.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Row pointer array of the CSR matrix. Stores the starting index for each row of the CSR matrix in {@code data}
     * and
     * @param colIndices Non-zero column indices of the CSR matrix.
     */
    public static void sortCsrMatrix(double[] entries, int[] rowPointers, int[] colIndices) {
        for (int row = 0; row < rowPointers.length - 1; row++) {
            int start = rowPointers[row];
            int end = rowPointers[row + 1];

            // Create an array of indices for sorting
            Integer[] indices = new Integer[end - start];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = start + i;
            }

            // Sort the indices based on the corresponding colData data
            Arrays.sort(indices, (i, j) -> Integer.compare(colIndices[i], colIndices[j]));

            // Reorder colData and data based on sorted indices
            int[] sortedColIndex = new int[end - start];
            double[] sortedValues = new double[end - start];
            for (int i = 0; i < indices.length; i++) {
                sortedColIndex[i] = colIndices[indices[i]];
                sortedValues[i] = entries[indices[i]];
            }

            // Copy sorted arrays back to the original colData and data
            System.arraycopy(sortedColIndex, 0, colIndices, start, sortedColIndex.length);
            System.arraycopy(sortedValues, 0, entries, start, sortedValues.length);
        }
    }


    /**
     * Sorts the non-zero data and column indices of a sparse CSR matrix lexicographically by row and column index. The row
     * pointers in the CSR matrix are assumed to be correct already.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Row pointer array of the CSR matrix. Stores the starting index for each row of the CSR matrix in {@code data}
     * and
     * @param colIndices Non-zero column indices of the CSR matrix.
     */
    public static void sortCsrMatrix(Object[] entries, int[] rowPointers, int[] colIndices) {
        for (int row = 0; row < rowPointers.length - 1; row++) {
            int start = rowPointers[row];
            int end = rowPointers[row + 1];

            // Create an array of indices for sorting
            Integer[] indices = new Integer[end - start];
            for (int i = 0; i < indices.length; i++)
                indices[i] = start + i;

            // Sort the indices based on the corresponding colData data
            Arrays.sort(indices, (i, j) -> Integer.compare(colIndices[i], colIndices[j]));

            // Reorder colData and data based on sorted indices
            int[] sortedColIndex = new int[end - start];
            Object[] sortedValues = new Object[end - start];
            for (int i = 0; i < indices.length; i++) {
                sortedColIndex[i] = colIndices[indices[i]];
                sortedValues[i] = entries[indices[i]];
            }

            // Copy sorted arrays back to the original colData and data
            System.arraycopy(sortedColIndex, 0, colIndices, start, sortedColIndex.length);
            System.arraycopy(sortedValues, 0, entries, start, sortedValues.length);
        }
    }


    /**
     * <p>Checks if two {@link CsrMatrix CSR Matrices} are equal considering the fact that one may explicitly store zeros at some
     * position that the other does not store.
     *
     * <p>
     * If zeros are explicitly stored at some position in only one matrix, it will be checked that the
     * other matrix does not have a non-zero value at the same index. If it does, the matrices will not be equal. If no value is
     * stored for that index then it is assumed to be zero by definition of the CSR format and the equality check continues with the
     * remaining values.
     * 
     *
     * @param src1 First CSR matrix in the equality comparison.
     * @param src2 Second CSR matrix in the equality comparison.
     * @return True if all non-zero values stored in the two matrices are equal and occur at the same indices.
     */
    public static boolean CSREquals(CsrMatrix src1, CsrMatrix src2) {
        if(!src1.shape.equals(src2.shape)) return false;

        // Compare row by row
        for (int i=0; i<src1.numRows; i++) {
            int src1RowStart = src1.rowPointers[i];
            int src1RowEnd = src1.rowPointers[i + 1];
            int src2RowStart = src2.rowPointers[i];
            int src2RowEnd = src2.rowPointers[i + 1];

            int src1Index = src1RowStart;
            int src2Index = src2RowStart;

            while (src1Index < src1RowEnd || src2Index < src2RowEnd) {
                // Skip explicit zeros in both matrices
                while (src1Index < src1RowEnd && src1.data[src1Index] == 0)
                    src1Index++;

                while (src2Index < src2RowEnd && src2.data[src2Index] == 0)
                    src2Index++;

                if (src1Index < src1RowEnd && src2Index < src2RowEnd) {
                    int src1Col = src1.colIndices[src1Index];
                    int src2Col = src2.colIndices[src2Index];
                    double src1Value = src1.data[src1Index];
                    double src2Value = src2.data[src2Index];

                    if(src1Col != src2Col || Double.compare(src1Value, src2Value) != 0) {
                        return false;
                    }

                    src1Index++;
                    src2Index++;
                } else if (src1Index < src1RowEnd) {
                    // Remaining data in src1 row
                    if (src1.data[src1Index] != 0) return false;

                    src1Index++;
                } else if (src2Index < src2RowEnd) {
                    // Remaining data in src2 row
                    if (src2.data[src2Index] != 0) return false;

                    src2Index++;
                }
            }
        }

        return true;
    }


    /**
     * <p>Checks if two {@link CsrFieldMatrix CSR Field matrices} are equal considering the fact that one may explicitly store zeros at
     * some position that the other does not store.
     *
     * <p>
     * If zeros are explicitly stored at some position in only one matrix, it will be checked that the
     * other matrix does not have a non-zero value at the same index. If it does, the matrices will not be equal. If no value is
     * stored for that index then it is assumed to be zero by definition of the CSR format and the equality check continues with the
     * remaining values.
     * 
     *
     * @param src1 First CSR matrix in the equality comparison.
     * @param src2 Second CSR matrix in the equality comparison.
     * @return True if all non-zero values stored in the two matrices are equal and occur at the same indices.
     */
    public static <T extends Semiring<T>> boolean CSREquals(
            AbstractCsrSemiringMatrix<?, ?, ?, T> src1,
            AbstractCsrSemiringMatrix<?, ?, ?, T> src2) {
        if(src1 == src2) return true;
        if(!src1.shape.equals(src2.shape)) return false;

        return CooSemiringEquals.cooMatrixEquals(
                src1.toCoo(),
                src2.toCoo());
    }


    /**
     * A helper method which copies from a sparse COO matrix to a set of three arrays (non-zero data, row indices, and
     * column indices) but skips over a specified range.
     * @param srcEntries Non-zero matrix to copy ranges from.
     * @param srcRowIndices Row indices of matrix to copy ranges from.
     * @param srcColIndices Column indices of matrix to copy ranges from.
     * @param destEntries Array to copy {@code} src non-zero data to.
     * @param destRowIndices Array to copy {@code} src row indices data to.
     * @param destColIndices Array to copy {@code} src column indices data to.
     * @param startEnd An array of length two specifying the {@code start} (inclusive) and {@code end} (exclusive)
     *                 indices of the range to skip during the copy.
     */
    public static <T> void copyRanges(
            T[] srcEntries, int[] srcRowIndices, int[] srcColIndices,
            T[] destEntries, int[] destRowIndices, int[] destColIndices, IntPair startEnd) {
        int start = startEnd.first();
        int end = startEnd.second();

        if(start > 0) {
            System.arraycopy(srcEntries, 0, destEntries, 0, start);
            System.arraycopy(srcEntries, end, destEntries, start, destEntries.length - start);

            System.arraycopy(srcRowIndices, 0, destRowIndices, 0, start);
            System.arraycopy(srcRowIndices, end, destRowIndices, start, destEntries.length - start);

            System.arraycopy(srcColIndices, 0, destColIndices, 0, start);
            System.arraycopy(srcColIndices, end, destColIndices, start, destEntries.length - start);
        } else {
            System.arraycopy(srcEntries, 0, destEntries, 0, destEntries.length);
            System.arraycopy(srcRowIndices, 0, destRowIndices, 0, destRowIndices.length);
            System.arraycopy(srcColIndices, 0, destColIndices, 0, destColIndices.length);
        }
    }


    /**
     * Coalesces this sparse COO matrix. An uncoalesced matrix is a sparse matrix with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data.
     * @param aggregator Custom aggregation function to combine multiple.
     * @param shape Shape of the COO matrix.
     * @param data The non-zero data of the COO matrix.
     * @param rowIndices The non-zero row indices of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @return A {@link SparseMatrixData} object containing the data of the COO matrix resulting from the coalesce operation.
     */
    public static <T> SparseMatrixData<T> coalesce(
            BinaryOperator<T> aggregator, Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        HashMap<Pair<Integer, Integer>, T> coalescedValues = new LinkedHashMap<>();
        List<Integer> destRowIndices = new ArrayList<>(data.length);
        List<Integer> destColIndices = new ArrayList<>(data.length);

        for(int i = 0; i<data.length; i++) {
            Pair<Integer, Integer> idx = new Pair<>(rowIndices[i], colIndices[i]);
            T value = data[i];

            if(coalescedValues.containsKey(idx)) {
                // The index is already present.
                value = aggregator.apply(coalescedValues.get(idx),  value);
            } else {
                destRowIndices.add(idx.first());
                destColIndices.add(idx.second());
            }

            coalescedValues.put(idx, value);
        }

        List<T> destValues = new ArrayList<>(coalescedValues.values());

        return new SparseMatrixData<>(shape, destValues, destRowIndices, destColIndices);
    }


    /**
     * Coalesces this sparse COO tensor. An uncoalesced matrix is a sparse tensor with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data.
     * @param aggregator Custom aggregation function to combine multiple.
     * @param shape Shape of the COO tensor.
     * @param data The non-zero data of the COO tensor.
     * @param rowIndices The non-zero row indices of the COO tensor.
     * @param colIndices The non-zero column indices of the COO tensor.
     * @return A {@link SparseTensorData} object containing the data of the COO tensor resulting from the coalesce operation.
     */
    public static <T> SparseTensorData<T> coalesce(
            BinaryOperator<T> aggregator, Shape shape, T[] data, int[][] indices) {
        HashMap<IntTuple, T> coalescedValues = new HashMap<>();
        List<int[]> destIndices = new ArrayList<>(data.length);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            IntTuple idx = new IntTuple(indices[i].clone());
            T value = data[i];

            if(coalescedValues.containsKey(idx)) {
                // The index is already present.
                value = aggregator.apply(coalescedValues.get(idx),  value);
            } else {
                destIndices.add(idx.data());
            }

            coalescedValues.put(idx, value);
        }

        List<T> destValues = new ArrayList<>(coalescedValues.values());

        return new SparseTensorData<>(shape, destValues, destIndices);
    }


    /**
     * Coalesces this sparse COO vector. An uncoalesced matrix is a sparse vector with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data.
     * @param aggregator Custom aggregation function to combine multiple.
     * @param shape Shape of the COO vector.
     * @param data The non-zero data of the COO vector.
     * @param rowIndices The non-zero row indices of the COO vector.
     * @param colIndices The non-zero column indices of the COO vector.
     * @return A {@link SparseVectorData} object containing the data of the COO vector resulting from the coalesce operation.
     */
    public static <T> SparseVectorData<T> coalesce(
            BinaryOperator<T> aggregator, Shape shape, T[] data, int[] indices) {
        HashMap<Integer, T> coalescedValues = new HashMap<>();
        List<Integer> destIndices = new ArrayList<>(data.length);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            int idx = indices[i];
            T value = data[i];

            if(coalescedValues.containsKey(idx)) {
                // The index is already present.
                value = aggregator.apply(coalescedValues.get(idx),  value);
            } else {
                destIndices.add(idx);
            }

            coalescedValues.put(idx, value);
        }

        List<T> destValues = new ArrayList<>(coalescedValues.values());

        return new SparseVectorData<>(shape, destValues, destIndices);
    }


    /**
     * Drops any explicit zeros in this sparse COO matrix.
     @return A {@link SparseMatrixData} object containing the data of the COO matrix resulting from dropping all explicit zeros.
     */
    public static <T extends Semiring<T>> SparseMatrixData<T> dropZeros(Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        int estSize = data.length / 2;
        List<T> destValues = new ArrayList<>(estSize);
        List<Integer> destRowIndices = new ArrayList<>(estSize);
        List<Integer> destColIndices = new ArrayList<>(estSize);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            T v = data[i];

            if(!v.isZero()) {
                destValues.add(v);
                destRowIndices.add(rowIndices[i]);
                destColIndices.add(colIndices[i]);
            }
        }

        return new SparseMatrixData<>(shape, destValues, destRowIndices, destColIndices);
    }


    /**
     * Drops any explicit zeros in this sparse COO tensor.
     * @return A {@link SparseTensorData} object containing the data of the COO tensor resulting from dropping all explicit zeros.
     */
    public static <T extends Semiring<T>> SparseTensorData<T> dropZeros(Shape shape, T[] data, int[][] indices) {
        int estSize = data.length / 2;
        List<T> destValues = new ArrayList<>(estSize);
        List<int[]> destIndices = new ArrayList<>(estSize);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            T v = data[i];

            if(!v.isZero()) {
                destValues.add(v);
                destIndices.add(indices[i].clone());
            }
        }

        return new SparseTensorData<>(shape, destValues, destIndices);
    }


    /**
     * Drops any explicit zeros in this sparse COO vector.
     * @return A {@link SparseVectorData} object containing the data of the COO vector resulting from dropping all explicit zeros.
     */
    public static <T extends Semiring<T>> SparseVectorData<T> dropZeros(Shape shape, T[] data, int[] indices) {
        int estSize = data.length / 2;
        List<T> destValues = new ArrayList<>(estSize);
        List<Integer> destIndices = new ArrayList<>(estSize);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            T v = data[i];

            if(!v.isZero()) {
                destValues.add(v);
                destIndices.add(indices[i]);
            }
        }

        return new SparseVectorData<>(shape, destValues, destIndices);
    }


    /**
     * Coalesces this sparse COO matrix. An uncoalesced matrix is a sparse matrix with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data.
     * @param aggregator Custom aggregation function to combine multiple.
     * @param shape Shape of the COO matrix.
     * @param data The non-zero data of the COO matrix.
     * @param rowIndices The non-zero row indices of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @return A {@link SparseMatrixData} object containing the data of the COO matrix resulting from the coalesce operation.
     */
    public static SparseMatrixData<Double> coalesce(
            BinaryOperator<Double> aggregator, Shape shape, double[] data, int[] rowIndices, int[] colIndices) {
        HashMap<Pair<Integer, Integer>, Double> coalescedValues = new HashMap<>();
        List<Integer> destRowIndices = new ArrayList<>(data.length);
        List<Integer> destColIndices = new ArrayList<>(data.length);

        for(int i = 0; i< data.length; i++) {
            Pair<Integer, Integer> idx = new Pair<>(rowIndices[i], colIndices[i]);
            double value = data[i];

            if(coalescedValues.containsKey(idx)) {
                // The index is already present.
                value = aggregator.apply(coalescedValues.get(idx),  value);
            } else {
                destRowIndices.add(idx.first());
                destColIndices.add(idx.second());
            }

            coalescedValues.put(idx, value);
        }

        List<Double> destValues = new ArrayList<>(coalescedValues.values());

        return new SparseMatrixData<>(shape, destValues, destRowIndices, destColIndices);
    }


    /**
     * Coalesces this sparse COO tensor. An uncoalesced matrix is a sparse tensor with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data.
     * @param aggregator Custom aggregation function to combine multiple.
     * @param shape Shape of the COO tensor.
     * @param data The non-zero data of the COO tensor.
     * @param rowIndices The non-zero row indices of the COO tensor.
     * @param colIndices The non-zero column indices of the COO tensor.
     * @return A {@link SparseTensorData} object containing the data of the COO tensor resulting from the coalesce operation.
     */
    public static SparseTensorData<Double> coalesce(
            BinaryOperator<Double> aggregator, Shape shape, double[] data, int[][] indices) {
        HashMap<IntTuple, Double> coalescedValues = new HashMap<>();
        List<int[]> destIndices = new ArrayList<>(data.length);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            IntTuple idx = new IntTuple(indices[i].clone());
            double value = data[i];

            if(coalescedValues.containsKey(idx)) {
                // The index is already present.
                value = aggregator.apply(coalescedValues.get(idx),  value);
            } else {
                destIndices.add(idx.data());
            }

            coalescedValues.put(idx, value);
        }

        List<Double> destValues = new ArrayList<>(coalescedValues.values());

        return new SparseTensorData<>(shape, destValues, destIndices);
    }


    /**
     * Drops any explicit zeros in this sparse COO matrix.
     @return A {@link SparseMatrixData} object containing the data of the COO matrix resulting from dropping all explicit zeros.
     */
    public static SparseMatrixData<Double> dropZeros(Shape shape, double[] data, int[] rowIndices, int[] colIndices) {
        int estSize = data.length / 2;
        List<Double> destValues = new ArrayList<>(estSize);
        List<Integer> destRowIndices = new ArrayList<>(estSize);
        List<Integer> destColIndices = new ArrayList<>(estSize);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            double v = data[i];

            if(v != 0.0) {
                destValues.add(v);
                destRowIndices.add(rowIndices[i]);
                destColIndices.add(colIndices[i]);
            }
        }

        return new SparseMatrixData<>(shape, destValues, destRowIndices, destColIndices);
    }


    /**
     * Drops any explicit zeros in this sparse COO tensor.
     * @return A {@link SparseTensorData} object containing the data of the COO tensor resulting from dropping all explicit zeros.
     */
    public static SparseTensorData<Double> dropZeros(Shape shape, double[] data, int[][] indices) {
        int estSize = data.length / 2;
        List<Double> destValues = new ArrayList<>(estSize);
        List<int[]> destIndices = new ArrayList<>(estSize);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            double v = data[i];

            if(v != 0.0) {
                destValues.add(v);
                destIndices.add(indices[i].clone());
            }
        }

        return new SparseTensorData<>(shape, destValues, destIndices);
    }


    /**
     * Coalesces this sparse COO vector. An uncoalesced matrix is a sparse vector with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data.
     * @param aggregator Custom aggregation function to combine multiple.
     * @param shape Shape of the COO vector.
     * @param data The non-zero data of the COO vector.
     * @param rowIndices The non-zero row indices of the COO vector.
     * @param colIndices The non-zero column indices of the COO vector.
     * @return A {@link SparseVectorData} object containing the data of the COO vector resulting from the coalesce operation.
     */
    public static SparseVectorData<Double> coalesce(
            BinaryOperator<Double> aggregator, Shape shape, double[] data, int[] indices) {
        HashMap<Integer, Double> coalescedValues = new HashMap<>();
        List<Integer> destIndices = new ArrayList<>(data.length);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            int idx = indices[i];
            double value = data[i];

            if(coalescedValues.containsKey(idx)) // The index is already present.
                value = aggregator.apply(coalescedValues.get(idx),  value);
            else
                destIndices.add(idx);


            coalescedValues.put(idx, value);
        }

        List<Double> destValues = new ArrayList<>(coalescedValues.values());

        return new SparseVectorData<>(shape, destValues, destIndices);
    }


    /**
     * Drops any explicit zeros in this sparse COO vector.
     * @return A {@link SparseVectorData} object containing the data of the COO vector resulting from dropping all explicit zeros.
     */
    public static SparseVectorData<Double> dropZeros(Shape shape, double[] data, int[] indices) {
        int estSize = data.length / 2;
        List<Double> destValues = new ArrayList<>(estSize);
        List<Integer> destIndices = new ArrayList<>(estSize);

        for(int i=0, nnz=data.length; i<nnz; i++) {
            double v = data[i];

            if(v != 0.0) {
                destValues.add(v);
                destIndices.add(indices[i]);
            }
        }

        return new SparseVectorData<>(shape, destValues, destIndices);
    }


    /**
     * Drops all explicit zeros within a sparse CSR matrix.
     * @param src Matrix to drop zeros from.
     * @return A copy of the {@code src} matrix with all zeros dropped.
     */
    public static CsrMatrix dropZerosCsr(CsrMatrix src) {
        List<Double> destData = new ArrayList<>(src.data.length);
        List<Integer> destRowIndices = new ArrayList<>(src.data.length);
        List<Integer> destColIndices = new ArrayList<>(src.data.length);

        // Construct as COO matrix then convert to CSR to simplify implementation.
        for(int i=0, rows=src.numRows; i<rows; i++) {
            for(int j=src.rowPointers[i], stop=src.rowPointers[i + 1]; i<stop; i++) {
                if(src.data[j] != 0.0) {
                    destData.add(src.data[j]);
                    destRowIndices.add(i);
                    destColIndices.add(src.colIndices[i]);
                }
            }
        }

        return new CooMatrix(src.shape, destData, destRowIndices, destColIndices).toCsr();
    }


    /**
     * Drops all explicit zeros within a sparse CSR matrix.
     * @param shape Shape of the CSR matrix.
     * @param data Non-zero entries of the CSR matrix.
     * @param rowPointers Non-zero row indices of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A {@link SparseMatrixData} containing the data of the CSR matrix resulting from dropping all zeros from the specified
     * CSR matrix. Note, the returned data will be in COO format and <em>not</em> CSR.
     */
    public static <T extends Semiring<T>> SparseMatrixData<T> dropZerosCsr(
            Shape shape, T[] data, int[] rowPointers, int[] colIndices) {
        List<T> destData = new ArrayList<>(data.length);
        List<Integer> destRowIndices = new ArrayList<>(data.length);
        List<Integer> destColIndices = new ArrayList<>(data.length);

        // Construct as COO matrix then convert to CSR to simplify implementation.
        for(int i=0, rows=rowPointers.length-1; i<rows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i + 1]; i<stop; i++) {
                if(data[j].isZero()) {
                    destData.add(data[j]);
                    destRowIndices.add(i);
                    destColIndices.add(colIndices[i]);
                }
            }
        }

        return new SparseMatrixData<>(shape, destData, destRowIndices, destColIndices);
    }


    /**
     * Validates that the specified slice is a valid slice of a matrix with the specified {@code shape}.
     * @param shape Shape of the matrix.
     * @param rowStart Starting row index of the slice (inclusive).
     * @param rowEnd Ending row index of the slice (exclusive).
     * @param colStart Starting column index of the slice (inclusive).
     * @param colEnd Ending column index of the slice (exclusive).
     * @throws IllegalArgumentException If <em>any</em> of the following are {@code true}:
     * <ul>
     *     <li>{@code rowStart >= rowEnd}</li>
     *     <li>{@code colStart >= colEnd}</li>
     *     <li>{@code rowStart < 0 || rowEnd > shape.get(0)}</li>
     *     <li>{@code colStart < 0 || colEnd > shape.get(1)}</li>
     * </ul>
     */
    public static void validateSlice(Shape shape, int rowStart, int rowEnd, int colStart, int colEnd) {
        if(rowStart >= rowEnd) {
            throw new IllegalArgumentException("rowStart must be greater than rowEnd but got: rowStart="
                    + rowStart + " and rowEnd=" + rowEnd + ".");
        }
        if(colStart >= colEnd) {
            throw new IllegalArgumentException("colStart must be greater than colEnd but got: colStart="
                    + colStart + " and colEnd=" + colEnd + ".");
        }
        if(rowStart < 0 || rowEnd > shape.get(0)) {
            throw new IllegalArgumentException("Invalid range specified for row indices: [" + rowStart + ", " + rowEnd + ").\n" +
                    "Out of bounds for matrix with shape: " + shape);
        }
        if(colStart < 0 || colEnd > shape.get(1)) {
            throw new IllegalArgumentException("Invalid range specified for column indices: [" + colStart + ", " + colEnd + ").\n" +
                    "Out of bounds for matrix with shape: " + shape);
        }
    }


    /**
     * Validates that the provided arguments specify a valid CSR matrix.
     * @param shape Shape of the CSR matrix.
     * @param nnz The number of non-zero entries of the CSR matrix.
     * @param rowPointers The non-zero row pointers of the CSR matrix.
     * @param colIndices The non-zero column indices of the CSR matrix.
     *
     * @throws IllegalArgumentException If <em>any</em> of the following are {@code true}:
     * <ul>
     *     <li>{@code shape.getRank() != 2}</li>
     *     <li>{@code rowPointers.length != shape.get(0) + 1}</li>
     *     <li>{@code nnz != colIndices.length}</li>
     * </ul>
     */
    public static void validateCsrMatrix(Shape shape, int nnz, int[] rowPointers, int[] colIndices) {
        if (shape.getRank() != 2) {
            throw new IllegalArgumentException("Invalid CSR definition: shape must be of rank 2 but got: " + shape);
        }
        if (rowPointers.length != shape.get(0) + 1) {
            throw new IllegalArgumentException("Invalid CSR definition: the number of row pointers must be " +
                    "equal to the number of rows plus 1 but got row pointer length " +
                    rowPointers.length + " for shape " + shape + ".");
        }

        if (nnz != colIndices.length) {
            throw new IllegalArgumentException("Illegal CSR definition: the number of column indices must be equal" +
                    "to the number of non-zero entries but got " + colIndices.length + " for nnz=" + nnz + ".");
        }
    }
}
