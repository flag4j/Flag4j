/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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

package org.flag4j.arrays.sparse;


import org.flag4j.arrays.ArrayMask;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldTensor;
import org.flag4j.arrays.backend.semiring_arrays.TensorOverSemiring;
import org.flag4j.arrays.dense.FieldTensor;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * Represents a sparse tensor whose non-zero-elements are stored in Coordinate List (COO) format, with all data elements
 * belonging to a specified {@link Field} type.
 *
 * <p>The COO format stores sparse data as a list of coordinates (indices) coupled with their corresponding non-zero values,
 * rather than allocating memory for every element in the full tensor shape. This allows efficient representation and
 * manipulation of high-dimensional tensors that contain a substantial number of zeros.
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li><b>Shape:</b> The full {@link #shape} of the tensor specifying its total dimensionality
 *     and size along each dimension. Although the shape is fixed, it can represent tensors of any rank.</li>
 *
 *     <li><b>Data:</b> Non-zero values are stored in a one-dimensional array, {@link #data}. Any element not specified in
 *     {@code data} is implicitly zero. It is also possible to explicitly store zero values
 *     in this array. To remove any explicitly defined zeros in the tensor use {@link #dropZeros()}.</li>
 *
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many ops assume indices to be sorted in a
 *     row-major format (i.e., last index increased fastest) but often this is not explicitly verified.
 *
 *     <li><b>Indices:</b> The {@link #indices} array, which has dimensions {@code (nnz, rank)}, associates each non-zero
 *     value with its coordinates in the tensor. Here, {@link #nnz} is the count of non-zero-elements and {@link #rank}
 *     is the tensor’s number of dimensions. Each row in {@link #indices} corresponds to the multidimensional index of the
 *     corresponding entry in {@link #data}.</li>
 *     </li>
 * </ul>
 *
 * <p>The total number of non-zero-elements ({@link #nnz}) and the shape are fixed for a given instance, but the specific
 * values in {@link #data} and their corresponding {@link #indices} may be updated. Many operations assume the indices
 * are sorted lexicographically in row-major order (i.e., the last dimension’s index varies fastest), although this is not explicitly
 * enforced. All provided operations will preserve lexicographically row-major sorting of the indices.
 * If there is any doubt that the indices of this tensor may not be sorted, use {@link #sortIndices()} to insure the indices are
 * explicitly sorted. COO tensors may also store multiple entries for the same index (referred to as an uncoalesced tensor). To combine
 * all duplicated entries use {@link #coalesce()} or {@link #coalesce(BinaryOperator)}.
 *
 * <p>COO tensors are optimized for "hyper-sparse" tensors where the proportion of non-zero-elements
 * is extremely low, offering significant memory savings and potentially more efficient computational operations than
 * equivalent dense representations.
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Define shape, data, and indices.
 * Shape shape = new Shape(15, 30, 45, 5)
 * Complex128[] data = {
 *      new Complex128(1, 2), new Complex128(3, 4), new Complex128(5, 6), new Complex128(7, 8)
 * };
 * int[][] indices = {
 *     {0, 1, 2, 3},
 *     {1, 2, 3, 4},
 *     {12, 22, 40, 3},
 *     {12, 22, 41, 0}
 * };
 *
 * // Create COO tensor.
 * CooFieldTensor<Complex128> tensor = new CooFieldTensor(shape, data, indices);
 *
 * // Compute element-wise sum.
 * CooFieldTensor<Complex128> sum = tensor.add(tensor);
 *
 * // Sum of all non-zero entries.
 * Complex128 = tensor.sum();
 *
 * // Reshape tensor.
 * CooFieldTensor<Complex128> reshaped = tensor.reshape(15, 150, 45)
 *
 * // Compute tensor dot product (result is 5&times;5 dense tensor).
 * FieldTensor<Complex128> dot = tensor.dot(tensor,
 *      new int[]{0, 1, 2},
 *      new int[]{0, 1, 2}
 * );
 *
 * // Compute tensor transposes.
 * CooFieldTensor<Complex128> transpose = tensor.T();
 * transpose = tensor.T(0, 1);
 * transpose = tensor.T(1, 3, 0, 2);
 * }</pre>
 *
 * @param <T> The type of elements stored in this tensor, constrained by the {@link Field} interface.
 * @see CooFieldVector
 * @see CooFieldMatrix
 * @see FieldTensor
 * @see Field
 */
public class CooFieldTensor<T extends Field<T>>
        extends AbstractCooFieldTensor<CooFieldTensor<T>, FieldTensor<T>, T> {

    private static final long serialVersionUID = 1L;

    /**
     * creates a tensor with the specified data and shape.
     *
     * @param shape shape of this tensor.
     * @param entries Non-zero data in this tensor. if this tensor is dense, this specifies all data within the
     * tensor.
     * if this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooFieldTensor(Shape shape, T[] entries, int[][] indices) {
        super(shape, entries, indices);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data in this tensor. If this tensor is dense, this specifies all data within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooFieldTensor(Shape shape, List<T> entries, List<int[]> indices) {
        super(shape, (T[]) entries.toArray(), indices.toArray(new int[0][]));
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing COO tensors.
     * @param shape The shape of the tensor to construct.
     * @param data The Non-zero data in this tensor.
     * @param indices The indices of the non-zero data.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooFieldTensor(Shape shape, T[] data, int[][] indices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO tensor in this manner is unsafe.
        super(shape, data, indices, dummy);
    }


    /**
     * <p>Factory to construct a COO tensor which bypasses any validation checks on the data and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param shape The full size of the COO tensor.
     * @param data The non-zero data of the COO tensor.
     * @param indices The non-zero indices of the COO tensor.
     * @return A COO tensor constructed from the provided parameters.
     */
    public static <T extends Field<T>> CooFieldTensor<T> unsafeMake(Shape shape, T[] data, int[][] indices) {
        return new CooFieldTensor(shape, data, indices, null);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, T[] entries, int[][] indices) {
        return new CooFieldTensor<>(shape, entries, indices);
    }


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     *
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     *
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     *
     * @throws ArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    @Override
    public AbstractNDArray<?, ?, T> get(ArrayMask mask) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param index Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooFieldTensor<T> set(T value, int... index) {
        ValidateParameters.validateTensorIndex(shape, index);
        CooFieldTensor<T> dest;

        // Check if value already exists in tensor.
        int idx = -1;
        for(int i=0; i<indices.length; i++) {
            if(Arrays.equals(indices[i], index)) {
                idx = i;
                break; // Found in tensor, no need to continue.
            }
        }

        if(idx > -1) {
            // Copy data and set new value.
            dest = new CooFieldTensor<T>(shape, data.clone(), ArrayUtils.deepCopy2D(indices, null));
            dest.data[idx] = value;
            dest.indices[idx] = index;
        } else {
            // Copy old indices and insert new one.
            int[][] newIndices = new int[indices.length + 1][getRank()];
            ArrayUtils.deepCopy2D(indices, newIndices);
            newIndices[indices.length] = index;

            // Copy old data and insert new one.
            T[] newEntries = Arrays.copyOf(data, data.length+1);
            newEntries[newEntries.length-1] = value;

            dest = new CooFieldTensor<T>(shape, newEntries, newIndices);
            dest.sortIndices();
        }

        return dest;
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and data.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param data Entries of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and data.
     */
    @Override
    public CooFieldTensor<T> makeLikeNDArray(Shape shape, T[] data) {
        return new CooFieldTensor(shape, data, ArrayUtils.deepCopy2D(indices, null));
    }


    /**
     * Computes the sum of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the sum. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     *
     * @see #sum()
     */
    @Override
    public TensorOverSemiring<?, ?, ?, T> sum(int... axes) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Computes the product of all values in this tensor along the specified {@code axes}.
     *
     * @param axes Axes along which to compute the product. All axes must be in the range {@code [0, this.rank() - 1]}.
     *
     * @return A tensor with the same shape as this tensor but with the specified axes removed.
     * The returned tensor will contain the summations along the specified {@code axes}.
     *
     * @see #prod()
     */
    @Override
    public TensorOverSemiring<?, ?, ?, T> prod(int... axes) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Applies a map to each item in this nD array. This operation is done in-place.
     * If this nD array is sparse, the {@code mapper} operation will only be applied to the non-zero
     * elements in this nD array.
     *
     * @param mapper The operation to apply to each item in this nD array.
     *
     * @return A reference to this nD array.
     *
     * @throws NullPointerException If {@code mapper} is {@code null}.
     */
    @Override
    public CooFieldTensor<T> map(UnaryOperator<T> mapper) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the
     * starting value of the
     * reduction.
     * @param accumulator A binary operator that combines the current accumulated
     * result with the next array element and returns the updated result.
     *
     * @return The final accumulated scalar of type {@code V}. If this nD array is empty, {@code identity} will be returned.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator, int...)
     */
    @Override
    public T reduce(T identity, BinaryOperator<T> accumulator) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Reduces all elements of this array to a single scalar by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to <em>every</em> element of this nD array in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction (this may be {@code null}).
     * If {@code null}, then the first entry of this array will be used as the starting value of the
     * reduction.
     * @param accumulator The binary operator used to accumulate elements of this nD array.
     * For the results to be well-defined, the accumulator must be associative and communitive.
     * @param axes The axes along which reduce this nD array.
     *
     * @return An nD array of the same shape as this nD array but with the specified {@code axes} removed.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(V, BinaryOperator)
     */
    @Override
    public AbstractNDArray<?, ?, T> reduce(T identity, BinaryOperator<T> accumulator, int... axes) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Checks if each entry in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each entry in this nD array against.
     *
     * @return An {@link ArrayMask} of the same shape as this nD array containing the boolean results from evaluating each
     * entry in the nD array against the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #filter(Function)
     */
    @Override
    public ArrayMask where(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Extracts elements of this nD array that satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return A flat 1D array containing the elements of this nD array that satisfy the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #where(Function)
     */
    @Override
    public AbstractNDArray<?, ?, T> filter(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Checks if <em>any</em> element in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return {@code true} if <em>any</em> element in this nD array satisfies the {@code predicate}; otherwise {@code false}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #all(Function)
     */
    @Override
    public boolean any(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return false;
    }


    /**
     * Checks if <em>all</em> elements in this nD array satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return {@code true} if <em>all</em> elements in this nD array satisfy the {@code predicate}; otherwise {@code false}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #any(Function)
     */
    @Override
    public boolean all(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return false;
    }


    /**
     * Counts the number of elements in this nD array which satisfy the specified {@code predicate}.
     *
     * @param predicate The predicate to check each element in this nD array against.
     *
     * @return The number of elements in this nD array which satisfy the specified {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     */
    @Override
    public int countTrue(Function<T, Boolean> predicate) {
        // TODO: Implement this method
        return 0;
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero data of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given shape and data.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, List<T> entries, List<int[]> indices) {
        return new CooFieldTensor(shape, entries, indices);
    }


    /**
     * Constructs a dense tensor that is a similar type as this sparse COO tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries The data of the dense tensor to construct.
     *
     * @return A dense tensor that is a similar type as this sparse COO tensor.
     */
    @Override
    public FieldTensor<T> makeLikeDenseTensor(Shape shape, T[] entries) {
        return new FieldTensor<T>(shape, entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public FieldVector<T> toVector() {
        return new FieldVector<T>(data.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must have the same number of {@link ValidateParameters#ensureTotalEntriesEqual(Shape, Shape) total entries}.
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public CooFieldMatrix<T> toMatrix(Shape matShape) {
        ValidateParameters.ensureRank(matShape, 2);

        CooFieldTensor<T> t = reshape(matShape); // Reshape as rank 2 tensor. Total entries check is made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooFieldMatrix<T>(matShape, t.data.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public CooFieldMatrix<T> toMatrix() {
        CooFieldMatrix<T> mat;

        if(getRank()==2) {
            int[][] tIndices = RealDenseTranspose.standardIntMatrix(indices);
            mat = new CooFieldMatrix<T>(shape, data.clone(), tIndices[0], tIndices[1]);
        } else {
            CooFieldTensor<T> flat = reshape(new Shape(1, shape.numelIntValueExact()));
            int[][] tIndices = RealDenseTranspose.standardIntMatrix(flat.indices);
            mat = new CooFieldMatrix<T>(flat.shape, flat.data.clone(), tIndices[0], tIndices[0]);
        }

        return mat;
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooFieldTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooFieldTensor<T> src2 = (CooFieldTensor<T>) object;

        return CooSemiringEquals.cooTensorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for(int i=0; i<nnz; i++) {
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specifying the full shape,
     * non-zero data, and non-zero indices.
     *
     * @return A human-readable string specifying the full shape, non-zero data, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centering = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("nnz: ").append(nnz).append("\n");
        sb.append("Non-zero Entries: " + PrettyPrint.abbreviatedArray(data, maxCols, padding, precision, centering) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centering));

        return sb.toString();
    }
}
