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
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingVector;
import org.flag4j.arrays.backend.semiring_arrays.TensorOverSemiring;
import org.flag4j.arrays.dense.RingMatrix;
import org.flag4j.arrays.dense.RingVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;

import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * Represents a sparse vector whose non-zero-elements are stored in Coordinate List (COO) format, with all items elements
 * belonging to a specified {@link RingElement} type.
 *
 * <p>The COO format stores sparse vector items as a list of coordinates (indices) coupled with their
 * corresponding non-zero values, rather than allocating memory for every element in the full vector shape. This
 * allows efficient representation and manipulation of large vector containing a substantial number of zeros.
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li><b>Shape:</b> The full {@link #shape}/{@link #size} of the vector specifying the total number of values (including zeros)
 *     in the vector.</li>
 *
 *     <li><b>Data:</b> Non-zero values are stored in a one-dimensional array, {@link #data}. Any element not specified in
 *     {@code items} is implicitly zero. It is also possible to explicitly store zero values in this array, although this
 *     is generally not desirable. To remove explicitly defined zeros, use {@link #dropZeros()}.</li>
 *
 *     <li><b>Indices:</b> Non-zero values are associated with their coordinates in the vector via a single 1D array:
 *     {@link #indices}. This array specifies the positions of each
 *     non-zero entry in {@link #data}. The total number of non-zero-elements is given by {@link #nnz}.</li>
 * </ul>
 *
 * <p>The total number of non-zero-elements ({@link #nnz}) and the shape/size is fixed for a given instance, but the values
 * in {@link #data} and their corresponding {@link #indices} may be updated. Many operations
 * assume that the indices are sorted lexicographically, but this is not strictly enforced.
 * All provided operations preserve the lexicographical sorting of indices. If there is any doubt about the ordering of
 * indices, use {@link #sortIndices()} to ensure they are sorted. COO tensors may also store multiple entries
 * for the same index (referred to as an uncoalesced tensor). To combine all duplicated entries use {@link #coalesce()} or
 * {@link #coalesce(BinaryOperator)}.
 *
 * <p>COO vectors are optimized for "hyper-sparse" scenarios where the proportion of non-zero-elements is extremely low,
 * offering significant memory savings and potentially more efficient computational operations than equivalent dense
 * representations.
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // shape, items, and indices for COO vector.
 * Shape shape = new Shape(512);
 * RealI32[] items = {
 *      new RealI32(1), new RealI32(2), new RealI32(3),
 *      new RealI32(4), new RealI32(5), new RealI32(6)
 * };
 * int[] indices = {0, 4, 128, 128, 128, 256};
 *
 * // Create COO vector.
 * CooRingVector<RealI32> vector = new CooRingVector(shape, items, indices);
 *
 * // Sum vectors.
 * CooRingVector<RealI32> sum = vector.add(vector);
 *
 * // Compute vector inner product.
 * RealI32 prod = vector.inner(vector);
 *
 * // Compute vector outer product.
 * RingMatrix<RealI32> prod = vector.outer(vector);
 * }</pre>
 *
 * @param <T> The type of elements stored in this vector, constrained by the {@link RingElement} interface.
 * @see CooRingTensor
 * @see CooRingMatrix
 * @see RingMatrix
 * @see RingElement
 */
public class CooRingVector<T extends RingElement<T>> extends AbstractCooRingVector<
        CooRingVector<T>, RingVector<T>, CooRingMatrix<T>, RingMatrix<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param size
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     * @param indices
     */
    public CooRingVector(int size, T[] entries, int[] indices) {
        super(new Shape(size), entries, indices);
    }


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param size
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     * @param indices
     */
    public CooRingVector(Shape shape, T[] entries, int[] indices) {
        super(shape, entries, indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero items, and non-zero indices.
     *
     * @param Shape The shape of this vector. Must be rank-1.
     * @param entries The non-zero items of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooRingVector(Shape shape, List<T> entries, List<Integer> indices) {
        super(shape, (T[]) entries.toArray(RingElement[]::new), ArrayConversions.fromIntegerList(indices));
    }


    /**
     * Creates a zero vector of the specified {@code size}.
     */
    public CooRingVector(int size) {
        super(new Shape(size), (T[]) new RingElement[0], new int[0]);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero items, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero items of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooRingVector(int size, List<T> entries, List<Integer> indices) {
        super(new Shape(size), (T[]) entries.toArray(RingElement[]::new), ArrayConversions.fromIntegerList(indices));
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing COO vectors.
     * @param shape Shape of the COO vector to construct.
     * @param data The non-zero items of this vector.
     * @param indices The indices of the non-zero values.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooRingVector(Shape shape, T[] data, int[] indices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO vector in this manner is unsafe.
        super(shape, data, indices, dummy);
    }


    /**
     * <p>Factory to construct a COO vector which bypasses any validation checks on the items and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param size The full size of the COO vector.
     * @param data The non-zero entries of the COO vector.
     * @param indices The non-zero indices of the COO vector.
     * @return A COO vector constructed from the provided parameters.
     */
    public static <T extends RingElement<T>> CooRingVector<T> unsafeMake(int size, T[] data, int[] indices) {
        return new CooRingVector(new Shape(size), data, indices, null);
    }


    /**
     * <p>Factory to construct a COO vector which bypasses any validation checks on the items and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param Shape Full shape of the COO vector. Assumed to be rank 1 (this is <em>not</em> enforced).
     * @param data The non-zero entries of the COO vector.
     * @param indices The non-zero indices of the COO vector.
     * @return A COO vector constructed from the provided parameters.
     */
    public static <T extends RingElement<T>> CooRingVector<T> unsafeMake(Shape shape, T[] data, int[] indices) {
        return new CooRingVector(shape, data, indices, null);
    }


    /**
     * Constructs a sparse COO vector of the same type as this vector with the specified non-zero items and indices.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero items of the vector to construct.
     * @param indices Non-zero row indices of the vector to construct.
     *
     * @return A sparse COO vector of the same type as this vector with the specified non-zero items and indices.
     */
    @Override
    public CooRingVector<T> makeLikeTensor(Shape shape, T[] entries, int[] indices) {
        return new CooRingVector<>(shape, entries, indices);
    }


    /**
     * Constructs a dense vector of a similar type as this vector with the specified shape and items.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the vector to construct.
     *
     * @return A dense vector of a similar type as this vector with the specified items.
     */
    @Override
    public RingVector<T> makeLikeDenseTensor(Shape shape, T... entries) {
        return new RingVector<>(shape, entries);
    }


    /**
     * Constructs a dense matrix of a similar type as this vector with the specified shape and items.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A dense matrix of a similar type as this vector with the specified items.
     */
    @Override
    public RingMatrix<T> makeLikeDenseMatrix(Shape shape, T... entries) {
        return new RingMatrix<>(shape, entries);
    }


    /**
     * Constructs a COO vector with the specified shape, non-zero items, and non-zero indices.
     *
     * @param shape Shape of the vector.
     * @param entries Non-zero values of the vector.
     * @param indices Indices of the non-zero values in the vector.
     *
     * @return A COO vector of the same type as this vector with the specified shape, non-zero items, and non-zero indices.
     */
    @Override
    public CooRingVector<T> makeLikeTensor(Shape shape, List<T> entries, List<Integer> indices) {
        return new CooRingVector<>(shape, entries, indices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero items, and row and column indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero items of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     *
     * @return A COO matrix of similar type as this vector with the specified shape, non-zero items, and non-zero row/col indices.
     */
    @Override
    public CooRingMatrix<T> makeLikeMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooRingMatrix<>(shape, entries, rowIndices, colIndices);
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
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code items}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code items}.
     */
    @Override
    public CooRingVector<T> makeLikeNDArray(Shape shape, T[] data) {
        return new CooRingVector<>(shape, data, indices.clone());
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
    public CooRingVector<T> map(UnaryOperator<T> mapper) {
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
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooFieldVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooRingVector<T> src2 = (CooRingVector<T>) object;

        return CooSemiringEquals.cooVectorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < data.length; i++) {
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Integer.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public CooRingMatrix<T> toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[data.length];

            return new CooRingMatrix<T>(size, 1, data.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[data.length];
            int[] colIndices = indices.clone();

            return new CooRingMatrix<T>(1, size, data.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooRingTensor<T> toTensor() {
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(new int[][]{indices});
        return new CooRingTensor(shape, data.clone(), tIndices);
    }


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooRingTensor<T> toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }



    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CooVector abs() {
        double[] dest = new double[data.length];
        RingOps.abs(data, dest);
        return CooVector.unsafeMake(shape, dest, indices.clone());
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flatten items of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("nnz: ").append(nnz).append("\n");
        result.append("Non-zero items: [");

        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();
        int precision = PrintOptions.getPrecision();

        if(size > 0) {
            int stopIndex = Math.min(maxCols - 1, size - 1);
            int width;
            String value;

            // Get items up until the stopping point.
            for(int i = 0; i < stopIndex; i++) {
                value = StringUtils.ValueOfRound(data[i], precision);
                width = padding + value.length();
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size - 1) {
                width = padding + 3;
                value = "...";
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(data[size - 1], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");
        result.append("Indices: ")
                .append(PrettyPrint.abbreviatedArray(indices, maxCols, padding, centering));

        return result.toString();
    }
}
