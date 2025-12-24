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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldVector;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.List;
import java.util.function.BinaryOperator;


/**
 * Represents a sparse vector whose non-zero-elements are stored in Coordinate List (COO) format, with all data elements
 * belonging to a specified {@link Field} type.
 *
 * <p>The COO format stores sparse vector data as a list of coordinates (indices) coupled with their
 * corresponding non-zero values, rather than allocating memory for every element in the full vector shape. This
 * allows efficient representation and manipulation of large vector containing a substantial number of zeros.
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li><b>Shape:</b> The full {@link #shape}/{@link #size} of the vector specifying the total number of values (including zeros)
 *     in the vector.</li>
 *
 *     <li><b>Data:</b> Non-zero values are stored in a one-dimensional array, {@link #data}. Any element not specified in
 *     {@code data} is implicitly zero. It is also possible to explicitly store zero values in this array, although this
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
 * // shape, data, and indices for COO vector.
 * Shape shape = new Shape(512);
 * Complex128[] data = {
 *      new Complex128(1, 2), new Complex128(3, 4), new Complex128(5, 6)
 *      new Complex128(7, 8), new Complex128(9, 10), new Complex128(11, 12)
 * };
 * int[] indices = {0, 4, 128, 128, 128, 256};
 *
 * // Create COO vector.
 * CooFieldVector<Complex128> vector = new CooFieldVector(shape, data, indices);
 *
 * // Sum vectors.
 * CooFieldVector<Complex128> sum = vector.add(vector);
 *
 * // Compute vector inner product.
 * Complex128 prod = vector.inner(vector);
 *
 * // Compute vector outer product.
 * FieldMatrix<Complex128> prod = vector.outer(vector);
 * }</pre>
 *
 * @param <T> The type of elements stored in this vector, constrained by the {@link Field} interface.
 * @see CooFieldTensor
 * @see CooFieldMatrix
 * @see FieldMatrix
 * @see Field
 */
public class CooFieldVector<T extends Field<T>> extends AbstractCooFieldVector<CooFieldVector<T>,
        FieldVector<T>, CooFieldMatrix<T>, FieldMatrix<T>, T> {

    private static final long serialVersionUID = 1L;

    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooFieldVector(int size, T[] entries, int[] indices) {
        super(new Shape(size), entries, indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param Shape The full shape of the vector.
     * @param entries The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooFieldVector(Shape shape, T[] entries, int[] indices) {
        super(shape, entries, indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooFieldVector(int size, List<T> entries, List<Integer> indices) {
        super(new Shape(size), (T[]) entries.toArray(Field[]::new), ArrayConversions.fromIntegerList(indices));
    }


    /**
     * Creates a zero vector of the specified {@code size}.
     */
    public CooFieldVector(int size) {
        super(new Shape(size), (T[]) new Field[0], new int[0]);
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing COO vectors.
     * @param shape Shape of the COO vector to construct.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooFieldVector(Shape shape, T[] data, int[] indices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO vector in this manner is unsafe.
        super(shape, data, indices, dummy);
    }


    /**
     * <p>Factory to construct a COO vector which bypasses any validation checks on the data and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param size The full size of the COO vector.
     * @param data The non-zero entries of the COO vector.
     * @param indices The non-zero indices of the COO vector.
     * @return A COO vector constructed from the provided parameters.
     */
    public static <T extends Field<T>> CooFieldVector<T> unsafeMake(int size, T[] data, int[] indices) {
        return new CooFieldVector(new Shape(size), data, indices, null);
    }


    /**
     * <p>Factory to construct a COO vector which bypasses any validation checks on the data and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param Shape Full shape of the COO vector. Assumed to be rank 1 (this is <em>not</em> enforced).
     * @param data The non-zero entries of the COO vector.
     * @param indices The non-zero indices of the COO vector.
     * @return A COO vector constructed from the provided parameters.
     */
    public static <T extends Field<T>> CooFieldVector<T> unsafeMake(Shape shape, T[] data, int[] indices) {
        return new CooFieldVector(shape, data, indices, null);
    }


    /**
     * Constructs a sparse COO vector of the same type as this vector with the specified non-zero data and indices.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero data of the vector to construct.
     * @param indices Non-zero row indices of the vector to construct.
     *
     * @return A sparse COO vector of the same type as this vector with the specified non-zero data and indices.
     */
    @Override
    public CooFieldVector<T> makeLikeTensor(Shape shape, T[] entries, int[] indices) {
        return new CooFieldVector<>(shape, entries, indices);
    }


    /**
     * Constructs a dense vector of a similar type as this vector with the specified shape and data.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the vector to construct.
     *
     * @return A dense vector of a similar type as this vector with the specified data.
     */
    @Override
    public FieldVector<T> makeLikeDenseTensor(Shape shape, T... entries) {
        ValidateParameters.ensureRank(shape, 1);
        return new FieldVector<>(entries);
    }


    /**
     * Constructs a dense matrix of a similar type as this vector with the specified shape and data.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A dense matrix of a similar type as this vector with the specified data.
     */
    @Override
    public FieldMatrix<T> makeLikeDenseMatrix(Shape shape, T... entries) {
        return new FieldMatrix<>(shape, entries);
    }


    /**
     * Constructs a COO vector with the specified shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the vector.
     * @param entries Non-zero values of the vector.
     * @param indices Indices of the non-zero values in the vector.
     *
     * @return A COO vector of the same type as this vector with the specified shape, non-zero data, and non-zero indices.
     */
    @Override
    public CooFieldVector<T> makeLikeTensor(Shape shape, List<T> entries, List<Integer> indices) {
        return new CooFieldVector<>(size, entries, indices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and row and column indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     *
     * @return A COO matrix of similar type as this vector with the specified shape, non-zero data, and non-zero row/col indices.
     */
    @Override
    public CooFieldMatrix<T> makeLikeMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CooFieldVector<T> makeLikeNDArray(Shape shape, T[] data) {
        return new CooFieldVector<T>(shape, data, indices.clone());
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

        CooFieldVector<T> src2 = (CooFieldVector<T>) object;

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
    public CooFieldMatrix<T> toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[data.length];

            return new CooFieldMatrix<T>(this.size, 1, data.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[data.length];
            int[] colIndices = indices.clone();

            return new CooFieldMatrix<T>(1, this.size, data.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooFieldTensor<T> toTensor() {
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(new int[][]{indices});
        return new CooFieldTensor(shape, data.clone(), tIndices);
    }


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooFieldTensor<T> toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flatten data of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("nnz: ").append(nnz).append("\n");
        result.append("Non-zero data: [");

        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();
        int precision = PrintOptions.getPrecision();

        if(size > 0) {
            int stopIndex = Math.min(maxCols -1, size-1);
            int width;
            String value;

            // Get data up until the stopping point.
            for(int i = 0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(data[i], precision);
                width = padding + value.length();
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = padding + 3;
                value = "...";
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(data[size-1], precision);
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
