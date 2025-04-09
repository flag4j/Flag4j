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
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingTensor;
import org.flag4j.arrays.dense.RingTensor;
import org.flag4j.arrays.dense.RingVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.Ring;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;

/**
 * Represents a sparse tensor whose non-zero elements are stored in Coordinate List (COO) format, with all data elements
 * belonging to a specified {@link Ring} type.
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
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.
 *
 *     <li><b>Indices:</b> The {@link #indices} array, which has dimensions {@code (nnz, rank)}, associates each non-zero
 *     value with its coordinates in the tensor. Here, {@link #nnz} is the count of non-zero elements and {@link #rank}
 *     is the tensor’s number of dimensions. Each row in {@link #indices} corresponds to the multidimensional index of the
 *     corresponding entry in {@link #data}.</li>
 *     </li>
 * </ul>
 *
 * <p>The total number of non-zero elements ({@link #nnz}) and the shape are fixed for a given instance, but the specific
 * values in {@link #data} and their corresponding {@link #indices} may be updated. Many operations assume the indices
 * are sorted lexicographically in row-major order (i.e., the last dimension’s index varies fastest), although this is not explicitly
 * enforced. All provided operations will preserve lexicographically row-major sorting of the indices.
 * If there is any doubt that the indices of this tensor may not be sorted, use {@link #sortIndices()} to insure the indices are
 * explicitly sorted. COO tensors may also store multiple entries for the same index (referred to as an uncoalesced tensor). To combine
 * all duplicated entries use {@link #coalesce()} or {@link #coalesce(BinaryOperator)}.
 *
 * <p>COO tensors are optimized for "hyper-sparse" tensors where the proportion of non-zero elements
 * is extremely low, offering significant memory savings and potentially more efficient computational operations than
 * equivalent dense representations.
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Define shape, data, and indices.
 * Shape shape = new Shape(15, 30, 45, 5)
 * RealInt32[] data = {
 *      new RealInt32(1), new RealInt32(2), new RealInt32(3), new RealInt32(4)
 * };
 * int[][] indices = {
 *     {0, 1, 2, 3},
 *     {1, 2, 3, 4},
 *     {12, 22, 40, 3},
 *     {12, 22, 41, 0}
 * };
 *
 * // Create COO tensor.
 * CooRingTensor<RealInt32> tensor = new CooRingTensor(shape, data, indices);
 *
 * // Compute element-wise sum.
 * CooRingTensor<RealInt32> sum = tensor.add(tensor);
 *
 * // Sum of all non-zero entries.
 * RealInt32 = tensor.sum();
 *
 * // Reshape tensor.
 * CooRingTensor<RealInt32> reshaped = tensor.reshape(15, 150, 45)
 *
 * // Compute tensor dot product (result is 5&times;5 dense tensor).
 * RingTensor<RealInt32> dot = tensor.dot(tensor,
 *      new int[]{0, 1, 2},
 *      new int[]{0, 1, 2}
 * );
 *
 * // Compute tensor transposes.
 * CooRingTensor<RealInt32> transpose = tensor.T();
 * transpose = tensor.T(0, 1);
 * transpose = tensor.T(1, 3, 0, 2);
 * }</pre>
 *
 * @param <T> The type of elements stored in this tensor, constrained by the {@link Ring} interface.
 * @see CooRingVector
 * @see CooRingMatrix
 * @see RingTensor
 * @see Ring
 */
public class CooRingTensor<T extends Ring<T>> extends AbstractCooRingTensor<CooRingTensor<T>, RingTensor<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero data of this tensor of this tensor. If this tensor is dense, this specifies all data within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooRingTensor(Shape shape, T[] data, int[][] indices) {
        super(shape, data, indices);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero data of this tensor of this tensor. If this tensor is dense, this specifies all data within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooRingTensor(Shape shape, List<T> data, List<int[]> indices) {
        super(shape, (T[]) data.toArray(), indices.toArray(new int[0][]));
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing COO tensors.
     * @param shape The shape of the tensor to construct.
     * @param data The non-zero data of this tensor.
     * @param indices The indices of the non-zero data.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooRingTensor(Shape shape, T[] data, int[][] indices, Object dummy) {
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
    public static <T extends Ring<T>> CooRingTensor<T> unsafeMake(Shape shape, T[] data, int[][] indices) {
        return new CooRingTensor(shape, data, indices, null);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    @Override
    public CooRingTensor<T> makeLikeTensor(Shape shape, T[] data, int[][] indices) {
        return new CooRingTensor<>(shape, data, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    @Override
    public CooRingTensor<T> makeLikeTensor(Shape shape, List<T> data, List<int[]> indices) {
        return new CooRingTensor<>(shape, data, indices);
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
    public RingTensor<T> makeLikeDenseTensor(Shape shape, T[] entries) {
        return new RingTensor<>(shape, entries);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CooRingTensor<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooRingTensor<>(shape, entries, ArrayUtils.deepCopy2D(indices, null));
    }

    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public RingVector<T> toVector() {
        return new RingVector<T>(data.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must have the same number of {@link ValidateParameters#ensureTotalEntriesEqual(Shape, Shape) total entries}.
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public CooRingMatrix<T> toMatrix(Shape matShape) {
        ValidateParameters.ensureRank(matShape, 2);

        CooRingTensor<T> t = reshape(matShape); // Reshape as rank 2 tensor. total entries check is made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooRingMatrix<T>(matShape, t.data.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public CooRingMatrix<T> toMatrix() {
        CooRingMatrix<T> mat;

        if(getRank()==2) {
            int[][] tIndices = RealDenseTranspose.standardIntMatrix(indices);
            mat = new CooRingMatrix<T>(shape, data.clone(), tIndices[0], tIndices[1]);
        } else {
            CooRingTensor<T> flat = reshape(new Shape(1, shape.totalEntriesIntValueExact()));
            int[][] tIndices = RealDenseTranspose.standardIntMatrix(flat.indices);
            mat = new CooRingMatrix<T>(flat.shape, flat.data.clone(), tIndices[0], tIndices[0]);
        }

        return mat;
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CooTensor abs() {
        double[] dest = new double[data.length];
        RingOps.abs(data, dest);
        return CooTensor.unsafeMake(shape, dest, ArrayUtils.deepCopy2D(indices, null));
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooRingTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooRingTensor<T> src2 = (CooRingTensor<T>) object;

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
