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
import org.flag4j.arrays.IntTuple;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseTensorData;
import org.flag4j.arrays.backend.primitive_arrays.AbstractDoubleNDArray;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.coo.CooDataSorter;
import org.flag4j.linalg.ops.sparse.coo.real.RealCooTensorDot;
import org.flag4j.linalg.ops.sparse.coo.real.RealCooTensorOps;
import org.flag4j.linalg.ops.sparse.coo.real.RealSparseEquals;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexCooTensorOps;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ShapeUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.*;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Function;


/**
 * <p>A real sparse tensor stored in coordinate list (COO) format. The {@link #data} of this COO tensor are
 * primitive doubles.
 *
 * <p>The {@link #data non-zero items} and {@link #indices non-zero indices} of a COO tensor are mutable but the {@link #shape}
 * and total number of non-zero items is fixed.
 *
 * <p>Sparse tensors allow for the efficient storage of and ops on tensors that contain many zero values.
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e., tensors which contain almost all zeros relative to the size of the
 * tensor).
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #data} of the tensor. All other items in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #data}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many ops assume indices to be sorted in a
 *     row-major format (i.e., the last index increased fastest), but often this is not explicitly verified.
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero items in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the nD
 *     index of {@code items[i]}.
 *     </li>
 * </ul>
 *
 * <p>Some ops on sparse tensors behave differently than on dense tensors. For instance, {@link #add(double)} will not
 * add the scalar to all items of the tensor since this would cause catastrophic loss of sparsity. Instead, such non-zero preserving
 * element-wise ops only act on the non-zero items of the sparse tensor as to not affect the sparsity.
 *
 * <p>Note: many ops assume that the items of the COO tensor are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.
 */
public class CooTensor extends AbstractDoubleNDArray<CooTensor> {

    private static final long serialVersionUID = 1L;

    /**
     * The non-zero indices of this tensor. Must have shape {@code (nnz, rank)}.
     */
    public final int[][] indices;
    /**
     * The number of non-zero items in this tensor.
     */
    public final int nnz;
    /**
     * The sparsity of this matrix.
     */
    private double sparsity = -1;

    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero items in this COO tensor.
     * @param indices The non-zero indices of this COO tensor. Must have dimensions {@code (nnz, rank)}.
     *
     * @throws IllegalArgumentException If {@code shape}, {@code items}, and {@code indices} do <em>not</em>
     * specify a valid COO tensor.
     */
    public CooTensor(Shape shape, double[] data, int[][] indices) {
        super(shape, data);
        this.indices = indices;
        this.nnz = data.length;
        SparseValidation.validateCoo(shape, this.nnz, this.indices);
    }


    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero items in this tensor.
     * @param indices The non-zero indices of this COO tensor. Must have dimensions {@code (nnz, rank)}.
     *
     * @throws IllegalArgumentException If {@code shape}, {@code items}, and {@code indices} do <em>not</em>
     * specify a valid COO tensor.
     */
    public CooTensor(Shape shape, List<Double> data, List<int[]> indices) {
        super(shape, ArrayConversions.fromDoubleList(data));
        this.indices = indices.toArray(new int[0][]);
        this.nnz = super.data.length;
        SparseValidation.validateCoo(shape, this.nnz, this.indices);
    }


    /**
     * Creates a zero-matrix with the specified shape.
     * @param shape The shape of the zero-matrix to construct.
     */
    public CooTensor(Shape shape) {
        super(shape, new double[0]);
        this.indices = new int[0][getRank()];
        this.nnz = super.data.length;
    }


    /**
     * Creates a sparse COO matrix with the specified shape, non-zero items, and indices.
     * @param shape Shape of the matrix to construct.
     * @param data Non-zero items of the sparse COO matrix.
     * @param indices Indices of the non-zero items in the sparse COO matrix.
     */
    public CooTensor(Shape shape, int[] data, int[][] indices) {
        super(shape, ArrayConversions.asDouble(data, null));
        this.indices = indices;
        this.nnz = data.length;
        SparseValidation.validateCoo(shape, this.nnz, this.indices);
    }


    /**
     * Constructs a copy of the specified matrix.
     * @param b Matrix to make a copy of.
     */
    public CooTensor(CooTensor b) {
        super(b.shape, b.data.clone());
        this.indices = ArrayUtils.deepCopy2D(b.indices, null);
        this.nnz = b.nnz;
        SparseValidation.validateCoo(shape, this.nnz, this.indices);
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing COO tensors.
     * @param shape The shape of the tensor to construct.
     * @param data The Non-zero items in this tensor.
     * @param indices The indices of the non-zero items.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooTensor(Shape shape, double[] data, int[][] indices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO tensor in this manner is unsafe.
        super(shape, data);
        this.indices = indices;
        this.nnz = data.length;
    }


    /**
     * Creates a real dense nD COO tensor with the specified {@code shape}, non-zero items, and indices.
     *
     * @param shape The shape of the COO tensor to construct.
     * @param indexDataMap A map where the keys represent the non-zero indices and the values represent the non-zero values of
     * the COO tensor to construct.
     */
    public CooTensor(Shape shape, Map<IntTuple, Double> indexDataMap) {
        super(shape, new double[indexDataMap.size()]);
        this.nnz = indexDataMap.size();
        this.indices = new int[nnz][shape.getRank()];

        int count = 0;
        for(Map.Entry<IntTuple, Double> entry : indexDataMap.entrySet()) {
            this.indices[count] = entry.getKey().data();
            this.data[count] = entry.getValue();
            count++;
        }
    }


    /**
     * <p>Factory to construct a COO tensor which bypasses any validation checks on the items and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param shape The full size of the COO tensor.
     * @param data The non-zero items of the COO tensor.
     * @param indices The non-zero indices of the COO tensor.
     * @return A COO tensor constructed from the provided parameters.
     */
    public static CooTensor unsafeMake(Shape shape, double[] data, int[][] indices) {
        return new CooTensor(shape, data, indices, null);
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and items.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param data Entries of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and items.
     */
    @Override
    public CooTensor makeLikeNDArray(Shape shape, double[] data) {
        return new CooTensor(shape, data, ArrayUtils.deepCopy2D(indices, null));
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given shape, non-zero items, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param data Non-zero items of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given shape and items.
     */
    public CooTensor makeLikeTensor(Shape shape, double[] data, int[][] indices) {
        return new CooTensor(shape, data, indices);
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given shape, non-zero items, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param data Non-zero items of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given shape and items.
     */
    public CooTensor makeLikeTensor(Shape shape, List<Double> data, List<int[]> indices) {
        return new CooTensor(shape, data, indices);
    }


    /**
     * Makes a dense tensor with the specified shape and items, which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param data Entries of the dense tensor.
     *
     * @return A dense tensor with the specified shape and items which is a similar type to this sparse tensor.
     */
    public Tensor makeDenseTensor(Shape shape, double[] data) {
        return new Tensor(shape, data);
    }


    /**
     * The sparsity of this sparse tensor. That is, the decimal percentage of elements in this tensor that are zero.
     * @return The density of this sparse tensor.
     * @see #getDensity()
     */
    public double getSparsity() {
        // Check if the sparsity has already been computed.
        if (this.sparsity < 0)
            this.sparsity = SparseUtils.computeSparsity(shape, nnz);

        return sparsity;
    }


    /**
     * Gets the density of this tensor as a decimal percentage.
     * That is, the percentage of elements in this tensor that are non-zero.
     * @return The density of this tensor as a decimal percentage.
     * @see #getSparsity()
     */
    public double getDensity() {
        return 1.0 - getSparsity();
    }


    /**
     * Converts this tensor to an equivalent complex tensor.
     * @return A complex COO tensor whose non-zero items have a real component equal to the Non-zero items in this tensor and
     * imaginary parts zero.
     */
    public CooCTensor toComplex() {
        return CooCTensor.unsafeMake(shape,
                ArrayConversions.toComplex128(data, null),
                ArrayUtils.deepCopy2D(indices, null));
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    public Tensor toDense() {
        double[] denseData = new double[totalEntries().intValueExact()];

        for(int i=0; i<nnz; i++)
            denseData[shape.get1DIndex(indices[i])] = this.data[i];

        return new Tensor(shape, denseData);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public Double get(int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        if(data.length == 0) return null;  // Cannot get reference of field, so there is no way to get the zero-element.

        for(int i=0; i<nnz; i++)
            if(Arrays.equals(this.indices[i], indices)) return data[i];

        return 0.0; // Return zero if the index is not found.
    }


    /**
     * Gets elements of this nD array according to a boolean {@code mask} (i.e., "masked select").
     *
     * @param mask The boolean mask specifying which elements to get from this nD array. Must be the same shape as this nD array.
     *
     * @return A 1D array containing the elements indexed by the {@code true} values in {@code mask}.
     * That is, the values in this nD array at all indices where {@code mask} is {@code true}.
     *
     * @throws NDArrayShapeException If {@code mask} has a different shape as this nD array.
     */
    @Override
    public Vector get(ArrayMask mask) {
        double[] maskedData = new double[mask.cardinality()];
        BitSet maskBits = mask.data;

        for(int i = maskBits.nextSetBit(0), j = 0; i >= 0; i=maskBits.nextSetBit(i + 1)) {
            int[] nDIndex = shape.getNdIndices(i);
            maskedData[j++] = get(nDIndex);
        }

        return new Vector(maskedData);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param Index Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooTensor set(Double value, int... index) {
        ValidateParameters.validateTensorIndex(shape, index);
        CooTensor dest;

        // Check if the value already exists in the tensor.
        int idx = -1;
        for(int i=0; i<indices.length; i++) {
            if(Arrays.equals(indices[i], index)) {
                idx = i;
                break; // Found in tensor, no need to continue.
            }
        }

        if(idx > -1) {
            // Copy items and set the new value.
            dest = unsafeMake(shape, data.clone(), ArrayUtils.deepCopy2D(indices, null));
            dest.data[idx] = value;
            dest.indices[idx] = index;
        } else {
            // Copy old indices and insert new one.
            int[][] newIndices = new int[indices.length + 1][getRank()];
            ArrayUtils.deepCopy2D(indices, newIndices);
            newIndices[indices.length] = index;

            // Copy old items and insert new one.
            double[] newEntries = Arrays.copyOf(data, data.length+1);
            newEntries[newEntries.length-1] = value;

            dest = unsafeMake(shape, newEntries, newIndices);
            dest.sortIndices();
        }

        return dest;
    }


    /**
     * Flattens tensor to a single dimension while preserving the order of items.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CooTensor flatten() {
        int[][] destIndices = new int[data.length][1];

        for(int i = 0, size = data.length; i<size; i++)
            destIndices[i][0] = shape.get1DIndex(indices[i]);

        return makeLikeTensor(new Shape(shape.numel().intValueExact()), data.clone(), destIndices);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public CooTensor flatten(int axis) {
        ValidateParameters.validateArrayIndices(indices[0].length, axis);
        int[][] destIndices = new int[indices.length][indices[0].length];

        // Compute new shape.
        int[] destShape = new int[indices[0].length];
        Arrays.fill(destShape, 1);
        destShape[axis] = shape.numel().intValueExact();

        for(int i = 0, size = data.length; i<size; i++)
            destIndices[i][axis] = shape.get1DIndex(indices[i]);

        return makeLikeTensor(new Shape(destShape), data.clone(), destIndices);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws NDArrayShapeException If {@code newShape} does not have the same number of total entries as {@link #shape this.shape}.
     */
    @Override
    public CooTensor reshape(Shape newShape) {
        ValidateParameters.ensureTotalEntriesEqual(shape, newShape);

        int rank = indices[0].length;
        int newRank = newShape.getRank();
        int nnz = data.length;

        int[] oldStrides = shape.getStrides();
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

        return makeLikeTensor(newShape, data.clone(), newIndices);
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public CooTensor sub(Double b) {
        return super.sub(b);  // Overrides superclass to emphasize this method only acts on non-zero items of the tensor.
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Double b) {
        super.subEq(b); // Overrides superclass to emphasize this method only acts on non-zero items of the tensor.
    }


    /**
     * Adds a scalar field value to each non-zero entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public CooTensor add(Double b) {
        return super.add(b); // Overrides superclass to emphasize this method only acts on non-zero items of the tensor.
    }


    /**
     * Adds a scalar value to each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        super.addEq(b); // Overrides superclass to emphasize this method only acts on non-zero items of the tensor.
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooTensor add(CooTensor b) {
        return RealCooTensorOps.add(this, b);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCTensor add(CooCTensor b) {
        return RealComplexCooTensorOps.add(b, this);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooTensor sub(CooTensor b) {
        return RealCooTensorOps.sub(this, b);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        return indices[RealProperties.argmin(data)];
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return indices[RealProperties.argmax(data)];
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return indices[RealProperties.argminAbs(data)];
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return indices[RealProperties.argmaxAbs(data)];
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooTensor elemMult(CooTensor b) {
        return RealCooTensorOps.elemMult(this, b);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 Tensor to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensor's shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public Tensor tensorDot(CooTensor src2, int[] aAxes, int[] bAxes) {
        return RealCooTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values in the 2D subarrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param axis1 First axis for 2D subarray.
     * @param axis2 Second axis for 2D subarray.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}. This will be a tensor of rank
     * {@code this.getRank() - 2} with the same shape as this tensor but with {@code axis1} and {@code axis2} removed.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensor's rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e., the axes are equal, or the tensor does not have the same length along the two axes.)
     */
    @Override
    public CooTensor tensorTr(int axis1, int axis2) {
        // Validate parameters.
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.validateArrayIndices(getRank(), axis1, axis2);
        ValidateParameters.ensureAllEqual(shape.getSize(axis1), shape.getSize(axis2));

        int rank = getRank();
        int[] dims = shape.getDims();

        // Determine the shape of the resulting tensor.
        int[] traceShape = new int[rank - 2];
        int newShapeIndex = 0;
        for (int i = 0; i < rank; i++) {
            if (i != axis1 && i != axis2) {
                traceShape[newShapeIndex++] = dims[i];
            }
        }

        // Use a map to accumulate non-zero items that are on the diagonal.
        Map<Integer, Double> resultMap = new HashMap<>();
        int[] strides = shape.getStrides();

        // Iterate through the non-zero items and accumulate trace for those on the diagonal.
        for (int i = 0; i < this.nnz; i++) {
            int[] indices = this.indices[i];
            double value = this.data[i];

            // Check if the current entry is on the diagonal
            if (indices[axis1] == indices[axis2]) {
                // Compute a linear index for the resulting tensor by ignoring axis1 and axis2.
                int linearIndex = 0;
                int stride = 1;

                for (int j = rank - 1; j >= 0; j--) {
                    if (j != axis1 && j != axis2) {
                        linearIndex += indices[j] * stride;
                        stride *= dims[j];
                    }
                }

                // Accumulate the value in the result map.
                resultMap.put(linearIndex, resultMap.getOrDefault(linearIndex, 0.0) + value);
            }
        }

        // Construct the result tensor from the accumulated non-zero items
        int resultNnz = resultMap.size();
        int[][] resultIndices = new int[resultNnz][rank - 2];
        double[] resultEntries = new double[resultNnz];
        int resultIndex = 0;

        for (Map.Entry<Integer, Double> entry : resultMap.entrySet()) {
            int linearIndex = entry.getKey();
            double entryValue = entry.getValue();

            // Use the getIndices method to convert the flat index to n-dimensional index.
            int[] multiDimIndices = shape.getNdIndices(linearIndex);

            // Copy relevant dimensions to resultIndices, excluding axis1 and axis2.
            int resultDimIndex = 0;
            for (int j = 0; j < rank; j++) {
                if (j != axis1 && j != axis2) {
                    resultIndices[resultIndex][resultDimIndex++] = multiDimIndices[j];
                }
            }

            resultEntries[resultIndex] = entryValue;
            resultIndex++;
        }

        return makeLikeTensor(new Shape(traceShape), resultEntries, resultIndices);
    }


    /**
     * <p>Computes the product of all non-zero values in this tensor.
     *
     * <p>NOTE: This is <b>only</b> the product of the non-zero values in this tensor.
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public Double prod() {
        // Override from FieldTensorBase to emphasize that the product is only for non-zero items.
        return super.prod();
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public CooTensor T(int axis1, int axis2) {
        int rank = getRank();
        ValidateParameters.validateArrayIndices(rank, axis1, axis2);

        if(axis1 == axis2) return copy();   // Simply Return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        double[] transposeEntries = new double[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = data[i];
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create a sparse coo tensor and sort values lexicographically by indices.
        CooTensor transpose = makeLikeTensor(shape.swapAxes(axis1, axis2), transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public CooTensor T(int... axes) {
        int rank = getRank();
        ValidateParameters.ensureAllEqual(rank, axes.length);
        ValidateParameters.ensurePermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        double[] transposeEntries = new double[nnz];

        // Permute the indices according to the permutation array.
        for(int i=0; i < nnz; i++) {
            transposeEntries[i] = data[i];
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create a sparse COO tensor and sort values lexicographically by indices.
        CooTensor transpose = makeLikeTensor(shape.permuteAxes(axes), transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * <p>Computes the element-wise reciprocals of the non-zero-elements in this sparse tensor.
     *
     * <p>Note: This method <b>only</b> computes the reciprocals of the non-zero-elements.
     *
     * @return A tensor containing the reciprocal non-zero-elements of this tensor.
     */
    @Override
    public CooTensor recip() {
        /* This method is override from FieldTensorBase to make clear it is only computing the
            multiplicative inverse for the non-zero-elements of the tensor */
        double[] recip = new double[data.length];

        for(int i = 0, size = data.length; i<size; i++)
            recip[i] = 1.0/data[i];

        return makeLikeNDArray(shape, recip);
    }


    /**
     * <p>Reduces elements of this array, along a specified set of axes, by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to elements of this nD array along the specified axes in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * @param identity The starting value for the reduction.
     * <strong>Note:</strong>Unlike with dense nD array objects, the identity may <i>not</i> be {@code null}.
     * @param accumulator The binary operator used to accumulate elements of this nD array.
     * For the results to be well-defined, the accumulator must be associative and communitive.
     * @param axes The axes along which reduce this nD array.
     *
     * @return An nD array of the same shape as this nD array, but with the specified {@code axes} removed, containing the result of
     * the reduction operation.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(Double, BinaryOperator)
     */
    @Override
    public AbstractDoubleNDArray<?> reduce(Double identity, DoubleBinaryOperator accumulator, int... axes) {
        Objects.requireNonNull(identity, "The identity object must not be null when reducing sparse nD ndarrays.");
        ValidateParameters.ensureValidAxes(shape, axes);

        final int rank = shape.getRank();
        if (axes.length == 0) return copy();  // No axes to reduce; simply return a copy.

        // Determine which axes survive the reduction.
        boolean[] reduceAxis = ShapeUtils.getNormalizedFlags(axes, rank);
        int reducedRank = rank - axes.length;
        int[] reducedDims = new int[reducedRank];
        int[] keepToOrigAxis = new int[reducedRank];

        for (int i = 0, j = 0; i < rank; ++i) {
            if (!reduceAxis[i]) {
                reducedDims[j] = shape.getSize(i);
                keepToOrigAxis[j++] = i;
            }
        }
        Shape reducedShape = new Shape(reducedDims);
        int[] reducedStrides = reducedShape.getStrides();

         // Accumulate values keyed using a flat index.
        Map<Integer, Double> accMap = new HashMap<>();
        for (int p = 0, nnz = data.length; p < nnz; ++p) {
            int[] coord = indices[p];

            int flatIdx = 0;
            for (int k = 0; k < reducedRank; ++k)
                flatIdx += coord[keepToOrigAxis[k]]*reducedStrides[k];

            final double value = data[p];
            accMap.compute(flatIdx, (k, oldVal) ->
                    accumulator.applyAsDouble(oldVal == null ? identity : oldVal, value));
        }

         // Build the sparse COO output tensor.
        int outNnz = accMap.size();
        double[] outData = new double[outNnz];
        int[][]  outIdx  = new int[outNnz][reducedRank];

        int q = 0;
        for (Map.Entry<Integer, Double> e : accMap.entrySet()) {
            int flat = e.getKey();
            outData[q] = e.getValue();

            // un-flatten the coordinate.
            int rem = flat;
            for (int k = 0; k < reducedRank; k++) {
                int stride = reducedStrides[k];
                int coordinate = rem / stride;
                outIdx[q][k] = coordinate;
                rem -= coordinate * stride;
            }

            q++;
        }

        return new CooTensor(reducedShape, outData, outIdx);
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public CooTensor copy() {
        return unsafeMake(shape, data.clone(), ArrayUtils.deepCopy2D(indices, null));
    }


    /**
     * Checks if each non-zero entry in this nD array satisfies the specified {@code predicate}.
     *
     * @param predicate The predicate to check each non-zero entry in this nD array against.
     *
     * @return An {@link ArrayMask} of the same shape as this nD array containing the boolean results from evaluating each
     * entry in the nD array against the {@code predicate}.
     *
     * @throws NullPointerException If {@code predicate} is {@code null}.
     * @see #filter(Function)
     */
    @Override
    public ArrayMask where(Function<Double, Boolean> predicate) {
        BitSet result = new BitSet();

        for(int i = 0, size = data.length; i<size; i++)
            result.set(i, predicate.apply(data[i]));

        return new ArrayMask(shape, result);
    }


    /**
     * Adds a scalar value to each non-zero value of this tensor.
     *
     * @param b Value to add to each non-zero value of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public CooTensor add(double b) {
        // Overrides method in super class to emphasize that the method works on the non-zero-elements only.
        return super.add(b);
    }


    /**
     * Subtracts a scalar value from each non-zero value of this tensor.
     *
     * @param b Value to subtract from each non-zero value of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public CooTensor sub(double b) {
        // Overrides method in super class to emphasize that the method works on the non-zero-elements only.
        return super.sub(b);
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     * <p><b>WARNING</b>: This method is not supported for sparse tensors. If called on a sparse tensor,
     * an {@link UnsupportedOperationException} will be thrown. Element-wise division is undefined for sparse tensors as it
     * would almost certainly result in a division by zero.
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    @Override
    public CooTensor div(CooTensor b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse tensors.");
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        CooDataSorter.wrap(data, indices).sparseSort().unwrap(data, indices);
    }


    /**
     * Coalesces this sparse COO tensor. An uncoalesced tensor is a sparse tensor with multiple items for a single index. This
     * method will ensure that each index only has one non-zero value by summing up duplicated items. If another form of aggregation other
     * than summation is desired, use {@link #coalesce(BinaryOperator)}.
     * @return A new coalesced sparse COO tensor which is equivalent to this COO tensor.
     * @see #coalesce(BinaryOperator)
     */
    public CooTensor coalesce() {
        SparseTensorData<Double> tensor = SparseUtils.coalesce(Double::sum, shape, data, indices);
        return makeLikeTensor(tensor.shape(), tensor.data(), tensor.indices());
    }


    /**
     * Coalesces this sparse COO tensor. An uncoalesced tensor is a sparse tensor with multiple items for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated items using {@code aggregator}.
     * @param aggregator Custom aggregation function to combine multiple.
     * @return A new coalesced sparse COO tensor which is equivalent to this COO tensor.
     * @see #coalesce()
     */
    public CooTensor coalesce(BinaryOperator<Double> aggregator) {
        SparseTensorData<Double> tensor = SparseUtils.coalesce(aggregator, shape, data, indices);
        return makeLikeTensor(tensor.shape(), tensor.data(), tensor.indices());
    }


    /**
     * Drops any explicit zeros in this sparse COO tensor.
     * @return A copy of this COO tensor with any explicitly stored zeros removed.
     */
    public CooTensor dropZeros() {
        SparseTensorData<Double> tensor = SparseUtils.dropZeros(shape, data, indices);
        return makeLikeTensor(tensor.shape(), tensor.data(), tensor.indices());
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooTensor src2 = (CooTensor) object;

        return RealSparseEquals.cooTensorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < data.length; i++) {
            if (data[i] != 0.0) {
                result = 31*result + Double.hashCode(data[i]);
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specifying the full shape,
     * non-zero items, and non-zero indices.
     *
     * @return A human-readable string specifying the full shape, non-zero items, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centering = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("nnz: ").append(nnz).append("\n");
        sb.append("Non-zero Entries: " +
                PrettyPrint.abbreviatedArray(data, maxCols, padding, precision, centering) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centering));

        return sb.toString();
    }
}
