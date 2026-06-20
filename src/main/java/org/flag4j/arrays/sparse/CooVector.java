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
import org.flag4j.arrays.SparseVectorData;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.primitive_arrays.AbstractDoubleNDArray;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOps;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.coo.CooDataSorter;
import org.flag4j.linalg.ops.sparse.coo.real.RealCooVectorOps;
import org.flag4j.linalg.ops.sparse.coo.real.RealSparseEquals;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.*;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.*;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Function;


/**
 * <p>A real sparse vector stored in coordinate list (COO) format. The {@link #data} of this COO vector are
 * primitive doubles.
 *
 * <p>The {@link #data non-zero data} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero data is fixed.
 *
 * <p>Sparse vectors allow for the efficient storage of and ops on vectors that contain many zero values.
 *
 * <p>COO vectors are optimized for hyper-sparse vectors (i.e., vectors which contain almost all zeros relative to the size of the
 * vector).
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #data} of the vector. All other data in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Some ops on sparse tensors behave differently than on dense tensors. For instance, {@link #add(Complex128)} will not
 * add the scalar to all data of the tensor since this would cause catastrophic loss of sparsity. Instead, such non-zero preserving
 * element-wise ops only act on the non-zero data of the sparse tensor as to not affect the sparsity.
 *
 * <p>Note: many ops assume that the data of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.
 */
public class CooVector extends AbstractDoubleNDArray<CooVector>
        implements VectorMixin<CooVector, CooMatrix, Matrix, Double> {

    private static final long serialVersionUID = 1L;

    /**
     * The indices of the non-zero data in this sparse COO vector.
     */
    public final int[] indices;
    /**
     * The full size of this sparse COO vector (including the zeros.).
     */
    public final int size;
    /**
     * The number of non-zero values in this sparse COO vector.
     */
    public final int nnz;
    /**
     * The sparsity of this vector.
     */
    private double sparsity = -1.0;


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(Shape shape, double[] data, int[] indices) {
        super(shape, data);
        ValidateParameters.ensureRank(shape, 1);
        this.size = shape.getSize(0);
        this.indices = indices;
        this.nnz = data.length;
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(Shape shape, List<Double> data, List<Integer> indices) {
        super(shape, ArrayConversions.fromDoubleList(data));
        ValidateParameters.ensureRank(shape, 1);
        this.size = shape.getSize(0);
        this.indices = ArrayConversions.fromIntegerList(indices);
        this.nnz = this.data.length;
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(int size, double[] data, int[] indices) {
        super(new Shape(size), data);
        ValidateParameters.ensureArrayLengthsEq(data.length, indices.length);
        this.size = size;
        this.indices = indices;
        this.nnz = data.length;
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(int size, List<Double> data, List<Integer> indices) {
        super(new Shape(size), ArrayConversions.fromDoubleList(data));
        ValidateParameters.ensureArrayLengthsEq(data.size(), indices.size());
        this.indices = ArrayConversions.fromIntegerList(indices);
        this.size = size;
        this.nnz = this.data.length;
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Creates a zero vector of the specified {@code size}.
     */
    public CooVector(int size) {
        super(new Shape(size), new double[0]);
        indices = new int[0];
        nnz = 0;
        this.size = size;
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(int size, int[] data, int[] indices) {
        super(new Shape(size), ArrayConversions.asDouble(data, null));
        SparseValidation.validateCoo(size, data.length, indices);
        this.indices = indices;
        this.size = size;
        nnz = data.length;
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Constructs a copy of the specified sparse COO vector.
     * @param b The vector to construct a copy of.
     */
    public CooVector(CooVector b) {
        super(b.shape, b.data.clone());
        indices = b.indices.clone();
        nnz = b.nnz;
        size = b.size;
        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Constructs a sparse COO vector from index value pairs.
     * @param shape The shape of this COO vector. Must be rank 1.
     * @param indexDataMap A map containing the index value pairs of this vector.
     */
    public CooVector(Shape shape, HashMap<Integer, Double> indexDataMap) {
        super(shape, new double[indexDataMap.size()]);
        this.size = shape.getSize(0);
        this.nnz = data.length;
        indices = new int[nnz];

        int loc = 0;
        for(Map.Entry<Integer, Double> kv : indexDataMap.entrySet()) {
            data[loc] = kv.getValue();
            indices[loc++] = kv.getKey();
        }

        SparseValidation.validateCoo(this.size, this.nnz, this.indices);
    }


    /**
     * Constructor useful for avoiding unnecessary parameter validation while constructing COO vectors.
     * @param shape Shape of the COO vector to construct.
     * @param data The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooVector(Shape shape, double[] data, int[] indices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO vector in this manner is unsafe.
        super(shape, data);
        this.size = shape.getSize(0);
        this.indices = indices;
        this.nnz = data.length;
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
    public static CooVector unsafeMake(int size, double[] data, int[] indices) {
        return new CooVector(new Shape(size), data, indices, null);
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
    public static CooVector unsafeMake(Shape shape, double[] data, int[] indices) {
        return new CooVector(shape, data, indices, null);
    }


    /**
     * Creates a sparse tensor from a dense tensor.
     *
     * @param src Dense tensor to convert to a sparse tensor.
     * @return A sparse tensor which is equivalent to the {@code src} dense tensor.
     */
    public static CooVector fromDense(Vector src) {
        List<Double> nonZeroEntries = new ArrayList<>((int) (src.data.length*0.8));
        List<Integer> indices = new ArrayList<>((int) (src.data.length*0.8));

        // Fill data with non-zero values.
        for(int i = 0; i<src.data.length; i++) {
            if(src.data[i] != 0d) {
                nonZeroEntries.add(src.data[i]);
                indices.add(i);
            }
        }

        return CooVector.unsafeMake(
                src.size,
                ArrayConversions.fromDoubleList(nonZeroEntries),
                ArrayConversions.fromIntegerList(indices)
        );
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given shape and data.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given shape and data.
     */
    @Override
    public CooVector makeLikeNDArray(Shape shape, double[] data) {
        return new CooVector(shape, data, indices.clone());
    }


    /**
     * Constructs a vector of the same type as this tensor with the given shape and data.
     *
     * @param shape Shape of the vector to construct.
     * @param data Non-zero data of the vector to construct.
     * @param indices Indices of the non-zero values in this vector.
     *
     * @return A vector of the same type as this tensor with the given shape and data.
     */
    public CooVector makeLikeTensor(int size, double[] data, int[] indices) {
        return new CooVector(size, data, indices);
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
    public Vector tensorDot(CooVector src2, int[] aAxes, int[] bAxes) {
        if(aAxes.length != 1 || bAxes.length != 1) {
            throw new LinearAlgebraException("Vector dot product requires exactly one dimension for each vector but got "
                    + aAxes.length + " and " + bAxes.length + ".");
        }
        if(aAxes[0] != 0 || bAxes[0] != 0) {
            throw new LinearAlgebraException("Both axes must be 0 for vector dot product but got "
                    + aAxes[0] + " and " + bAxes[0] + ".");
        }

        return new Vector(dot(src2));
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
    public CooVector T(int axis1, int axis2) {
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        return copy();
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
    public CooVector T(int... axes) {
        ValidateParameters.ensurePermutation(axes);
        return copy();
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
    public ArrayMask where(Function<Double, Boolean> predicate) {
        BitSet mask = new BitSet();

        for(int i=0; i<data.length; i++)
            mask.set(indices[i], predicate.apply(data[i]));

        return new ArrayMask(shape, mask);
    }


    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CooVector join(CooVector b) {
        double[] newEntries = new double[this.data.length + b.data.length];
        Arrays.fill(newEntries, 0.0);
        int[] newIndices = new int[this.indices.length + b.indices.length];

        // Copy values from this vector.
        System.arraycopy(this.data, 0, newEntries, 0, this.data.length);
        // Copy values from vector b.
        System.arraycopy(b.data, 0, newEntries, this.data.length, b.data.length);

        // Copy indices from this vector.
        System.arraycopy(this.indices, 0, newIndices, 0, this.data.length);

        // Copy the indices from vector b with a shift.
        for(int i=0; i<b.indices.length; i++)
            newIndices[this.indices.length+i] = b.indices[i] + size;

        return makeLikeTensor(size + b.size, newEntries, newIndices);
    }


    /**
     * <p>Computes the inner product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #dot(CooVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #dot(CooVector)
     */
    @Override
    public Double inner(CooVector b) {
        return RealCooVectorOps.inner(this, b);
    }


    /**
     * <p>Computes the inner product between this vector and itself.
     * <p>This method <em>may</em> be slightly more efficient than calling {@link #inner(VectorMixin) this.inner(this)}.
     *
     * @return The inner product between this vector and itself.
     *
     * @see #inner(VectorMixin)
     */
    @Override
    public double innerSelf() {
        return VectorNorms.normSquared(data);
    }


    /**
     * <p>Computes the dot product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #inner(CooVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #inner(CooVector) 
     */
    @Override
    public Double dot(CooVector b) {
        return inner(b);
    }


    /**
     * Computes the norm of this vector. This is the same as {@link #mag()}.
     * @return The norm (specifically &ell;<sup>2</sup>) of this vector.
     * @see #mag()
     * @see #magSquared()
     */
    @Override
    public double norm() {
        return VectorNorms.norm(data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm(int p) {
        return VectorNorms.norm(data, p);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector.
     * If this vector is all zeros, then an equivalently sized
     * zero vector will be returned.
     */
    public CooVector normalize() {
        double norm = VectorNorms.norm(data);
        return norm==0 ? new CooVector(size) : this.div(norm);
    }


    /**
     * Computes the squared magnitude of this vector.
     *
     * @return The squared magnitude of this vector.
     *
     * @see #mag()
     */
    @Override
    public double magSquared() {
        return VectorNorms.normSquared(data);
    }


    /**
     * Gets multiple items from this vector.
     *
     * @param indices The indices of each item to get from this vector.
     *
     * @return A vector containing the entries of this vector at the specified {@code indices}.
     *
     * @throws IndexOutOfBoundsException If any index in {@code indices} is not within the bounds of this vector.
     * @see #get(int)
     * @see #getSlice(int, int)
     */
    @Override
    public Vector getItems(int... indices) {
        double[] itemData = new double[indices.length];

        for(int i=0; i<indices.length; i++)
            itemData[i] = get(indices[i]);

        return new Vector(itemData);
    }


    /**
     * Gets the element of this vector at the specified index.
     *
     * @param target Index of the element to get within this vector.
     *
     * @return The element of this vector at index {@code target}.
     */
    @Override
    public Double get(int target) {
        ValidateParameters.validateTensorIndex(shape, target);
        int idx = Arrays.binarySearch(indices, target);
        return idx>=0 ? data[idx] : 0;
    }


    /**
     * Gets a slice of this vector over the specified range of indices.
     *
     * @param startIdx Staring index of slice (inclusive).
     * @param endIdx Ending index of slice (exclusive).
     *
     * @return A vector of length {@code endIdx - startIdx} whose entries are the elements of this vector
     * over the specified range of indices [{@code startIdx}, {@code endIdx}).
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} or {@code endIdx - 1} are not within the bounds of this vector.
     * @throws IllegalArgumentException  If {@code startIdx >= endIdx}.
     * @see #get(int)
     */
    @Override
    public CooVector getSlice(int startIdx, int endIdx) {
        if (startIdx >= endIdx) {
            throw new IllegalArgumentException("startIdx must be less than endIdx but got startIdx="
                    + startIdx + " and endIdx=" + endIdx + ".");
        }
        ValidateParameters.validateTensorIndex(shape, startIdx);
        ValidateParameters.validateTensorIndex(shape, endIdx-1);

        int startLoc = Arrays.binarySearch(indices, startIdx);
        if(startLoc < 0) startLoc = -startLoc - 1;

        int endLoc = Arrays.binarySearch(indices, endIdx);
        if(endLoc < 0) endLoc = -endLoc - 1;

        int[] slicesIndices = Arrays.copyOfRange(indices, startLoc, endLoc);
        ArrayUtils.shift(-startIdx, slicesIndices);

        return new CooVector(endIdx-startIdx,
                Arrays.copyOfRange(data, startLoc, endLoc),
                slicesIndices);
    }


    /**
     * Sets a slice of this vector to the entries of another vector.
     *
     * @param values A vector containing the values to set.
     * @param startIdx The starting index of the slice to set. The size of the slice will be {@code values.length}.
     *
     * @return The operation will be done out-of-place in a copy of this vector, and that copy will be returned.
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this vector.
     * @throws IllegalArgumentException  If {@code values} does not fit within this vector when its first entry is placed
     *                                   at {@code startIdx}.
     */
    @Override
    public CooVector setSlice(CooVector values, int startIdx) {
        int startLoc = Arrays.binarySearch(indices, startIdx);
        int endLoc = Arrays.binarySearch(indices, startIdx + values.size);

        if(startLoc < 0) startLoc = -startLoc - 1;
        if(endLoc < 0) endLoc = -endLoc - 1;

        int inVectorSliceSize = endLoc - startLoc;

        double[] newValues = new double[values.nnz + nnz - inVectorSliceSize];
        int[] newIndices = new int[newValues.length];

        System.arraycopy(data, 0, newValues, 0, startLoc);
        System.arraycopy(data, endLoc, newValues, startLoc, nnz - endLoc);
        System.arraycopy(values.data, 0, newValues, startLoc + nnz - endLoc, values.nnz);

        System.arraycopy(indices, 0, newIndices, 0, startLoc);
        System.arraycopy(indices, endLoc, newIndices, startLoc, nnz - endLoc);
        System.arraycopy(values.indices, 0, newIndices, startLoc + nnz - endLoc, values.nnz);

        ArrayUtils.shift(startIdx, newIndices);

        return new CooVector(shape, newValues, newIndices);
    }


    /**
     * Sets a slice of this vector to the entries of an array.
     *
     * @param values Array containing the values to set.
     * @param startIdx The starting index of the slice to set. The size of the slice will be {@code values.length}.
     *
     * @return If this vector is dense, the operation will be done in-place and a reference to this tensor will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and this copy will be returned.
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this vector.
     * @throws IllegalArgumentException  If {@code values} does not fit within this vector when its first entry is placed
     *                                   at {@code startIdx}.
     */
    @Override
    public CooVector setSlice(Double[] values, int startIdx) {
        return setSlice(ArrayConversions.unbox(values, null), startIdx);
    }


    /**
     * Sets a slice of this vector to the entries of an array.
     *
     * @param values Array containing the values to set.
     * @param startIdx The starting index of the slice to set. The size of the slice will be {@code values.length}.
     *
     * @return If this vector is dense, the operation will be done in-place and a reference to this tensor will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and this copy will be returned.
     *
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this vector.
     * @throws IllegalArgumentException  If {@code values} does not fit within this vector when its first entry is placed
     *                                   at {@code startIdx}.
     */
    public CooVector setSlice(double[] values, int startIdx) {
        int startLoc = Arrays.binarySearch(indices, startIdx);
        int endLoc = Arrays.binarySearch(indices, startIdx + values.length);

        if(startLoc < 0) startLoc = -startLoc - 1;
        if(endLoc < 0) endLoc = -endLoc - 1;

        int inVectorSliceSize = endLoc - startLoc;

        double[] newValues = new double[values.length + nnz - inVectorSliceSize];
        int[] newIndices = new int[newValues.length];
        int[] valueIndices = ArrayBuilder.intRange(startIdx, startIdx + values.length);

        System.arraycopy(data, 0, newValues, 0, startLoc);
        System.arraycopy(data, endLoc, newValues, startLoc, nnz - endLoc);
        System.arraycopy(values, 0, newValues, startLoc + nnz - endLoc, values.length);

        System.arraycopy(indices, 0, newIndices, 0, startLoc);
        System.arraycopy(indices, endLoc, newIndices, startLoc, nnz - endLoc);
        System.arraycopy(valueIndices, 0, newIndices, startLoc + nnz - endLoc, values.length);

        return new CooVector(shape, newValues, newIndices);
    }


    /**
     * Sets multiple items of this vector.
     *
     * @param values New values it set the specified items to.
     * @param indices The indices indicating where each value in {@code values} should be set within this vector.
     *
     * @return If this vector is dense, the operation will be done in-place and a reference to this vector will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and that copy will be returned.
     *
     * @throws IndexOutOfBoundsException If any index in {@code indices} is not within the bounds of this vector.
     * @throws IllegalArgumentException  If {@code values.length != indices.length}.
     */
    @Override
    public CooVector setItems(Double[] values, int[] indices) {
        ValidateParameters.ensureArrayLengthsEq(values.length, indices.length);

        // Create a map with the vector's indices and values
        HashMap<Integer, Double> valueMap = new HashMap<>(nnz);
        for(int i=0; i<nnz; i++)
            valueMap.put(indices[i], values[i]);

        // Add or overwrite the vector's values.
        for(int i=0, size=values.length; i<size; i++)
            valueMap.put(indices[i], values[i]);

        return new CooVector(shape, valueMap);
    }


    /**
     * Sets multiple items of this vector.
     *
     * @param values New values it set the specified items to.
     * @param indices The indices indicating where each value in {@code values} should be set within this vector.
     *
     * @return If this vector is dense, the operation will be done in-place and a reference to this vector will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and that copy will be returned.
     *
     * @throws IndexOutOfBoundsException If any index in {@code indices} is not within the bounds of this vector.
     * @throws IllegalArgumentException  If {@code values.length != indices.length}.
     */
    public CooVector setItems(double[] values, int[] indices) {
        ValidateParameters.ensureArrayLengthsEq(values.length, indices.length);

        // Create a map with the vector's indices and values
        HashMap<Integer, Double> valueMap = new HashMap<>(nnz);
        for(int i=0; i<nnz; i++)
            valueMap.put(indices[i], values[i]);

        // Add or overwrite the vector's values.
        for(int i=0, size=values.length; i<size; i++)
            valueMap.put(indices[i], values[i]);

        return new CooVector(shape, valueMap);
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return {@code true} if the vector {@code b} is parallel to this vector and the same size; {@code false} otherwise.
     * @implNote This method uses a tolerance of {@code tol = 1.0e-12} to verify if the two vectors are parallel. Specifically, the
     * scaled
     * difference between the two vectors is computed using their first non-zero data as
     * {@code scale = this.data[i]/b.data[this.indices[i]]} for some {@code i} such that
     * {@code b.data[this.indices[i]] != 0.0}. Then if all data satisfy
     * {@code Math.abs(b.data[i]*scale - data[j]) < tol} where
     * {@code this.indices[i] == b.indices[j]}, the two vectors will be considered parallel.
     * @see #isPerp(CooVector)
     */
    public boolean isParallel(CooVector b) {
        final double tol = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        boolean result;

        if(this.size!=b.size) {
            return false;
        } else if(this.size<=1) {
            return true;
        } else if(this.isAllZeros() || b.isAllZeros()) {
            return true; // Any vector is parallel to a zero vector.
        } else {
            result = true;
            int sparseIndex = 0;
            double scale = 0;

            // Find the first non-zero entry in b and compute the scaling factor (we know there is at least one from else-if).
            for(int i=0; i<b.size; i++) {
                if(b.data[i]!=0) {
                    scale = this.data[i]/b.data[this.indices[i]];
                    break;
                }
            }

            for(int i=0; i<b.size; i++) {
                if(sparseIndex >= this.data.length || i!=this.indices[sparseIndex]) {
                    // Then this index is not in the sparse vector.
                    if(b.data[i] != 0) return false;

                } else {
                    // Ensure the scaled entry is approximately equal to the entry in this vector.
                    if(Math.abs(b.data[i]*scale - data[sparseIndex]) > tol) return false;
                    sparseIndex++;
                }
            }
        }

        return true;
    }


    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return {@code true} if the vector {@code b} is the same size and perpendicular to this vector; {@code false} otherwise.
     *
     * @see #isParallel(CooVector)
     * @implNote This method checks if the vector is perpendicular by checking if the inner product is essentially zero:
     * {@code Math.abs(this.inner(b)) < TOL} where {@code TOL} is a small non-negative value.
     */
    public boolean isPerp(CooVector b) {
        final double TOL = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        return size!=b.size ? false : Math.abs(inner(b)) < TOL;
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e., the number of data, in this vector.
     */
    @Override
    public int length() {
        return size;
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
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
     * Gets the density of this matrix as a decimal percentage.
     * That is, the percentage of data in this matrix that are non-zero.
     * @return The density of this matrix as a decimal percentage.
     * @see #getSparsity()
     */
    public double getDensity() {
        return 1.0 - sparsity;
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    public Vector toDense() {
        double[] denseData = new double[size];
        for(int i = 0; i<nnz; i++)
            denseData[indices[i]] = this.data[i];

        return new Vector(denseData);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        CooDataSorter.wrap(data, indices).sparseSort().unwrap(data, indices);
    }


    /**
     * Gets the element of this tensor at the specified target index.
     *
     * @param target Index of the element to get.
     *
     * @return The element of this tensor at the specified target index.
     * @throws IllegalArgumentException If {@code target.length != 1}.
     */
    @Override
    public Double get(int... target) {
        ValidateParameters.ensureArrayLengthsEq(target.length, 1);
        return get(target[0]);
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
        double[] values = new double[mask.countTrue()];
        BitSet maskBits = mask.data;

        for(int i = maskBits.nextSetBit(0), j = 0; i >= 0; i = maskBits.nextSetBit(i+1))
            values[j++] = get(i);

        return new Vector(values);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooVector set(Double value, int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        int idx = Arrays.binarySearch(this.indices, indices[0]);
        double[] destEntries;
        int[] destIndices;

        if(idx >= 0) {
            // Then the index was found in the sparse vector.
            destIndices = this.indices.clone();
            destEntries = data.clone();
            destEntries[idx] = value;

        } else{
            // Then the index was not found in the sparse vector.
            destIndices = new int[this.indices.length+1];
            destEntries = new double[data.length+1];
            idx = -(idx+1);

            System.arraycopy(this.indices, 0, destIndices, 0, idx);
            destIndices[idx] = indices[0];
            System.arraycopy(this.indices, idx, destIndices, idx+1, this.indices.length-idx);

            System.arraycopy(data, 0, destEntries, 0, idx);
            destEntries[idx] = value;
            System.arraycopy(data, idx, destEntries, idx+1, data.length-idx);
        }

        return CooVector.unsafeMake(size, destEntries, destIndices);
    }


    /**
     * Flattens tensor to a single dimension while preserving the order of data.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CooVector flatten() {
        return copy();
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
    public CooVector flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        return copy();
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws NDArrayShapeException If {@code newShape} does not have the same number of total entries as
     * {@link #shape this.shape}.
     */
    @Override
    public CooVector reshape(Shape newShape) {
        ValidateParameters.ensureRank(newShape, 1);
        ValidateParameters.ensureTotalEntriesEqual(shape, newShape);
        return copy();
    }


    /**
     * Reduces elements of this array, along a specified set of axes, by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to elements of this nD array along the specified axes in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * <strong>Warning:</strong> This method is provided as part of the nD Array API.
     * However, it is unlikely that users would wish to use this method over {@link #reduce(Double, DoubleBinaryOperator)}
     * or {@link #reduceToDense(Double, DoubleBinaryOperator, int...)} for a vector.
     * There are only two valid values for {@code axes} as follows:
     * <ul>
     *     <li>If {@code axes.length == 0} then this method is equivalent to calling {@link #toTensor()}.</li>
     *     <li>If {@code axes.length == 1} and {@code axes[0] == 0} then this method will return a sparse scalar tensor with
     *     a single value.
     *     This scalar value can be gotten in a dense scalar tensor or as a single primitive value by calling
     *     {@link #reduceToDense(Double, DoubleBinaryOperator, int...)} or {@link #reduce(Double, DoubleBinaryOperator)}
     *     respectively.</li>
     * </ul>
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
     * @see #reduceToDense(Double, DoubleBinaryOperator, int...)
     */
    @Override
    public CooTensor reduce(Double identity, DoubleBinaryOperator accumulator, int... axes) {
        Objects.requireNonNull(identity, "The identity object must not be null when reducing sparse nD arrays.");

        if(axes.length == 0) return toTensor();
        if(axes.length != 1 || axes[0] != 0) {
            throw new IllegalArgumentException(
                    "Illegal axes specified for vector. Only a single axis (zero) may be specified for vector reduction.");
        }

        return new CooTensor(
                new Shape(),
                new double[]{reduce(identity, accumulator)},
                new int[][]{{0}}
        );
    }


    /**
     * Reduces elements of this array, along a specified set of axes, by repeatedly applying
     * the specified {@code accumulator} to an ongoing intermediate result that is initialized to
     * {@code identity}.
     *
     * <p>The {@code accumulator} is applied to elements of this nD array along the specified axes in order.
     * If this nD array is sparse, then the {@code accumulator} will <em>only</em> be
     * applied to the non-zero elements of this nD array.
     *
     * <strong>Warning:</strong> Not specifying any axes (i.e., {@code axes.length == 0}) will result in converting this
     * COO vector to a dense tensor which is likely not desirable.
     * <ul>
     *     <li>If {@code axes.length == 0} then this method is equivalent to calling {@code this.toDense().toTensor()}.</li>
     *     <li>If {@code axes.length == 1} and {@code axes[0] == 0} then this method is equivalent to calling
     *     {@code this.reduce(identity, accumulator)} then wrapping it in a scalar {@link Tensor}.</li>
     * </ul>
     *
     * @param identity The starting value for the reduction.
     * <strong>Note:</strong>Unlike with dense nD array objects, the identity may <i>not</i> be {@code null}.
     * @param accumulator The binary operator used to accumulate elements of this nD array.
     * For the results to be well-defined, the accumulator must be associative and communitive.
     * @param axes The axes along which reduce this nD array.
     *
     * @return A dense nD array of the same shape as this nD array, but with the specified {@code axes} removed, containing the
     * result of the reduction operation.
     *
     * @throws NullPointerException If {@code accumulator} is {@code null}.
     * @see #reduce(Double, BinaryOperator)
     */
    public Tensor reduceToDense(Double identity, DoubleBinaryOperator accumulator, int... axes) {
        Objects.requireNonNull(identity, "The identity object must not be null when reducing sparse nD arrays.");

        if(axes.length == 0) return toDense().toTensor();
        if(axes.length != 1 || axes[0] != 0) {
            throw new IllegalArgumentException(
                    "Illegal axes specified for vector. Only a single axis (zero) may be specified for vector reduction.");
        }

        return new Tensor(
                new Shape(),
                new double[]{reduce(identity, accumulator)}
        );
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor's non-zero values and the scalar {@code b}.
     */
    @Override
    public CooVector sub(Double b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor's non-zero values and the scalar {@code b}.
     */
    public CooCVector sub(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.sub(data, b, dest);
        return CooCVector.unsafeMake(shape, dest, indices.clone());
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Double b) {
        super.subEq(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Adds a scalar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public CooVector add(Double b) {
        return super.add(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Adds a scalar field value to each non-zero entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sparse COO vector resulting from adding {@code b} to each non-zero entry of this vector.
     */
    public CooCVector add(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.add(data, b, dest);
        return CooCVector.unsafeMake(shape, dest, indices.clone());
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        super.addEq(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
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
    public CooVector add(CooVector b) {
        return RealCooVectorOps.add(this, b);
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
    public CooCVector add(CooCVector b) {
        return RealComplexSparseVectorOps.add(b, this);
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
    public CooVector sub(CooVector b) {
        return RealCooVectorOps.sub(this, b);
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
    public CooVector elemMult(CooVector b) {
        return RealCooVectorOps.elemMult(this, b);
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
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than, this tensor's rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e., the axes are equal, or the tensor does not have the same length along the two axes.)
     */
    @Override
    public CooVector tensorTr(int axis1, int axis2) {
        throw new LinearAlgebraException("Tensor trace cannot be computed for a rank 1 tensor " +
                "(must be rank 2 or " + "greater).");
    }


    /**
     * Computes the product of all non-zero values in this tensor.
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public Double prod() {
        return super.prod(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public CooVector T() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public CooVector H() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public CooVector H(int axis1, int axis2) {
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        return copy();
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public CooVector H(int... axes) {
        ValidateParameters.ensurePermutation(axes);
        return copy();
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        return new int[]{indices[RealProperties.argmin(data)]};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return new int[]{indices[RealProperties.argmax(data)]};
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return new int[]{indices[RealProperties.argminAbs(data)]};
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return new int[]{indices[RealProperties.argmaxAbs(data)]};
    }


    /**
     * Computes the element-wise reciprocals of the non-zero values in this tensor.
     *
     * @return A tensor containing the reciprocal elements of the non-zero values in this tensor.
     */
    @Override
    public CooVector recip() {
        // Overrides method from super class to emphasize that this method works on the non-zero values only.
        return super.recip();
    }


    /**
     * Adds a scalar value to each non-zero-element of this tensor.
     *
     * @param b Value to add to each non-zero entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public CooVector add(double b) {
        // Overrides method from super class to emphasize that this method works on the non-zero values only.
        return super.add(b);
    }


    /**
     * Subtracts a scalar value non-zero from each element of this tensor.
     *
     * @param b Value to subtract from each non-zero entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public CooVector sub(double b) {
        // Overrides method from super class to emphasize that this method works on the non-zero values only.
        return super.sub(b);
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     *
     * <p><b>WARNING</b>: This method is not supported for sparse vectors. If called on a sparse vector,
     * an {@link UnsupportedOperationException} will be thrown. Element-wise division is undefined for sparse vectors as it
     * would almost certainly result in a division by zero.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    @Override
    public CooVector div(CooVector b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse vectors.");
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooVector src2 = (CooVector) object;

        return RealSparseEquals.cooVectorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < data.length; i++) {
            if (data[i] != 0.0) {
                result = 31*result + Double.hashCode(data[i]);
                result = 31*result + Integer.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat a vector.
     * @param axis Axis along which to repeat vector:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    @Override
    public CooMatrix repeat(int n, int axis) {
        return RealCooVectorOps.repeat(this, n, axis);
    }


    /**
     * Stacks two vectors vertically as if they were row vectors to form a matrix with two rows.
     *
     * @param b Vector to stack below this vector.
     *
     * @return The result of stacking this vector and vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of data in
     *                                  the vector {@code b}.
     */
    @Override
    public CooMatrix stack(CooVector b) {
        return RealCooVectorOps.stack(this, b);
    }


    /**
     * <p>Stacks two vectors along the specified axis.
     *
     * <p>Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2&times;n} matrix.
     *
     * <p>Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n&times;2} matrix.
     *
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of
     *                                  data in the vector {@code b}.
     * @throws IllegalArgumentException If {@code axis} is not either 0 or 1.
     */
    @Override
    public CooMatrix stack(CooVector b, int axis) {
        ValidateParameters.ensureAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of data.
     */
    @Override
    public Matrix outer(CooVector b) {
        return RealCooVectorOps.outerProduct(this, b);
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
    public CooMatrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[data.length];

            return CooMatrix.unsafeMake(new Shape(this.size, 1), data.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[data.length];
            int[] colIndices = indices.clone();

            return CooMatrix.unsafeMake(new Shape(1, this.size), data.clone(), rowIndices, colIndices);
        }
    }


    /**
     * <p>Converts this vector to a matrix with a specified shape.
     * <p>Note, the following must be satisfied: {@code rows*cols == this.size}.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     *
     * @return A matrix with the specified number of rows and columns containing the entries of this vector.
     */
    @Override
    public CooMatrix toMatrix(int rows, int cols) {
        return toMatrix(new Shape(rows, cols));
    }


    /**
     * <p>Converts this vector to a matrix with a specified shape.
     * <p>Note, the following must be satisfied: {@code shape.totalEntriesIntValueExact() == this.size}.
     *
     * @param shape Shape of the matrix. Must be rank 2.
     *
     * @return A matrix with the specified number of rows and columns containing the entries of this vector.
     */
    @Override
    public CooMatrix toMatrix(Shape shape) {
        return toMatrix().reshape(shape);
    }


    /**
     * Converts this vector to an equivalent sparse vector.
     * @return A complex COO vector equivalent to this vector.
     */
    public CooCVector toComplex() {
        return new CooCVector(size, data, indices.clone());
    }


    /**
     * Computes the element-wise multiplication between this vector and a real dense vector.
     * @param b The real dense vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     * @throws NDArrayShapeException If the two vectors have different sizes.
     */
    public CooVector elemMult(Vector b) {
        return RealDenseSparseVectorOps.elemMult(b, this);
    }


    /**
     * Multiplies this vector by a complex scalar value.
     * @param factor Scalar value to multiply this vector by.
     * @return The result of multiplying this vector by the scalar {@code factor}.
     */
    public CooCVector mult(Complex128 factor) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.mult(data, factor, dest);
        return CooCVector.unsafeMake(shape, dest, indices.clone());
    }


    /**
     * Converts this sparse vector to an equivalent tensor.
     * @return A tensor equivalent to this vector.
     */
    public CooTensor toTensor() {
        return CooTensor.unsafeMake(
                this.shape,
                this.data.clone(),
                RealDenseTranspose.standardIntMatrix(new int[][]{this.indices})
        );
    }


    /**
     * Computes the element-wise division of two vectors.
     * @param b The second vector in the element-wise quotient (denominator).
     * @return The element-wise quotient of this vector and {@code b}.
     */
    public CooVector div(Vector b) {
        return RealDenseSparseVectorOps.elemDiv(this, b);
    }


    /**
     * Computes the element-wise division of two vectors.
     * @param b The second vector in the element-wise quotient (denominator).
     * @return The element-wise quotient of this vector and {@code b}.
     */
    public CooCVector div(CVector b) {
        return RealComplexDenseSparseVectorOps.elemDiv(this, b);
    }


    /**
     * Divides each element of this sparse COO vector by a complex-valued scalar.
     * @param divisor Scalar in the vector-scalar quotient.
     * @return The vector-scalar quotient of this vector and {@code divisor}.
     */
    public CooCVector div(Complex128 divisor) {
        return CooCVector.unsafeMake(size, Complex128Ops.scalDiv(data, divisor), indices.clone());
    }


    /**
     * Coalesces this sparse COO vector. An uncoalesced vector is a sparse vector with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by summing up duplicated data. 
     * If another form of aggregation other than summation is desired, use {@link #coalesce(BinaryOperator)}.
     * @return A new coalesced sparse COO vector which is equivalent to this COO vector.
     * @see #coalesce(BinaryOperator)
     */
    public CooVector coalesce() {
        SparseVectorData<Double> vec = SparseUtils.coalesce(Double::sum, shape, data, indices);
        return new CooVector(vec.shape(), vec.data(), vec.indices());
    }


    /**
     * Coalesces this sparse COO vector. An uncoalesced vector is a sparse vector with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data using {@code aggregator}.
     * @param aggregator Custom aggregation function to combine multiple.
     * @return A new coalesced sparse COO vector which is equivalent to this COO vector.
     * @see #coalesce()
     */
    public CooVector coalesce(BinaryOperator<Double> aggregator) {
        SparseVectorData<Double> vec = SparseUtils.coalesce(aggregator, shape, data, indices);
        return new CooVector(vec.shape(), vec.data(), vec.indices());
    }


    /**
     * Drops any explicit zeros in this sparse COO vector.
     * @return A copy of this COO vector with any explicitly stored zeros removed.
     */
    public CooVector dropZeros() {
        SparseVectorData<Double> vec = SparseUtils.dropZeros(shape, data, indices);
        return new CooVector(vec.shape(), vec.data(), vec.indices());
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

            // Get the last entry now
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
