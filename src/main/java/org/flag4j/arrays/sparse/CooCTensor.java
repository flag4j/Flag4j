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
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldTensor;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Complex64;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>Sparse complex tensor stored in coordinate list (COO) format. The data of this COO tensor are of type {@link Complex128}
 *
 * <p>The non-zero data and non-zero indices of a COO tensor are mutable but the {@link #shape} and total number of
 * {@link #data non-zero data} is fixed.
 *
 * <p>Sparse tensors allow for the efficient storage of and ops on tensors that contain many zero values.
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e., tensors which contain almost all zeros relative to the size of the
 * tensor).
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #data} of the tensor. All other data in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #data}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many ops assume indices to be sorted in a
 *     row-major format (i.e., last index increased fastest) but often this is not explicitly verified.
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero data in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the nD
 *     index of {@code data[i]}.
 *     </li>
 * </ul>
 */
public class CooCTensor extends AbstractCooFieldTensor<CooCTensor, CTensor, Complex128> {

    private static final long serialVersionUID = 1L;

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, Complex128[] entries, int[][] indices) {
        super(shape, entries, indices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, List<Complex128> entries, List<int[]> indices) {
        super(shape, entries.toArray(new Complex128[0]), indices.toArray(new int[0][]));
        if(super.data.length == 0 || super.data[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, Complex64[] entries, int[][] indices) {
        super(shape, new Complex128[entries.length], indices);
        setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.data[i] = new Complex128((Complex64) entries[i]);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape) {
        super(shape, new Complex128[0], new int[0][shape.getRank()]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, double[] entries, int[][] indices) {
        super(shape, ArrayConversions.toComplex128(entries, null), indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of a sparse COO tensor.
     * @param b Tensor to construct copy of.
     */
    public CooCTensor(CooCTensor b) {
        super(b.shape, b.data.clone(), ArrayUtils.deepCopy2D(b.indices, null));
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing COO tensors.
     * @param shape The shape of the tensor to construct.
     * @param data The non-zero data of this tensor.
     * @param indices The indices of the non-zero data.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooCTensor(Shape shape, Complex128[] data, int[][] indices, Object dummy) {
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
    public static CooCTensor unsafeMake(Shape shape, Complex128[] data, int[][] indices) {
        return new CooCTensor(shape, data, indices, null);
    }


    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
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
    public CooCTensor makeLikeNDArray(Shape shape, Complex128[] data) {
        return new CooCTensor(shape, data, ArrayUtils.deepCopy2D(indices, null));
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
    public CooCTensor makeLikeTensor(Shape shape, Complex128[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
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
    public CooCTensor makeLikeTensor(Shape shape, List<Complex128> entries, List<int[]> indices) {
        return new CooCTensor(shape, entries, indices);
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
    public CTensor makeLikeDenseTensor(Shape shape, Complex128[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Converts this complex vector to a real vector.
     * @return A real vector containing the real components of all non-zero values in this vector. The imaginary components are
     * ignored.
     */
    public CooTensor toReal() {
        return CooTensor.unsafeMake(shape, Complex128Ops.toReal(data), indices.clone());
    }


    /**
     * Checks if all data of this matrix are real.
     * @return {@code true} if all data of this matrix are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this matrix has non-zero imaginary component.
     * @return {@code true} if any entry of this matrix has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Rounds all data within this tensor to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new tensor containing the data of this tensor rounded to the specified precision.
     */
    public CooCTensor round(int precision) {
        return unsafeMake(shape, Complex128Ops.round(data, precision), ArrayUtils.deepCopy2D(indices, null));
    }


    /**
     * Sets all elements of this tensor to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this tensor with all data within {@code tol} of zero set to zero.
     */
    public CooCTensor roundToZero(double tolerance) {
        Complex128[] rounded = Complex128Ops.roundToZero(data, tolerance);
        List<Complex128> dest = new ArrayList<>(data.length);
        List<int[]> destIndices = new ArrayList<>(data.length);

        for(int i = 0, size = data.length; i<size; i++) {
            if(!rounded[i].isZero()) {
                dest.add(rounded[i]);
                destIndices.add(indices[i].clone());
            }
        }

        return new CooCTensor(shape, dest, destIndices);
    }


    /**
     * Converts this tensor to a matrix with specified shape.
     * @param newShape Shape of matrix to convert this tensor to. Shape must be rank 2 and have the same number of total entries as
     * this tensors shape.
     * @return A matrix of the specified shape with the same non-zero data as this tensor.
     */
    public CooCMatrix toMatrix(Shape newShape) {
        ValidateParameters.ensureRank(newShape, 2);
        CooCTensor t = reshape(newShape); // Reshape as rank 2 tensor. Total entries check made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return CooCMatrix.unsafeMake(newShape, t.data.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Sets the element of this tensor at the specified target index.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param target Index of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code target} is not within the bounds of this tensor.
     */
    public CooCTensor set(double value, int... target) {
        return super.set(new Complex128(value), target);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooCTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        return CooSemiringEquals.cooTensorEquals(this, (CooCTensor) object);
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
        sb.append("Non-zero Entries: " +
                PrettyPrint.abbreviatedArray(data, maxCols, padding, precision, centering) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centering));

        return sb.toString();
    }
}
