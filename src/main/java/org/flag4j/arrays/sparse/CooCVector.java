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
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseVectorOps;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * <p>A complex sparse vector stored in coordinate list (COO) format. The {@link #data} of this COO vector are
 * {@link Complex128}s.
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
 * <p>Note: many ops assume that the data of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.
 */
public class CooCVector extends AbstractCooFieldVector<CooCVector, CVector, CooCMatrix, CMatrix, Complex128> {

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param size Full size of the vector.
     * @param entries Non-zero data of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooCVector(int size, Complex128[] entries, int[] indices) {
        super(new Shape(size), entries, indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Full shape of the vector. Must be rank 1.
     * @param entries Non-zero data of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooCVector(Shape shape, Complex128[] entries, int[] indices) {
        super(shape, entries, indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a complex COO vector with the specified size, non-zero data, and non-zero indices.
     * @param size Full size of the vector.
     * @param entries The non-zero data of the vector.
     * @param indices The indices of the non-zero data.
     */
    public CooCVector(int size, List<Complex128> entries, List<Integer> indices) {
        super(new Shape(size), entries.toArray(new Complex128[0]), ArrayConversions.fromIntegerList(indices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a complex COO vector with the specified size, non-zero data, and non-zero indices.
     * @param shape Full shape of the sparse vector. Must be rank 1.
     * @param entries The non-zero data of the vector.
     * @param indices The indices of the non-zero data.
     */
    public CooCVector(Shape shape, List<Complex128> entries, List<Integer> indices) {
        super(shape, entries.toArray(new Complex128[0]), ArrayConversions.fromIntegerList(indices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero vector of the specified {@code size}.
     * @param size Full size of the vector.
     */
    public CooCVector(int size) {
        super(new Shape(size), new Complex128[0], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a sparse complex COO vector from an array of double values.
     * @param size Full size of the vector.
     * @param entries Non-zero data of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooCVector(int size, double[] entries, int[] indices) {
        super(new Shape(size), ArrayConversions.toComplex128(entries, null), indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of the specified vector.
     * @param b The vector to create a copy of.
     */
    public CooCVector(CooCVector b) {
        super(b.shape, b.data.clone(), b.indices.clone());
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a sparse COO vector from index value pairs.
     * @param shape The shape of this COO vector. Must be rank 1.
     * @param indexDataMap A map containing the index value pairs of this vector.
     */
    public CooCVector(Shape shape, HashMap<Integer, Complex128> indexDataMap) {
        super(shape, new Complex128[indexDataMap.size()], new int[indexDataMap.size()]);

        int loc = 0;
        for(Map.Entry<Integer, Complex128> kv : indexDataMap.entrySet()) {
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
    private CooCVector(Shape shape, Complex128[] data, int[] indices, Object dummy) {
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
    public static CooCVector unsafeMake(int size, Complex128[] data, int[] indices) {
        return new CooCVector(new Shape(size), data, indices, null);
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
    public static CooCVector unsafeMake(Shape shape, Complex128[] data, int[] indices) {
        return new CooCVector(shape, data, indices, null);
    }


    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
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
    public CooCVector makeLikeTensor(Shape shape, Complex128[] entries, int[] indices) {
        return new CooCVector(shape, entries, indices);
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
    public CVector makeLikeDenseTensor(Shape shape, Complex128... entries) {
        return new CVector(shape, entries);
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
    public CMatrix makeLikeDenseMatrix(Shape shape, Complex128... entries) {
        return new CMatrix(shape, entries);
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
    public CooCVector makeLikeTensor(Shape shape, List<Complex128> entries, List<Integer> indices) {
        return new CooCVector(shape, entries, indices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and row and column indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     *
     * @return A COO matrix of a similar type as this vector with the specified shape, non-zero data, and non-zero row/col indices.
     */
    @Override
    public CooCMatrix makeLikeMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
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
    public CooCVector makeLikeNDArray(Shape shape, Complex128[] data) {
        return new CooCVector(shape, data, indices.clone());
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooCTensor toTensor() {
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(new int[][]{indices.clone()});
        return CooCTensor.unsafeMake(shape, data.clone(), tIndices);
    }


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooCTensor toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }


    /**
     * Converts this complex vector to a real vector.
     * @return A real vector containing the real components of all non-zero values in this vector. The imaginary components are
     * ignored.
     */
    public CooVector toReal() {
        return CooVector.unsafeMake(size, Complex128Ops.toReal(data), indices.clone());
    }


    /**
     * Checks if all data of this matrix are real.
     * @return {@code true} if all data of this matrix are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this matrix has a non-zero imaginary part.
     * @return {@code true} if any entry of this matrix has a non-zero imaginary part.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Rounds all data within this vector to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new vector containing the data of this vector rounded to the specified precision.
     */
    public CooCVector round(int precision) {
        return unsafeMake(shape, Complex128Ops.round(data, precision), indices.clone());
    }


    /**
     * Sets all elements of this vector to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this vector with all data within {@code tol} of zero set to zero.
     */
    public CooCVector roundToZero(double tolerance) {
        Complex128[] rounded = Complex128Ops.roundToZero(data, tolerance);
        List<Complex128> dest = new ArrayList<>(data.length);
        List<Integer> destIndices = new ArrayList<>(data.length);

        for(int i = 0, size = data.length; i<size; i++) {
            if(!rounded[i].isZero()) {
                dest.add(rounded[i]);
                destIndices.add(indices[i]);
            }
        }

        return new CooCVector(shape, dest, destIndices);
    }


    /**
     * Computes the element-wise sum of two vectors.
     * @param b Second vector in the sum.
     * @return The element-wise sum of this vector and {@code b}.
     */
    public CooCVector add(CooVector b) {
        return RealComplexSparseVectorOps.add(this, b);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public CooCVector normalize() {
        return div(mag());
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public double mag() {
        double mag = 0;

        for(Complex128 v : data)
            mag += (v.re*v.re + v.im*v.im);

        return Math.sqrt(mag);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooCVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        return CooSemiringEquals.cooVectorEquals(this, (CooCVector) object);
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
        boolean centering = PrintOptions.useCentering();
        int padding = PrintOptions.getPadding();
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
