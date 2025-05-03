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

package org.flag4j.arrays.dense;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldVectorOps;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Complex64;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;

// TODO: Javadoc.
public class CVector extends AbstractDenseFieldVector<CVector, CMatrix, Complex128> {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Complex128... entries) {
        super(new Shape(entries.length), entries);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Complex64... entries) {
        this(new Shape(entries.length), ArrayConversions.toComplex128(entries, null));
    }


    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(double... entries) {
        this(new Shape(entries.length), ArrayConversions.toComplex128(entries, null));
    }


    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(int... entries) {
        this(new Shape(entries.length), ArrayConversions.toComplex128(entries, null));
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex128 fillValue) {
        this(new Shape(size), fillValue);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex64 fillValue) {
        this(new Shape(size), fillValue);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, double fillValue) {
        this(new Shape(size), fillValue);
    }


    /**
     * Creates a complex zero vector with the specified {@code size}.
     * @param size The size of the vector.
     */
    public CVector(int size) {
        this(new Shape(size), Complex128.ZERO);
    }


    /**
     * Creates a copy of the provided vector.
     * @param vector Vector to create a copy of.
     */
    public CVector(CVector vector) {
        this(vector.shape, vector.data.clone());
    }


    /**
     * Constructs a dense complex vector with the given shape and entries.
     * @param shape The shape of the vector. Must be rank-1 and satisfy {@code shape.totalEntriesIntValueExact() == data.length}.
     * @param data The entries of the vector.
     * @throws LinearAlgebraException If {@code shape.getRank() != 1}
     * @throws IllegalArgumentException If {@code shape.totalEntriesIntValueExact() != data.length}
     */
    public CVector(Shape shape, Complex128[] data) {
        super(shape, data);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero vector with the specified shape.
     * @param shape Shape of the zero vector to construct. Must be rank 1.
     */
    public CVector(Shape shape) {
        this(shape, Complex128.ZERO);
    }


    /**
     * Constructs an empty vector with size 0.
     */
    public CVector() {
        super(new Shape(0), new Complex128[]{});
    }


    /**
     * Constructs a vector of the specified {@code shape} filled with {@code fillValue}.
     * @param shape Shape of the vector. Must be rank 1.
     * @param fillValue Value to fill this vector with.
     */
    public CVector(Shape shape, double fillValue) {
        this(shape, new Complex128(fillValue));
    }


    /**
     * Constructs a vector of the specified {@code shape} filled with {@code fillValue}.
     * @param shape Shape of the vector. Must be rank 1.
     * @param fillValue Value to fill this vector with.
     */
    public CVector(Shape shape, Complex64 fillValue) {
        this(shape, new Complex128(fillValue));
    }


    /**
     * Constructs a vector of the specified {@code shape} filled with {@code fillValue}.
     * @param shape Shape of the vector. Must be rank 1.
     * @param fillValue Value to fill this vector with.
     */
    public CVector(Shape shape, Complex128 fillValue) {
        super(shape, new Complex128[shape.get(0)]);
        Arrays.fill(data, fillValue);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a complex vector from an array of {@code String}s. Each {@code String} must be
     * parsable as a complex number.
     * @param data The entries of this vector.
     * @throws org.flag4j.util.exceptions.Flag4jParsingException If any of the {@code String}s in {@code data} cannot be
     * parsed as a complex number.
     */
    public CVector(String... data) {
        this(new Shape(data.length), ArrayConversions.toComplex128(data, null));
    }


    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
    }


    /**
     * Sets the specified index of this vector to the provided real value.
     * @param val Value to set in this vector.
     * @param index Index to set to {@code val} in this vector.
     * @throws IndexOutOfBoundsException If {@code index} is not in bounds of this vector.
     */
    public void set(double val, int index) {
        set(new Complex128(val), index);
    }


    /**
     * Constructs an empty vector with the specified size. The data of the resulting vector will be
     * all be {@code null}.
     * @param size The size of the vector to construct.
     * @return An empty vector (i.e., filled with {@code null} values) with the specified size.
     */
    public static CVector getEmpty(int size) {
        return new CVector(new Complex128[size]);
    }


    /**
     * Constructs a dense vector with the specified {@code data} of the same type as the vector.
     *
     * @param entries Entries of the dense vector to construct.
     */
    @Override
    public CVector makeLikeTensor(Complex128[] entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a matrix of a similar type to this vector with the specified {@code shape} and {@code data}.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of a similar type to this vector with the specified {@code shape} and {@code data}.
     */
    @Override
    public CMatrix makeLikeMatrix(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector add(CooCVector b) {
        return (CVector) DenseCooFieldVectorOps.add(this, b);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector add(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void addEq(Vector b) {
        RealFieldDenseOps.add(shape, data, b.shape, b.data, data);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector add(CooVector b) {
        return (CVector) RealFieldDenseCooVectorOps.add(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector sub(CooCVector b) {
        return (CVector) DenseCooFieldVectorOps.sub(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector sub(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }



    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector sub(CooVector b) {
        return (CVector) RealFieldDenseCooVectorOps.sub(this, b);
    }


    /**
     * Computes the element-wise multiplication between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCVector elemMult(CooCVector b) {
        return (CooCVector) DenseCooFieldVectorOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise multiplication between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CVector elemMult(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemMult(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Computes the element-wise multiplication between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCVector elemMult(CooVector b) {
        Complex128[] dest = RealFieldDenseCooVectorOps.elemMult(this, b);
        return CooCVector.unsafeMake(shape, dest, b.indices.clone());
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    public CVector div(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemDiv(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public CVector normalize() {
        return div(magAsDouble());
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public Complex128 mag() {
        return new Complex128(magAsDouble());
    }


    /**
     * Computes the magnitude of this vector as a double value.
     * @return The magnitude of this vector as a double value.
     */
    public double magAsDouble() {
        double mag = 0;

        for(int i = 0, size = data.length; i < size; i++) {
            Complex128 v =  data[i];
            mag += (v.re*v.re + v.im*v.im);
        }

        return Math.sqrt(mag);
    }


    /**
     * Constructs a sparse COO tensor, which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooCVector makeLikeCooTensor(Shape shape, Complex128[] entries, int[][] indices) {
        // Check for the case when the vector contains no non-zero data.
        return (indices.length == 0)
                ? CooCVector.unsafeMake(shape, entries, new int[0])
                : CooCVector.unsafeMake(shape, entries, indices[0]);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCVector toCoo() {
        return (CooCVector) super.toCoo();
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     *
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern, it is better to underestimate the sparsity.
     *
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCVector toCoo(double estimatedSparsity) {
        return (CooCVector) super.toCoo(estimatedSparsity);
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
    public CVector makeLikeTensor(Shape shape, Complex128[] entries) {
        ValidateParameters.ensureRank(shape, 1);
        ValidateParameters.ensureAllEqual(shape.totalEntriesIntValueExact(), entries.length);
        return new CVector(entries);
    }


    /**
     * Converts this complex vector to a real vector. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real vector containing the real components of all elements in this vector.
     */
    public Vector toReal() {
        return new Vector(shape, Complex128Ops.toReal(data));
    }


    /**
     * Checks if all data of this vector are real.
     * @return {@code true} if all data of this tensor are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this vector has a non-zero imaginary part.
     * @return {@code true} if any entry of this vector has a non-zero imaginary part.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Rounds all data within this vector to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new vector containing the data of this vector rounded to the specified precision.
     */
    public CVector round(int precision) {
        return new CVector(Complex128Ops.round(data, precision));
    }


    /**
     * Sets all elements of this vector to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this vector with all data within {@code tol} of zero set to zero.
     */
    public CVector roundToZero(double tolerance) {
        return new CVector(Complex128Ops.roundToZero(data, tolerance));
    }


    /**
     * Compute the inner product of this vector with itself.
     * @return
     */
    public double innerSelf() {
        return DenseFieldVectorOps.innerSelfProduct(data);
    }


    /**
     * <p>Computes the vector cross-product between two vectors.
     *
     * <p><b>Note</b>: Formally speaking, the vector cross-product cannot be
     * defined for complex vectors while maintaining the same properties as vectors from &#8477<sup>3</sup>
     * (e.g., orthogonality).
     * However, it can still be defined algebraically in the same way as it is for real vectors,
     * though it may fail to satisfy the properties of a true cross-product.
     * This is algebraic definition is what this method implements.
     *
     * @param b Second vector in the cross-product.
     *
     * @return The result of the vector cross-product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If either this vector or {@code b} do not have exactly 3 data.
     */
    public CVector cross(CVector b) {
        if(size != 3 || b.size != 3) {
            throw new LinearAlgebraException("Cross products can only be called vectors of size 3 but got sizes "
                    + size + " and " + b.size);
        }

        Complex128[] dest = new Complex128[3];
        Complex128 v0 = data[0]; Complex128 v1 = data[1]; Complex128 v2 = data[2];
        Complex128 bv0 = b.data[0]; Complex128 bv1 = b.data[1]; Complex128 bv2 = b.data[2];

        return new CVector(v1.mult(bv2).sub(v2.mult(bv1)),
                v2.mult(bv0).sub(v0.mult(bv2)),
                v0.mult(bv1).sub(v1.mult(bv0)));
    }


    /**
     * Computes the element-wise difference between two vectors of the same shape.
     * @param b Second tensor in the element-wise difference.
     * @throws TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public void subEq(Vector b) {
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, data);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return {@code true} if the two vectors have the same shape, are numerically equivalent, and are of type {@link CVector}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CVector src2 = (CVector) object;

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of data to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get data up until the stopping point.
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centering = PrintOptions.useCentering();

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

        result.append("]");

        return result.toString();
    }
}
