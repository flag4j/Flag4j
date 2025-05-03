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
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldTensor;
import org.flag4j.arrays.backend.ring_arrays.TensorOverRing;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.io.parsing.ComplexNumberParser;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldTensorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Complex64;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;


/**
 * <p>A dense complex tensor backed by an array of {@link Complex128}s.
 *
 * <p>The {@link #data} of a tensor are mutable but the {@link #shape} is fixed.
 */
public class CTensor extends AbstractDenseFieldTensor<CTensor, Complex128> {
    private static final long serialVersionUID = 1L;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public CTensor(Shape shape, Complex128[] data) {
        super(shape, data);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor from an nD array. The tensors shape will be inferred from.
     * @param nDArray Array to construct tensor from. Must be a rectangular array.
     * @throws IllegalArgumentException If {@code nDArray} is not an array or not rectangular.
     */
    public CTensor(Object nDArray) {
        super(ArrayUtils.nDArrayShape(nDArray),
                new Complex128[ArrayUtils.nDArrayShape(nDArray).totalEntriesIntValueExact()]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);
        ArrayUtils.nDFlatten(nDArray, shape, data, 0);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public CTensor(Shape shape, Complex64[] data) {
        super(shape, new Complex128[data.length]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);

        for(int i=0, size=data.length; i<size; i++)
            this.data[i] = new Complex128(data[i]);
    }


    /**
     * Creates a zero tensor with the specified dimensions.
     *
     * @param dims The dimension of each axis in the tensor. The returned tensor will have shape equivalent to {@code new Shape(dims)}.
     */
    public CTensor(int... dims) {
        this(new Shape(dims));
    }


    /**
     * Creates a zero tensor with the specified shape.
     *
     * @param shape Shape of this tensor.
     */
    public CTensor(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, Complex128 fillValue) {
        this(shape);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, Complex64 fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with. Must be a string representation of a complex number parsable by 
     * {@link ComplexNumberParser#parseNumberToComplex128(String)}.
     */
    public CTensor(Shape shape, String fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. Each value in {@code data} must be formated as a complex number such as:
     * <ul>
     *     <li>"a"</li>
     *     <li>"a + bi", "a - bi", "a + i", or "a - i"</li>
     *     <li>"bi", "i", or "-i"</li>
     * </ul>
     *
     * where "a" and "b" are integers or decimal numbers and white space does not matter.
     */
    public CTensor(Shape shape, String[] data) {
        super(shape, new Complex128[data.length]);
        if(data.length == 0 || data[0] == null) setZeroElement(Complex128.ZERO);

        for(int i=0, size=data.length; i<size; i++)
            this.data[i] = new Complex128(data[i]);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public CTensor(Shape shape, double[] data) {
        super(shape, ArrayConversions.toComplex128(data, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of the specified tensor.
     * @param tensor Tensor to create copy of.
     */
    public CTensor(CTensor tensor) {
        super(tensor.shape, (Complex128[]) tensor.data);
    }


    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
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
    public CTensor makeLikeTensor(Shape shape, Complex128[] data) {
        return new CTensor(shape, data);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param data Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooCTensor makeLikeCooTensor(Shape shape, Complex128[] data, int[][] indices) {
        return new CooCTensor(shape, data, indices);
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
    public CTensor add(CooCTensor b) {
        return (CTensor) DenseCooFieldTensorOps.add(this, b);
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
    public CTensor add(Tensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(shape, data, b.shape, b.data, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape and stores hte result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void addEq(Tensor b) {
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
    public CTensor add(CooTensor b) {
        return (CTensor) RealFieldDenseCooOps.add(this, b);
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
    public CTensor sub(CooCTensor b) {
        return (CTensor) DenseCooFieldTensorOps.sub(this, b);
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
    public CTensor sub(Tensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Computes the element-wise difference of this tensor with a real dense tensor and stores the result in this tensor.
     * @param b Second tensor in element-wise difference.
     */
    public void subEq(Tensor b) {
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, data);
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
    public CTensor sub(CooTensor b) {
        return (CTensor) RealFieldDenseCooOps.sub(this, b);
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
    public CooCTensor elemMult(CooCTensor b) {
        return (CooCTensor) DenseCooFieldTensorOps.elemMult(this, b);
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
    public CTensor elemMult(Tensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemMult(shape, data, b.shape, b.data, dest);
        return new CTensor(shape, dest);
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
    public CooCTensor elemMult(CooTensor b) {
        Complex128[] dest = new Complex128[b.nnz];
        int[][] indices = new int[b.nnz][rank];
        RealFieldDenseCooOps.elemMult(this, b, dest, indices);
        return CooCTensor.unsafeMake(shape, dest, indices);
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    public CTensor div(Tensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemDiv(shape, data, b.shape, b.data, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCTensor toCoo() {
        return (CooCTensor) super.toCoo();
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     *
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern it is better to underestimate the sparsity.
     *
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCTensor toCoo(double estimatedSparsity) {
        return (CooCTensor) super.toCoo(estimatedSparsity);
    }


    /**
     * Converts this complex tensor to a real tensor. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real tensor containing the real components of the data of this tensor.
     */
    public Tensor toReal() {
        return new Tensor(shape, Complex128Ops.toReal(data));
    }


    /**
     * Checks if all data of this tensor are real.
     * @return {@code true} if all data of this tensor are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this tensor has non-zero imaginary component.
     * @return {@code true} if any entry of this tensor has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Rounds all data within this tensor to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new tensor containing the data of this tensor rounded to the specified precision.
     */
    public CTensor round(int precision) {
        return new CTensor(shape, Complex128Ops.round(data, precision));
    }


    /**
     * Sets all elements of this tensor to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this tensor with all data within {@code tol} of zero set to zero.
     */
    public CTensor roundToZero(double tolerance) {
        return new CTensor(shape, Complex128Ops.roundToZero(data, tolerance));
    }


    /**
     * Converts this tensor to an equivalent matrix. If this tensor is not rank-2, it will be flattened to a row vector before
     * conversion.
     * @return A matrix which is equivalent to this tensor.
     */
    public CMatrix toMatrix() {
        Shape matShape = (shape.getRank() == 2) ? shape : new Shape(1, data.length);
        return new CMatrix(matShape, data.clone());
    }


    /**
     * Converts this tensor to an equivalent vector. If the tensor is not rank-1, it will be flattened first.
     * @return A vector which is equivalent to this tensor.
     */
    public CVector toVector() {
        return new CVector(data.clone());
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public TensorOverRing abs() {
        double[] abs = new double[data.length];
        RingOps.abs(data, abs);
        return new Tensor(shape, abs);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CTensor src2 = (CTensor) object;

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
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flattened data of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = shape.totalEntries().intValueExact();
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));

        result.append(PrettyPrint.abbreviatedArray(data,
                PrintOptions.getMaxColumns(),
                PrintOptions.getPadding(),
                PrintOptions.getPrecision(),
                PrintOptions.useCentering()));

        return result.toString();
    }
}
