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
import org.flag4j.arrays.backend.primitive_arrays.AbstractDenseDoubleTensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.dense.real.RealDenseEquals;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseCooTensorOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseCooOps;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A real dense tensor backed by a primitive double array.
 *
 * <p>A tensor is a multidimensional array. If N indices are required to uniquely identify all elements of a tensor, then the
 * tensor is considered an N-dimensional tensor/array or a rank-N tensor.
 *
 * <p>The {@link #data} of a Tensor are mutable but the {@link #shape} is fixed.
 */
public class Tensor extends AbstractDenseDoubleTensor<Tensor> {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a zero tensor with the specified dimensions.
     *
     * @param dims The dimension of each axis in the tensor. The returned tensor will have shape equivalent to {@code new Shape(dims)}.
     */
    public Tensor(int... dims) {
        this(new Shape(dims));
    }



    /**
     * Creates a zero tensor with the shape.
     *
     * @param shape Shape of this tensor.
     */
    public Tensor(Shape shape) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
    }


    /**
     * Creates a tensor from an nD array. The tensor's shape will be inferred from.
     * @param nDArray Array to construct tensor from. Must be a rectangular array.
     * @throws IllegalArgumentException If {@code nDArray} is not an array or not rectangular.
     */
    public Tensor(Object nDArray) {
        this(ArrayUtils.nDArrayShape(nDArray));
        ArrayUtils.nDFlatten(nDArray, shape, data, 0);
    }


    /**
     * Creates a tensor with the specified shape filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public Tensor(Shape shape, double fillValue) {
        this(shape);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public Tensor(Shape shape, double... data) {
        super(shape, data);
    }


    /**
     * Flattens tensor to a single dimension while preserving the order of data.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public Tensor flatten() {
        return new Tensor(shape.flatten(), data.clone());
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
    public Tensor flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = new int[rank];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntriesIntValueExact();
        return new Tensor(new Shape(dims), data.clone());
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public Tensor(Shape shape, int... data) {
        super(shape, new double[data.length]);
        ArrayConversions.asDouble(data, this.data);
    }


    /**
     * Constructs a copy of the specified tensor.
     * @param src The tensor to make a copy of.
     */
    public Tensor(Tensor src) {
        super(src.shape, src.data.clone());
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public Tensor(Shape shape, Double[] data) {
        super(shape, new double[data.length]);
        ArrayConversions.unbox(data, super.data);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor.
     */
    public Tensor(Shape shape, Integer[] data) {
        super(shape, new double[data.length]);
        ArrayConversions.asDouble(data, super.data);
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
    public Tensor makeLikeTensor(Shape shape, double[] data) {
        return new Tensor(shape, data);
    }


    /**
     * Converts this tensor to an equivalent vector. If this vector is not rank-1, it will first be flattened,
     * then converted to a vector.
     * @return A vector with data equivalent to this vector.
     */
    public Vector toVector() {
        return new Vector(data.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix. If this matrix is not rank-2, it will first be flattened
     * to a row vector, then converted to a matrix.
     * @return A matrix with data equivalent to this tensor.
     */
    public Matrix toMatrix() {
        if(rank == 2) return new Matrix(shape, data.clone());
        else return new Matrix(new Shape(1, data.length), data.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix with the specified shape.
     * @param shape New shape for the matrix. Must be rank-2 and have the same number of total entries as {@code this.shape}.
     * @return A matrix with the specified shape and data equivalent to this tensor.
     * @throws IllegalArgumentException If {@code shape} does not have the same total number of entries {@code this.shape}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code shape.getRank() != 2}.
     */
    public Matrix toMatrix(Shape shape) {
        // Matrix constructor checks the rank of the shape and
        // ensures that shape.totalEntriesIntValueExact() == data.length.
        return new Matrix(shape, data.clone());
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public CooTensor toCoo() {
        return toCoo(0.9);
    }


    /**
     * Adds a complex-valued scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero data of the tensor.
     *
     * @param b Scalar value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    public CTensor add(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(data, b, dest);
        return new CTensor(shape, dest);
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
    public Tensor add(CooTensor b) {
        return RealDenseCooTensorOps.add(this, b);
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
    public CTensor add(CTensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(b.shape, b.data, shape, data, dest);
        return new CTensor(shape, dest);
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
        Complex128[] dest = new Complex128[data.length];
        RealComplexDenseCooOps.add(shape, data, b.shape, b.data, b.indices, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Subtracts a complex-valued scalar from each element of this tensor.
     * @param b Scalar value in vector-scalar difference.
     * @return The tensor resulting from subtracting {@code b} from each entry of this tensor.
     */
    public CTensor sub(Complex128 b) {
        Complex128[] diff = new Complex128[data.length];
        RealFieldDenseOps.sub(data, b, diff);
        return new CTensor(shape, diff);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CTensor sub(CooCTensor b) {
        return RealComplexDenseCooOps.sub(this, b);
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
    public Tensor sub(CooTensor b) {
        return RealDenseCooTensorOps.sub(this, b);
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
    public CTensor sub(CTensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param b Second tensor in the element-wise multiplication.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooTensor elemMult(CooTensor b) {
        return RealDenseCooTensorOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise multiplication of two tensors.
     * @param b Second tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     * @throws IllegalArgumentException If {@code !this.shape.equals(b.shape)}
     */
    public CTensor elemMult(CTensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemMult(b.shape, b.data, shape, data, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param b Second tensor in the element-wise multiplication.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooCTensor elemMult(CooCTensor b) {
        return RealComplexDenseCooOps.elemMult(this, b);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern, it is better to underestimate the sparsity.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public CooTensor toCoo(double estimatedSparsity) {
        ValidateParameters.ensureInRange(estimatedSparsity, 0.0, 1.0, "estimatedSparsity");
        int estimatedSize = (int) (data.length*(1.0-estimatedSparsity));
        List<Double> cooEntries = new ArrayList<>(estimatedSize);
        List<int[]> cooIndices = new ArrayList<>(estimatedSize);
        final Double ZERO = Double.valueOf(0d);

        final int rows = shape.get(0);
        final int cols = shape.get(1);

        for(int i = 0, size = data.length; i<size; i++) {
            Double val = data[i];

            if(!val.equals(ZERO)) {
                cooEntries.add(val);
                cooIndices.add(shape.getNdIndices(i));
            }
        }

        return new CooTensor(shape, cooEntries, cooIndices);
    }


    /**
     * Converts this tensor to an equivalent complex-valued tensor.
     * @return A complex tensor whose real components are the same as the data of
     * this tensor and the imaginary components are zero.
     */
    public CTensor toComplex() {
        return new CTensor(shape, ArrayConversions.toComplex128(data, null));
    }


    /**
     * Multiplies a complex scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    public CTensor mult(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.mult(data, b, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Divides each entry of this tensor by a complex scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The tensor-scalar quotient of this tensor with {@code b}.
     */
    public CTensor div(Complex128 b) {
        return new CTensor(shape, Complex128Ops.scalDiv(data, b));
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param b The second tensor in the element-wise quotient.
     * @return The element-wise quotient of this tensor with {@code b}.
     * @throws IllegalArgumentException If {@code !this.shape.equals(b.shape)}
     */
    public CTensor div(CTensor b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemDiv(shape, data, b.shape, b.data, dest);
        return new CTensor(shape, dest);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link Tensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        Tensor src2 = (Tensor) object;

        return RealDenseEquals.tensorEquals(this.data, this.shape, src2.data, src2.shape);
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
