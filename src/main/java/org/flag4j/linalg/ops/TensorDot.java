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

package org.flag4j.linalg.ops;

import org.flag4j.arrays.Shape;
import org.flag4j.util.ArrayJoiner;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

/**
 * The base class for all classes whose instances may be used to compute a tensor dot product.
 * @param <T> Type of the storage for the data of the tensor.rf
 */
public abstract class TensorDot<T> {

    protected Shape shape1;
    protected Shape shape2;
    protected T src1;
    protected T src2;
    protected int[] src1Axes;
    protected int[] src2Axes;

    protected Shape newShape1;
    protected Shape newShape2;
    protected Shape destShape;
    protected int destLength;
    protected int[] src1NewAxes;
    protected int[] src2NewAxes;
    protected int[] src1Dims;
    protected int[] src2Dims;

    /**
     * Constructs a tensor dot product problem for computing the tensor contraction of two tensors over the
     * specified set of axes. That is, computes the sum of products between the two tensors along the specified set of axes.
     * @param shape1 Shape of the first tensor in the contraction.
     * @param src1 Entries/Non-zero data of the first tensor in the contraction.
     * @param shape2 Shape of the second tensor in the contraction.
     * @param src2 Entries/Non-zero data of the second tensor in the contraction.
     * @param src1Axes Axes along which to compute products for {@code src1} tensor.
     * @param src2Axes Axes along which to compute products for {@code src2} tensor.
     * @throws IllegalArgumentException If {@code src1Axes} and {@code src2Axes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor. Or, If the two tensor's shapes do not match along the specified axes pairwise
     * in {@code src1Axes} and {@code src2Axes}.
     */
    protected TensorDot(Shape shape1, T src1,
                        Shape shape2, T src2,
                        int[] src1Axes, int[] src2Axes) {
        // Each array must specify the same number of axes.
        ValidateParameters.ensureAllEqual(src1Axes.length, src2Axes.length);

        // Axis values must be less than the rank of the tensor and non-negative.
        ValidateParameters.ensureValidAxes(shape1, src1Axes);
        ValidateParameters.ensureValidAxes(shape2, src2Axes);

        this.shape1 = shape1;
        this.src1 = src1;
        this.shape2 = shape2;
        this.src2 = src2;
        this.src1Axes = src1Axes;
        this.src2Axes = src2Axes;

        computeShapes();
    }


    /**
     * Computes the shape of the tensor resulting from this tensor dot product.
     * @return The shape of the tensor resulting from this tensor dot product.
     */
    private void computeShapes() {
        int[] notin;
        int n1;
        int n2;
        int pos;

        // ---- Compute new axes and shapes for first tensor. ----
        notin = ArrayUtils.notInAxes(src1Axes, shape1.getRank());
        src1NewAxes = ArrayJoiner.join(notin, src1Axes);

        n2 = 1;
        for(int axis : src1Axes)
            n2 *= shape1.getSize(axis);

        n1 = 1;
        src1Dims = new int[notin.length];
        pos = 0;
        for(int axis : notin) {
            int a = shape1.getSize(axis);
            n1 *= a;
            src1Dims[pos++] = a;
        }

        newShape1 = new Shape(n1, n2);
        // -----------------------------------------------------

        // ---- Compute new axes and shapes for second tensor. ----
        notin = ArrayUtils.notInAxes(src2Axes, shape2.getRank());
        src2NewAxes = ArrayJoiner.join(src2Axes, notin);

        n2 = 1;
        for(int axis : src2Axes)
            n2 *= shape2.getSize(axis);

        n1 = 1;
        pos = 0;
        src2Dims = new int[notin.length];
        for(int axis : notin) {
            int a = shape2.getSize(axis);
            n1 *= a;
            src2Dims[pos++] = a;
        }

        newShape2 = new Shape(n2, n1);
        // -----------------------------------------------------

        destShape = new Shape(ArrayJoiner.join(src1Dims, src2Dims));
        destLength = destShape.numelIntValueExact();
    }


    /**
     * Gets the shape of the tensor resulting from this tensor dot product as specified in the constructor.
     * @return The shape of the tensor resulting from this tensor dot product as specified in the constructor.
     */
    public Shape getOutputShape() {
        return destShape;
    }


    /**
     * Gets the total number of data in the tensor resulting from this tensor dot product as specified in the constructor.
     * @return The total number of data in the tensor resulting from this tensor dot product as specified in the constructor.
     */
    public int getOutputSize() {
        return destLength;
    }


    /**
     * Computes this tensor dot product as specified in the constructor.
     * @param dest The array to store the data of the dense tensor resulting from this tensor dot product. The size of this array
     * should be computed using {@link #getOutputSize()}.
     */
    public abstract void compute(T dest);
}
