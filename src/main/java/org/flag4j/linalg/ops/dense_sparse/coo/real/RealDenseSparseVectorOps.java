/*
 * MIT License
 *
 * Copyright (c) 2023-2026. Jacob Watters
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

package org.flag4j.linalg.ops.dense_sparse.coo.real;

import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

/**
 * This class provides low level methods for computing ops between a real dense/sparse vector and a
 * real sparse/dense vector.
 */
public class RealDenseSparseVectorOps {


    private RealDenseSparseVectorOps() {
        // Hide default constructor in utility class.
    }


    /**
     * Computes the vector inner product between a real dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero data).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of data in the two vectors is not equivalent.
     */
    public static double inner(double[] src1, double[] src2, int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        double innerProd = 0;
        int index;

        for(int i=0; i<src2.length; i++) {
            index = indices[i];
            innerProd += src1[index]*src2[i];
        }

        return innerProd;
    }


    /**
     * Computes the vector outer product between a real dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero data of the sparse vector.
     * @param indices Indices of non-zero data of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero data.
     * @return The matrix resulting from the vector outer product.
     */
    public static double[] outerProduct(double[] src1, double[] src2, int[] indices, int sparseSize) {
        double[] dest = new double[src1.length*sparseSize];
        int index;

        for(int i=0; i<src1.length; i++) {
            for(int j=0; j<src2.length; j++) {
                index = indices[j];

                dest[i*src1.length + index] = src1[i]*src2[j];
            }
        }

        return dest;
    }


    /**
     * Computes the vector outer product between a real dense vector and a real sparse vector.
     * @param src1 Non-zero data of the sparse vector.
     * @param src2 Entries of the dense vector.
     * @param indices Indices of non-zero data of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero data.
     * @return The matrix resulting from the vector outer product.
     */
    public static double[] outerProduct(double[] src1, int[] indices, int sparseSize, double[] src2) {
        ValidateParameters.ensureAllEqual(sparseSize, src2.length);

        double[] dest = new double[src2.length*sparseSize];
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = indices[i]*src2.length;

            for(double v : src2) {
                dest[destIndex++] = src1[i]*v;
            }
        }

        return dest;
    }


    /**
     * Subtracts a real sparse vector from a real dense vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static Vector sub(Vector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Vector dest = new Vector(src1);

        for(int i=0; i<src2.nnz; i++) {
            dest.data[src2.indices[i]] -= src2.data[i];
        }

        return dest;
    }


    /**
     * Subtracts a real dense vector from a real sparse vector.
     * @param src1 Sparse vector.
     * @param src2 Dense vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static Vector sub(CooVector src1, Vector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Vector dest = new Vector(RealOps.scalMult(src2.data, -1, null));

        for(int i=0; i<src1.nnz; i++)
            dest.data[src1.indices[i]] += src1.data[i];

        return dest;
    }


    /**
     * Adds a real dense vector to a real sparse vector and stores the result in the first vector.
     * @param src1 Dense vector. Also, where the result will be stored.
     * @param src2 Sparse vector.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static void addEq(Vector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            src1.data[src2.indices[i]] += src2.data[i];
        }
    }


    /**
     * Adds a real dense vector to a real sparse vector and stores the result in the first vector.
     * @param src1 Dense vector. Also, where the result will be stored.
     * @param src2 Sparse vector.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static void subEq(Vector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            src1.data[src2.indices[i]] -= src2.data[i];
        }
    }


    /**
     * Computes the element-wise multiplication of a real dense vector with a real sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws NDArrayShapeException If the two vectors are not the same size.
     */
    public static CooVector elemMult(Vector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        double[] entries = new double[src2.data.length];

        for(int i=0; i<src2.nnz; i++)
            entries[i] = src1.data[src2.indices[i]]*src2.data[i];

        return CooVector.unsafeMake(src1.size, entries, src2.indices.clone());
    }


    /**
     * Adds a real dense vector to a real sparse vector.
     * @return The result of the vector addition.
     * @param src1 Entries of first vector in the sum.
     * @param src2 Entries of second vector in the sum.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static Vector add(Vector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Vector dest = new Vector(src1);

        for(int i=0; i<src2.nnz; i++) {
            dest.data[src2.indices[i]] += src2.data[i];
        }

        return dest;
    }


    /**
     * Compute the element-wise division between a sparse vector and a dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static CooVector elemDiv(CooVector src1, Vector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        double[] dest = new double[src1.data.length];

        for(int i = 0; i<src1.data.length; i++) {
            dest[i] = src1.data[i]/src2.data[src1.indices[i]];
        }

        return CooVector.unsafeMake(src1.size, dest, src1.indices.clone());
    }
}
