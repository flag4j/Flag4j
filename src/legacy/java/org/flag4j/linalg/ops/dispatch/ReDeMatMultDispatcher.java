/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.ops.dispatch;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.ops.dense.real.RealDenseMatMult;
import org.flag4j.linalg.ops.dispatch.configs.ReDeMatMultDispatchConfigs;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.function.BiFunction;

import static org.flag4j.linalg.ops.dispatch.ReDeMatMultKernels.*;

/**
 * <p>A dispatcher that selects the most suitable matrix multiplication kernel for two real dense matrices.
 *
 * <p>This class implements a threshold- and shape-based decision tree to optimize performance for various
 * matrix multiplication scenarios (e.g., small matrices, square matrices, wide/tall matrices, matrix-vector
 * products, etc.). It maintains a cache of recently used shape/kernels to further improve performance
 * for repeated multiplication patterns.
 *
 * <h2>Usage:</h2>
 * <ul>
 *   <li>Use {@link #dispatch(Matrix, Matrix)} to multiply two matrices, dynamically choosing the best kernel.</li>
 *   <li>This class is a singleton; call {@link #getInstance()} to retrieve the instance if you need direct
 *   access to the underlying dispatcher.</li>
 * </ul>
 *
 * <h2>Configuration:</h2>
 * The dispatcher reads various thresholds (e.g., {@code ASPECT_THRESH}, {@code SML_THRESH}, etc.)
 * from {@link ReDeMatMultDispatchConfigs}, allowing external tuning without modifying code.
 *
 * <h2>Thread Safety:</h2>
 * <ul>
 *   <li>The dispatcher itself does not maintain mutable state beyond a cached kernel lookup,
 *       which is also thread-safe.</li>
 *   <li>{@link #dispatch(Matrix, Matrix)} is safe to call from multiple threads.</li>
 * </ul>
 *
 * @see BiTensorOpDispatcher
 * @see ReDeMatMultDispatchConfigs
 */
public final class ReDeMatMultDispatcher extends BiTensorOpDispatcher<Matrix, Matrix, Matrix> {

    /**
     * Threshold for considering a matrix "near-square". If the quotient of the maximum and minimum dimension of either matrix
     * is less than this value, the matrix will be considered "near-square".
     */
    private static final double ASPECT_THRESH = ReDeMatMultDispatchConfigs.getAspectThreshold();

    /**
     * Threshold for the total number of entries in both matrices to consider the problem "small enough" to
     * default to the standard algorithm.
     */
    private static final int SML_THRESH = ReDeMatMultDispatchConfigs.getSmallThreshold();

    /**
     * Threshold for square matrices to use a standard concurrent kernel.
     */
    private static final int SQ_MT_STND_THRESH = ReDeMatMultDispatchConfigs.getSquareMtStandardThreshold();
    /**
     * Threshold for square matrices to use a reordered concurrent kernel.
     */
    private static final int SQ_MT_REORD_THRESH = ReDeMatMultDispatchConfigs.getSquareMtReorderedThreshold();

    /**
     * Threshold for non-square wide matrices. i.e., {@code m = max(m, n, k)} and {@code max(m, n, k) / min(m, n, k) >  aspectThreshold}.
     */
    private static final int WIDE_MT_REORD_THRESH = ReDeMatMultDispatchConfigs.getWideMtReorderedThreshold();
    /**
     * Threshold for using standard matrix-vector kernel when matrix is "near square".
     */
    private static final int SQ_SEQ_VEC_THRESH = ReDeMatMultDispatchConfigs.getSquareSequentialVecThreshold();

    /**
     * Threshold for considering the minimum dimension small enough to fall back to sequential kernel.
     */
    private static final int MIN_DIM_SML_THRESH = ReDeMatMultDispatchConfigs.getMinDimSmallThreshold();

    /**
     * The number of shape pairs to cache so that the kernel need not be recomputed if the shape pair has been seen recently.
     */
    private static final int CACHE_SIZE = ReDeMatMultDispatchConfigs.getCacheSize();


    /**
     * Creates a matrix multiplication dispatcher with the specified {@code cacheSize}.
     *
     * @param cacheSize The size of the cache for this matrix multiplication dispatcher.
     */
    private ReDeMatMultDispatcher(int cacheSize) {
        super(cacheSize);
    }


    /**
     * Validates the shapes are valid for the operation.
     *
     * @param aShape Shape of first tensor in the operation.
     * @param bShape Shape of second tensor in the operation.
     */
    @Override
    protected void validateShapes(Shape aShape, Shape bShape) {
        ValidateParameters.ensureMatMultShapes(aShape, bShape);
    }


    /**
     * Simple Holder class for lazy loading of the singleton instance.
     */
    private static class Holder {
        private static final ReDeMatMultDispatcher INSTANCE =
                new ReDeMatMultDispatcher(CACHE_SIZE);
    }


    /**
     * Gets the singleton instance of this class. If this class has not been instanced, a new instance will be created.
     * @return The singleton instance of this class.
     */
    public static ReDeMatMultDispatcher getInstance() {
        return Holder.INSTANCE;
    }


    /**
     * Dispatches the multiplication of two matrices to an appropriate implementation based on the shapes of the two matrices.
     * @param a Left matrix in the matrix multiplication.
     * @param b Right matrix in the matrix multiplication.
     * @return The matrix product of {@code a} and {@code b}.
     */
    public static Matrix dispatch(Matrix a, Matrix b) {
        // Use a long to protect against possible overflow.
        long totalEntries = (long) a.dataLength() + b.dataLength();

        // Return as fast as possible for "small-enough" matrices.
        if(totalEntries < SML_THRESH) {
            if(a.numCols != b.numRows)
                throw new LinearAlgebraException(ErrorMessages.matMultShapeErrMsg(a.shape, b.shape));

            return new Matrix(new Shape(a.numRows, b.numCols),
                    RealDenseMatMult.standard(a.data, a.shape, b.data, b.shape));
        }

        return Holder.INSTANCE.dispatch_(a, b);
    }


    /**
     * Computes the appropriate function to use when computing the matrix multiplication between two matrices.
     *
     * @param aShape Shape of the first matrix in the matrix multiplication problem.
     * @param bShape Shape of the second matrix in the matrix multiplication problem.
     * @param data1Length Full length of the items array within the first matrix.
     * @param data2Length Full length of the items array within the second matrix.
     * @return The appropriate function to use when computing the matrix multiplication between two matrices.
     */
    @Override
    protected BiFunction<Matrix, Matrix, Matrix> getFunc(Shape aShape, Shape bShape, int data1Length, int data2Length) {
        int m = aShape.getSize(0);
        int n = aShape.getSize(1);
        int k = bShape.getSize(1);

        if(k == 1) {
            // Then we have a matrix-vector product.
            double aspectRatio = (double) Math.max(m, n) / Math.min(m, n);

            if(aspectRatio <= 4*ASPECT_THRESH) {
                // The matrix is approximately square.
                if(m < SQ_SEQ_VEC_THRESH)
                    return BLK_VEC_AS_MAT;
                else
                    return MT_BLK_VEC_AS_MAT;
            } else {
                return BLK_VEC_AS_MAT;
            }
        } else {
            int minDim = Math.min(k, Math.min(m, n));
            int maxDim = Math.max(k, Math.max(m, n));

            double aspectRatio = (double) maxDim / minDim;

            if(aspectRatio <= ASPECT_THRESH) {
                // Matrices are very roughly square.
                if(n < SQ_MT_STND_THRESH)
                    return MT_STRD;
                else if(n < SQ_MT_REORD_THRESH)
                    return MT_REORD;
                else
                    return MT_BLK_REORD;
            } else {
                if (m == maxDim) {
                    if (minDim < MIN_DIM_SML_THRESH)
                        return STRD;
                    else
                        return MT_REORD;
                } else if(n == maxDim) {
                    if (minDim < MIN_DIM_SML_THRESH)
                        return STRD;
                    else if(m < WIDE_MT_REORD_THRESH)
                        return MT_STRD;
                    else
                        return MT_REORD;
                } else { // Then k == maxDim.
                    if (minDim < MIN_DIM_SML_THRESH)
                        return REORD;
                    else
                        return MT_REORD;
                }
            }
        }
    }
}
