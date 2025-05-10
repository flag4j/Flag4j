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
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMult;
import org.flag4j.linalg.ops.dispatch.configs.Cm128DeMatMultDispatchConfigs;
import org.flag4j.linalg.ops.dispatch.configs.ReDeMatMultDispatchConfigs;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.function.BiFunction;

import static org.flag4j.linalg.ops.dispatch.Cm128DeMatMultKernels.BLK_VEC;
import static org.flag4j.linalg.ops.dispatch.Cm128DeMatMultKernels.MT_BLK_VEC;


/**
 * <p>A dispatcher that selects the most suitable matrix-vector multiplication kernel for a complex matrix/vector pair.
 *
 * <p>This class implements a threshold- and shape-based decision tree to optimize performance for various
 * matrix-vector multiplication scenarios (e.g., small matrices, square matrices, wide/tall matrices, etc.).
 * It also maintains a cache of recently used shape/kernels to further improve performance
 * for repeated multiplication patterns.
 *
 * <h2>Usage:</h2>
 * <ul>
 *   <li>Use {@link #dispatch(CMatrix, CVector)} to compute the matrix-vector product, dynamically choosing the best kernel.</li>
 *   <li>This class is a singleton; call {@link #getInstance()} to retrieve the instance if you need direct
 *   access to the underlying dispatcher.</li>
 * </ul>
 *
 * <h2>Configuration:</h2>
 * The dispatcher reads various thresholds (e.g., {@code ASPECT_THRESH}, {@code SML_THRESH}, etc.)
 * from {@link Cm128DeMatMultDispatchConfigs}, allowing external tuning without modifying code.
 *
 * <h2>Thread Safety:</h2>
 * <ul>
 *   <li>The dispatcher itself does not maintain mutable state beyond a cached kernel lookup,
 *       which is also thread-safe.</li>
 *   <li>{@link #dispatch(CMatrix, CVector)} is safe to call from multiple threads.</li>
 * </ul>
 *
 * @see BiTensorOpDispatcher
 * @see ReDeMatMultDispatchConfigs
 */
public class Cm128DeMatVecMultDispatcher extends BiTensorOpDispatcher<CMatrix, CVector, CVector> {

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
     * Threshold for using standard matrix-vector kernel when matrix is "near square".
     */
    private static final int SQ_SEQ_VEC_THRESH = ReDeMatMultDispatchConfigs.getSquareSequentialVecThreshold();

    /**
     * The number of shape pairs to cache so that the kernel need not be recomputed if the shape pair has been seen recently.
     */
    private static final int CACHE_SIZE = ReDeMatMultDispatchConfigs.getCacheSize();


    /**
     * Creates a matrix multiplication dispatcher with the specified {@code cacheSize}.
     *
     * @param cacheSize The size of the cache for this matrix multiplication dispatcher.
     */
    protected Cm128DeMatVecMultDispatcher(int cacheSize) {
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
        private static final Cm128DeMatVecMultDispatcher INSTANCE =
                new Cm128DeMatVecMultDispatcher(CACHE_SIZE);
    }


    /**
     * Gets the singleton instance of this class. If this class has not been instanced, a new instance will be created.
     * @return The singleton instance of this class.
     */
    public static Cm128DeMatVecMultDispatcher getInstance() {
        return Cm128DeMatVecMultDispatcher.Holder.INSTANCE;
    }


    /**
     * Dispatches the multiplication of two matrices to an appropriate implementation based on the shapes of the two matrices.
     * @param a Left matrix in the matrix multiplication.
     * @param b Right matrix in the matrix multiplication.
     * @return The matrix product of {@code a} and {@code b}.
     */
    public static CVector dispatch(CMatrix a, CVector b) {
        long totalEntries = (long) a.dataLength() + b.dataLength();

        // Return as fast as possible for "small-enough" matrices.
        if(totalEntries < SML_THRESH) {
            if(a.numCols != b.size)
                throw new LinearAlgebraException(ErrorMessages.matMultShapeErrMsg(a.shape, b.shape));

            Complex128[] dest = new Complex128[a.numRows];
            DenseSemiringMatMult.standardVector(a.data, a.shape, b.data, b.shape, dest);
            return new CVector(dest);
        }

        return Cm128DeMatVecMultDispatcher.Holder.INSTANCE.dispatch_(a, b);
    }


    /**
     * Computes the appropriate function to use when computing the matrix multiplication between two matrices.
     *
     * @param aShape Shape of the first matrix in the matrix-vector multiplication problem.
     * @param bShape Shape of the vector in the matrix-vector multiplication problem.
     * @param data1Length Full length of the data array within the first matrix.
     * @param data2Length Full length of the data array within the vector.
     *
     * @return The appropriate function to use when computing the matrix multiplication between two matrices.
     */
    @Override
    protected BiFunction<CMatrix, CVector, CVector> getFunc(Shape aShape, Shape bShape, int data1Length, int data2Length) {
        int m = aShape.getSize(0);
        int n = aShape.getSize(1);
        double aspectRatio = (double) Math.max(m, n) / Math.min(m, n);

        if(aspectRatio <= 4*ASPECT_THRESH) {
            // The matrix is approximately square.
            if(m < SQ_SEQ_VEC_THRESH) return BLK_VEC;
            else return MT_BLK_VEC;
        } else {
            return BLK_VEC;
        }
    }
}
