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

package org.flag4j.linalg.decompositions.unitary;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.util.Flag4jConstants;

/**
 * <p>The base class for real matrix decompositions which proceed by using orthogonal transformations
 * (specifically Householder reflectors) to bring a matrix into an upper triangular/Hessenburg matrix. Specifically, the QR and
 * Hessenburg decompositions.
 *
 * <p>This class is provided because both the QR and Hessenburg decompositions proceed by very similar computations resulting in a
 * substantial amount of overlap in the implementations of the two decompositions. This class serves to implement these common
 * computations such that implementations of either decomposition may utilize them without the need of reimplementing them.
 */
public abstract class RealUnitaryDecomposition extends UnitaryDecomposition<Matrix, double[]> {

    /**
     * To store norms of columns in {@link #transformMatrix}.
     */
    protected double norm;
    /**
     * Scalar factor of the currently computed Householder reflector.
     */
    protected double currentFactor;
    /**
     * Stores the shifted value of the first entry in a Householder vector.
     */
    protected double shift;


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper triangular/Hessenburg matrix which is has zeros below
     * the specified sub-diagonal.
     *
     * @param subDiagonal Sub-diagonal of the upper triangular/Hessenburg matrix. That is, the sub-diagonal for which all items
     * below will be zero in the final upper quasi-triangular matrix. Must be Zero or one.
     * <ul>
     *     <li>{@code subDiagonal = 0}: Matrix will be upper triangular.</li>
     *     <li>{@code subDiagonal = 1}: Matrix will be upper Hessenburg.</li>
     * </ul>
     * @throws IllegalArgumentException If {@code subDiagonal < 0 || subDiagonal > 1}.
     */
    public RealUnitaryDecomposition(int subDiagonal) {
        super(subDiagonal, true, false);
    }


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper triangular/Hessenburg matrix which has zeros below
     * the specified sub-diagonal (must be 0 or 1).
     *
     * <p>Allows for specification if the reflectors used to bring matrix to upper triangular/Hessenburg form are to be stored or
     * not.
     *
     * <p>If the {@code Q} matrix is needed, then {@code storeReflectors} must be {@code true}. If {@code Q} is <b>NOT</b> needed, then
     * not storing the reflectors <em>may</em> improve performance slightly by avoiding unneeded copies.
     *
     * <p>It should be noted that if performance is improved, it will be a very slight improvement compared
     * to the total time to compute the decomposition. This is because the computation of {@code Q} is only
     * evaluated lazily once {@link #getQ()} is called, so this will only save on copy ops.
     *
     * @param subDiagonal Sub-diagonal of the upper triangular/Hessenburg matrix. That is, the sub-diagonal for which all items
     * below will be zero in the final upper quasi-triangular matrix. Must be Zero or one.
     * <ul>
     *     <li>{@code subDiagonal = 0}: Matrix will be upper triangular.</li>
     *     <li>{@code subDiagonal = 1}: Matrix will be upper Hessenburg.</li>
     * </ul>
     * @param storeReflectors Flag indicating if the reflectors used to bring the matrix to upper triangular/Hessenburg form
     * should be stored.
     * @throws IllegalArgumentException If {@code subDiagonal < 0 || subDiagonal > 1}.
     */
    public RealUnitaryDecomposition(int subDiagonal, boolean storeReflectors) {
        super(subDiagonal, storeReflectors, false);
    }


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper triangular/Hessenburg matrix which has zeros below
     * the specified sub-diagonal (must be 0 or 1).
     *
     * <p>Allows for specification if the reflectors used to bring matrix to upper triangular/Hessenburg form are to be stored or
     * not.
     *
     * <p>If the {@code Q} matrix is needed, then {@code storeReflectors} must be true. If {@code Q} is <b>NOT</b> needed, then
     * not storing the reflectors <i>may</i> improve performance slightly by avoiding unneeded copies.
     *
     * <p>It should be noted that if performance is improved, it will be a very slight improvement compared
     * to the total time to compute the decomposition. This is because the computation of {@code Q} is only
     * evaluated lazily once {@link #getQ()} is called, so this will only save on copy ops.
     *
     * @param subDiagonal Sub-diagonal of the upper triangular/Hessenburg matrix. That is, the sub-diagonal for which all items
     * below will be zero in the final upper quasi-triangular matrix. Must be Zero or one.
     * <ul>
     *     <li>{@code subDiagonal = 0}: Matrix will be upper triangular.</li>
     *     <li>{@code subDiagonal = 1}: Matrix will be upper Hessenburg.</li>
     * </ul>
     * @param storeReflectors Flag indicating if the reflectors used to bring the matrix to upper
     * triangular/Hessenburg form should be stored.
     * @param inPlace Flag indicating if the decomposition should be done in-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in-place.</li>
     *     <li>If {@code false}, then the decomposition will be done out-of-place.</li>
     * </ul>
     * @throws IllegalArgumentException If {@code subDiagonal < 0 || subDiagonal > 1}.
     */
    public RealUnitaryDecomposition(int subDiagonal, boolean storeReflectors, boolean inPlace) {
        super(subDiagonal, storeReflectors, inPlace);
    }


    /**
     * <p>Gets the unitary {@code Q} matrix from the unitary decomposition.
     *
     * <p>Note, if the reflectors for this decomposition were not saved, then {@code Q} can not be computed and this method will be
     * {@code null}.
     *
     * @return The {@code Q} matrix from the unitary decomposition. Note, if the reflectors for this decomposition were not saved,
     * then {@code Q} can not be computed and this method will return {@code null}.
     */
    @Override
    public Matrix getQ() {
        ensureHasDecomposed();
        if(!storeReflectors)
            return null;

        Matrix Q = initQ();
        int startCol = iLow + subDiagonal;
        int lastCol = Math.min(iHigh, minAxisSize) - subDiagonal;

        for(int j=lastCol - 1; j>=startCol; j--) {
            householderVector[j] = 1.0; // Ensure first value of reflector is 1.

            // Extract column containing reflector vector.
            for(int i=j + 1; i<iHigh; i++)
                householderVector[i] = transformData[i*numCols + j - subDiagonal];

            if(qFactors[j]!=0) // Otherwise, no reflector to apply.
                Householder.leftMultReflector(Q, householderVector, qFactors[j], j, j, iHigh, workArray);
        }

        return Q;
    }


    /**
     * Gets the upper triangular/Hessenburg matrix from the last decomposition.
     * @return The upper triangular/Hessenburg matrix from the last decomposition.
     */
    @Override
    protected Matrix getUpper(Matrix H) {
        ensureHasDecomposed();

        // Copy top rows.
        for(int i=0; i<subDiagonal; i++) {
            int rowOffset = i*numCols;
            System.arraycopy(transformData, rowOffset, H.data, rowOffset, numCols);
        }

        // Copy rest of the rows.
        for(int i=subDiagonal; i<minAxisSize; i++) {
            int rowOffset = i*numCols;

            int length = numCols - (i-subDiagonal);
            System.arraycopy(transformData, rowOffset + i - subDiagonal,
                    H.data, rowOffset + i - subDiagonal, length);
        }

        return H;
    }


    /**
     * Initialized any work ndarrays to be used in computing the decomposition with the proper size.
     *
     * @param maxAxisSize Length of the largest axis in the matrix to be decomposed. That is, {@code max(numRows, numCols)}
     */
    @Override
    protected void initWorkArrays(int maxAxisSize) {
        transformData = transformMatrix.data;
        qFactors = new double[minAxisSize]; // Stores scaler factors for the Householder vectors.
        householderVector = new double[maxAxisSize];
        workArray = new double[maxAxisSize];
    }


    /**
     * Computes the Householder vector for the first column of the submatrix with upper left corner at {@code (j, j)}.
     *
     * @param j Index of the upper left corner of the submatrix for which to compute the Householder vector for the first column.
     *          That is, a Householder vector will be computed for the portion of column {@code j} below row {@code j}.
     */
    protected void computeHouseholder(int j) {
        // Initialize storage array for Householder vector and compute maximum absolute value in jth column at or below jth row.
        double maxAbs = findMaxAndInit(j);
        norm = 0; // Ensure norm is reset.

        // If the maximum value in the range of the column is very small, consider it already in the correct form.
        applyUpdate = maxAbs >= Flag4jConstants.EPS_F64;

        if(!applyUpdate) {
            currentFactor = 0;
        } else {
            computePhasedNorm(j, maxAbs);
            double invShift = 1.0 / shift;

            householderVector[j] = 1.0; // Ensure first value in Householder vector is one.
            for(int i=j+1; i<iHigh; i++)
                householderVector[i] *= invShift; // Scale all but first entry of the Householder vector.
        }

        qFactors[j] = currentFactor; // Store the factor for the Householder vector.
    }


    /**
     * Computes the norm of column {@code j} below the {@code j}th row of the matrix to be decomposed. The norm will have the same
     * parity as the first entry in the sub-column.
     * @param j Column to compute norm of below the {@code j}th row.
     * @param maxAbs Maximum absolute value in the column. Used for scaling norm to minimize potential overflow issues.
     */
    protected void computePhasedNorm(int j, double maxAbs) {
        double maxInv = 1.0 / maxAbs;

        // Computes the 2-norm of the column.
        for(int i = j; i < iHigh; i++) {
            householderVector[i] *= maxInv; // Scale items of the householder vector to help reduce potential overflow.
            double scaledValue = householderVector[i];
            norm += scaledValue*scaledValue;
        }
        norm = Math.sqrt(norm); // Finish 2-norm computation for the column.

        // Change sign of norm depending on first entry in column for stability purposes in Householder vector.
        if(householderVector[j] < 0) norm = -norm;

        shift = householderVector[j] + norm;
        currentFactor = shift/norm;
        norm *= maxAbs; // Rescale norm.
    }


    /**
     * Finds the maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     * This method also initializes the first {@code numRows-j} items of the storage array {@link #householderVector}
     * to the items of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     */
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;
        int idx = j*numCols + j - subDiagonal;

        for(int i = j; i < iHigh; i++) {
            double d = householderVector[i] = transformData[idx];
            idx += numCols; // Move index to next row.
            maxAbs = Math.max(Math.abs(d), maxAbs);
        }

        return maxAbs;
    }


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of submatrix for which the Householder reflector was computed for.
     */
    @Override
    protected void updateData(int j) {
        if(subDiagonal >= 0) // Right multiply transform matrix to reflector. (i.e., left multiply reflector to matrix).
            Householder.leftMultReflector(transformMatrix, householderVector, qFactors[j], j, j, iHigh, workArray);

        if(subDiagonal == 1) // Left multiply transform matrix to reflector. (i.e., right multiply reflector to matrix).
            Householder.rightMultReflector(transformMatrix, householderVector, qFactors[j], 0, j, iHigh);

        if(j < numCols) transformData[j*numCols + j - subDiagonal] = -norm;

        if(storeReflectors) {
            // Store the Q matrix in the lower portion of the transformation items matrix.
            for(int i = j + 1; i < iHigh; i++)
                transformData[i*numCols + j - subDiagonal] = householderVector[i];
        }
    }
}
