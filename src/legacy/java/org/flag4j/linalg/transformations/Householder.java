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

package org.flag4j.linalg.transformations;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringOps;
import org.flag4j.numbers.Complex128;

/**
 * This class contains methods for computing real or complex Householder reflectors (also known as elementary reflectors).
 * A Householder reflector is a transformation matrix which reflects a vector about a hyperplane containing the origin.
 */
public final class Householder {

    private Householder() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the Householder reflector which describes a reflection through a hyperplane containing the origin which
     * is normal to the specified {@code normal} vector.
     * @param normal The vector normal to the plane the Householder reflector will reflect through.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e., a Householder reflector.
     */
    public static Matrix getReflector(Vector normal) {
        Vector v;

        double signedNorm = -Math.copySign(VectorNorms.norm(normal.data), normal.data[0]);
        v = normal.div(normal.data[0] - signedNorm);
        v.data[0] = 1.0;

        // Create Householder matrix
        Matrix H = v.outer(v).mult(-2.0/v.inner(v));

        int step = H.numCols+1;
        for(int i = 0; i<H.data.length; i+=step)
            H.data[i] = 1 + H.data[i];

        return H;
    }


    /**
     * Computes the vector <span class="latex-inline">v</span> in of a Householder matrix 
     * <span class="latex-inline">H = I-2vv<sup>T</sup></span> where <span class="latex-inline">H</span> is a
     * transformation matrix which reflects a vector across the plane normal to {@code normal}.
     *
     * <p>This method may be used in conjunction with {@link #leftMultReflector(Matrix, Vector, double, int, int, int)} and
     * {@link #rightMultReflector(Matrix, Vector, double, int, int, int)} to efficiently apply reflectors. 
     * Doing this is <span class="latex-inline">O(n^2)</span> while forming the full Householder matrix and performing 
     * matrix multiplication is <span class="latex-inline">O(n^3)</span>.
     *
     * @param normal Vector normal to the plane which <span class="latex-inline">H</span> reflects across.
     * @return The vector <span class="latex-inline">v</span> in of a Householder matrix
     * <span class="latex-inline">H = I-2vv<sup>T</sup></span> which reflects across a plane
     * normal to {@code normal}.
     */
    public static Vector getVector(Vector normal) {
        double normX = VectorNorms.norm(normal.data);
        double x1 = normal.data[0];
        normX = (x1 >= 0) ? -normX : normX;
        double v1 = x1 - normX;

        // Initialize v with norm and set first element
        Vector v = normal.copy();
        v.data[0] = v1;

        // Compute norm of v noting that it only differs from normal by the first element.
        double normV = Math.sqrt(normX*normX - x1*x1 + v1*v1);

        for(int i = 0; i<v.data.length; i++)
            v.data[i] /= normV; // Normalize v to make it a unit vector

        return v;
    }


    /**
     * Computes the Householder reflector which describes a reflection through a hyperplane containing the origin which
     * is normal to the specified {@code normal} vector.
     *
     * <p>This method may be used in conjunction with {@link #leftMultReflector(CMatrix, CVector, Complex128, int, int, int)} and
     * {@link #rightMultReflector(CMatrix, CVector, Complex128, int, int, int)} to efficiently apply reflectors. Doing this is
     * <span class="latex-inline">O(n^2)</span>
     * while forming the full Householder matrix and performing matrix multiplication is <span class="latex-inline">O(n^3)</span>
     *
     * @param normal The vector normal to the plane the Householder reflector will reflect through\.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e., a Householder reflector.
     */
    public static CMatrix getReflector(CVector normal) {
        CVector v;

        // Compute signed norm using modified sgn function.
        Complex128 signedNorm = normal.data[0].equals(0) ?
                new Complex128(-VectorNorms.norm(normal.data)) :
                Complex128.sgn(normal.data[0]).mult(-VectorNorms.norm(normal.data));

        v = normal.div(normal.data[0].sub(signedNorm));
        v.data[0] = new Complex128(1);

        // Create projection matrix
        CMatrix P = v.outer(v).mult(-2.0/v.innerSelf());

        int step = P.numCols+1;
        for(int i = 0; i<P.data.length; i+=step)
            P.data[i] = P.data[i].add(1.0);

        return P;
    }


    /**
     * Left multiplies a Householder matrix <span class="latex-inline">H = I - &alpha;vv<sup>T</sup></span>, represented by the vector
     * v, to another matrix A. That is, computes
     * <span class="latex-inline">HA = (I - &alpha;vv<sup>T</sup>)A</span>.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param workArray An array to store temporary column items. This can help both with cache performance and reducing unneeded
     *                  garbage collection if this method is called repeatedly.
     */
    public static void leftMultReflector(Matrix src,
                                         double[] householderVector,
                                         double alpha,
                                         int startCol,
                                         int startRow, int endRow,
                                         double[] workArray) {
        // Note: this computes A - (alpha*v)*(v^T*A) rather than A - (alpha*v*v^T)*A.
        // The first method takes ~2(n^2 + n) flops while the second method takes ~(n^3 + n^2 + 2n) flops.

        int numCols = src.numCols;
        int srcRowOffset = startRow*numCols;
        double v0 = householderVector[startRow];

        for(int i=startCol; i<numCols; i++)
            workArray[i] = v0*src.data[srcRowOffset + i];

        for(int k=startRow + 1; k<endRow; k++) {
            int srcIdx = k*numCols + startCol;
            double reflectorValue = householderVector[k];
            for(int i=startCol; i<numCols; i++)
                workArray[i] += reflectorValue*src.data[srcIdx++];
        }

        RealOps.scalMult(workArray, alpha, startCol, numCols, workArray);

        for(int i=startRow; i<endRow; i++) {
            double reflectorValue = householderVector[i];
            int indexA = i*numCols + startCol;

            for(int j=startCol; j<numCols; j++)
                src.data[indexA++] -= reflectorValue*workArray[j];
        }
    }


    /**
     * Right multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>T</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>. That is, computes 
     * <span class="latex-inline">AH<sup>T</sup> = A(I-&alpha; vv<sup>T</sup>)<sup>T</sup></span>.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(Matrix src,
                                          double[] householderVector,
                                          double alpha,
                                          int startCol,
                                          int startRow, int endRow) {

        for(int i=startCol; i<src.numRows; i++) {
            int startIndex = i*src.numCols + startRow;
            double sum = 0;
            int rowIndex = startIndex;

            for(int j = startRow; j < endRow; j++)
                sum += src.data[rowIndex++]*householderVector[j];
            sum *= -alpha;

            rowIndex = startIndex;
            for(int j=startRow; j<endRow; j++)
                src.data[rowIndex++] += sum*householderVector[j];
        }
    }


    /**
     * Left multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>H</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>. 
     * That is, computes <span class="latex-inline">HA = (I-&alpha; vv<sup>H</sup>)A</span>.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param workArray An array to store temporary column items. This can help both with cache performance and reducing unneeded
     *                  garbage collection if this method is called repeatedly.
     */
    public static void leftMultReflector(CMatrix src,
                                         Complex128[] householderVector,
                                         Complex128 alpha,
                                         int startCol,
                                         int startRow, int endRow,
                                         Complex128[] workArray) {
        int numCols = src.numCols;
        int srcRowOffset = startRow*numCols;
        Complex128 v0 = householderVector[startRow].conj();

        for(int i=startCol; i<numCols; i++)
            workArray[i] = v0.mult(src.data[srcRowOffset + i]);

        for(int k=startRow + 1; k<endRow; k++) {
            int srcIdx = k*numCols + startCol;
            Complex128 reflectorValue = householderVector[k].conj();

            for(int i=startCol; i<numCols; i++)
                workArray[i] = workArray[i].add(reflectorValue.mult(src.data[srcIdx++]));
        }

        SemiringOps.scalMult(workArray, alpha, workArray, startCol, numCols);

        for(int i=startRow; i<endRow; i++) {
            Complex128 reflectorValue = householderVector[i];
            int indexA = i*numCols + startCol;

            for(int j=startCol; j<numCols; j++)
                src.data[indexA] = src.data[indexA++].sub(reflectorValue.mult(workArray[j]));
        }
    }


    /**
     * Right multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>H</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>. That is, computes 
     * <span class="latex-inline">AH<sup>H</sup> = A(I-&alpha; vv<sup>H</sup>)<sup>H</sup></span>.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(CMatrix src,
                                          Complex128[] householderVector,
                                          Complex128 alpha,
                                          int startCol,
                                          int startRow, int endRow) {
        Complex128 negAlpha = alpha.negate();

        for(int i=startCol; i<src.numRows; i++) {
            int startIndex = i*src.numCols + startRow;
            Complex128 sum = Complex128.ZERO;
            int rowIndex = startIndex;

            for(int j = startRow; j < endRow; j++)
                sum = sum.add(src.data[rowIndex++].mult(householderVector[j]));
            sum = sum.mult(negAlpha);

            rowIndex = startIndex;
            for(int j=startRow; j<endRow; j++)
                src.data[rowIndex] = src.data[rowIndex++].add(sum.mult(householderVector[j].conj()));
        }
    }


    /**
     * <p>Applies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>T</sup></span>, represented by the vector
     * <span class="latex-inline">v</span> to a
     * symmetric matrix <span class="latex-inline">A</span> on both the left and right side. That is, computes
     * <span class="latex-inline">HAH<sup>T</sup></span>.
     *
     * <p>Note: no check is made to
     * explicitly check that the {@code src} matrix is actually symmetric.
     *
     * @param src Matrix to apply the Householder reflector to. Assumed to be square and symmetric. Upper triangular portion
     * overwritten with the result.
     * @param householderVector Householder vector <span class="latex-inline">v</span> from the definition of a Householder reflector
     * matrix.
     * @param alpha The scalar &alpha; value in Householder reflector matrix definition.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param workArray Array for storing temporary values during the computation. Contents will be overwritten.
     */
    public static void symmLeftRightMultReflector(Matrix src,
                                                  double[] householderVector,
                                                  double alpha,
                                                  int startCol,
                                                  double[] workArray) {
        int numRows = src.numRows;

        // Computes w = -alpha*A*v
        for(int i=startCol; i<numRows; i++) {
            double total = 0;
            int rowOffset = i*numRows;

            for(int j=startCol; j<i; j++)
                total += src.data[j*numRows + i]*householderVector[j];
            for(int j=i; j<src.numRows; j++)
                total += src.data[rowOffset + j]*householderVector[j];

            workArray[i] = -alpha*total;
        }

        // Computes -0.5*alpha*v^T*w
        double innerProd = 0;
        for(int i=startCol; i<numRows; i++)
            innerProd += householderVector[i]*workArray[i];
        innerProd *= -0.5*alpha;

        // Computes w + innerProd*v
        for(int i=startCol; i<numRows; i++)
            workArray[i] += innerProd*householderVector[i];

        // Computes A + w*v^T + v*w^T
        for(int i=startCol; i<numRows; i++) {
            double prod = workArray[i];
            double h = householderVector[i];
            int rowOffset = i*numRows;

            for(int j=i; j<src.numRows; j++)
                src.data[rowOffset + j] += prod*householderVector[j] + workArray[j]*h;
        }
    }


    /**
     * Left multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>T</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>. That is, computes
     * <span class="latex-inline">HA = (I-&alpha; vv<sup>T</sup> )A</span>.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     */
    public static void leftMultReflector(Matrix src,
                                         Vector householderVector,
                                         double alpha,
                                         int startCol,
                                         int startRow, int endRow) {
        leftMultReflector(src, householderVector.data, alpha, startCol, startRow, endRow, new double[src.numCols-startCol]);
    }


    /**
     * Right multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>T</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>. That is, computes
     * <span class="latex-inline">AH<sup>T</sup> = A(I-&alpha; vv<sup>T</sup>)<sup>T</sup></span>.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.
     *
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(Matrix src,
                                          Vector householderVector,
                                          double alpha,
                                          int startCol,
                                          int startRow, int endRow) {

        rightMultReflector(src, householderVector.data, alpha, startCol, startRow, endRow);
    }


    /**
     * Left multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>H</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>.
     * That is, computes <span class="latex-inline">HA = (I-&alpha; vv<sup>H</sup>)A</span>.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.
     *
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     */
    public static void leftMultReflector(CMatrix src,
                                         CVector householderVector,
                                         Complex128 alpha,
                                         int startCol,
                                         int startRow, int endRow) {
        leftMultReflector(src, householderVector.data, alpha, startCol, startRow, endRow, new Complex128[src.numCols-startCol]);
    }


    /**
     * Right multiplies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>H</sup></span>, represented by the vector
     * <span class="latex-inline">v</span>, to another matrix <span class="latex-inline">A</span>.
     * That is, computes <span class="latex-inline">AH<sup>H</sup> = A(I-&alpha; vv<sup>H</sup>)<sup>H</sup></span>.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.
     *
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector <span class="latex-inline">v</span>.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param startRow Starting row of submatrix in {@code src} to apply reflector to.
     * @param endRow Starting row of submatrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(CMatrix src,
                                          CVector householderVector,
                                          Complex128 alpha,
                                          int startCol,
                                          int startRow, int endRow) {
        rightMultReflector(src, householderVector.data, alpha, startCol, startRow, endRow);
    }


    /**
     * <p>Applies a Householder matrix <span class="latex-inline">H = I-&alpha; vv<sup>H</sup></span>, represented by the vector <span class="latex-inline">v</span> to a
     * Hermitian matrix <span class="latex-inline">A</span> on both the left and right side.
     * That is, computes <span class="latex-inline">HAH<sup>H</sup></span>.
     *
     * <p>Note: no check is made to
     * explicitly check that the {@code src} matrix is actually Hermitian.
     *
     * @param src Matrix to apply the Householder reflector to. Assumed to be square and Hermitian. Upper triangular portion
     * overwritten with the result.
     * @param householderVector Householder vector <span class="latex-inline">v</span> from the definition of a Householder reflector matrix.
     * @param alpha The scalar &alpha; value in Householder reflector matrix definition.
     * @param startCol Starting column of submatrix in {@code src} to apply reflector to.
     * @param workArray Array for storing temporary values during the computation. Contents will be overwritten.
     */
    public static void hermLeftRightMultReflector(CMatrix src,
                                                  Complex128[] householderVector,
                                                  Complex128 alpha,
                                                  int startCol,
                                                  Complex128[] workArray) {
        int numRows = src.numRows;

        // Computes w = -alpha*A*v (taking conjugate for lower triangular part)
        for (int i = startCol; i < numRows; i++) {
            Complex128 total = new Complex128(0, 0);
            int rowOffset = i * numRows;

            for (int j = startCol; j < i; j++)
                total = total.add(src.data[j*numRows + i].conj().mult(householderVector[j]));
            for (int j = i; j < src.numRows; j++)
                total = total.add(src.data[rowOffset + j].mult(householderVector[j]));

            workArray[i] = alpha.mult(total).negate();
        }

        // Computes -0.5*alpha*v^T*w (with conjugation in the scalar product)
        Complex128 innerProd = new Complex128(0, 0);
        for (int i = startCol; i < numRows; i++)
            innerProd = innerProd.add(householderVector[i].conj().mult(workArray[i]));

        innerProd = innerProd.mult(alpha).mult(new Complex128(-0.5, 0));

        // Computes w + innerProd*v
        for (int i = startCol; i < numRows; i++)
            workArray[i] = workArray[i].add(innerProd.mult(householderVector[i]));

        // Computes A + w*v^T + v*w^T (ensuring Hermitian property is maintained)
        for (int i = startCol; i < numRows; i++) {
            Complex128 prod = workArray[i];
            Complex128 h = householderVector[i].conj();
            int rowOffset = i * numRows;

            for (int j = i; j < src.numRows; j++) {
                src.data[rowOffset + j] = src.data[rowOffset + j]
                        .add(prod.mult(householderVector[j]))
                        .add(workArray[j].mult(h));
            }
        }
    }
}
