/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense_sparse.coo.real_complex;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringProperties;
import org.flag4j.numbers.Complex128;

import java.util.Arrays;

/**
 * This class contains methods for checking the equality of real dense/sparse and complex dense/sparse tensors.
 */
public final class RealComplexDenseSparseEquals {

    private RealComplexDenseSparseEquals() {
        // Hide default constructor.
        
    }


    /**
     * Checks if a real dense vector is equal to a complex sparse vector equals.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two vectors are equal. Returns false otherwise.
     */
    public static boolean vectorEquals(double[] src1, Complex128[] src2, int[] indices, int sparseSize) {
        boolean equal = true;

        if(src1.length==sparseSize) {
            int index;
            double[] src1Copy = Arrays.copyOf(src1, src1.length);

            for(int i=0; i<src2.length; i++) {
                index = indices[i];

                if(!src2[i].equals(src1[index])) {
                    equal=false;
                    break;

                } else {
                    src1Copy[index] = 0;
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the entriesStack
                equal = RealProperties.isAllZeros(src1Copy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a complex dense vector is equal to a real sparse vector.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two vectors are equal. Returns false otherwise.
     */
    public static boolean vectorEquals(Complex128[] src1, double[] src2, int[] indices, int sparseSize) {
        boolean equal = true;

        if(src1.length == sparseSize) {
            int index;
            Complex128[] src1Copy = new Complex128[src1.length];
            System.arraycopy(src1, 0, src1Copy, 0, src1.length);

            for(int i=0; i<indices.length; i++) {
                index = indices[i];

                if(!src1[index].equals(src2[i])) {
                    equal=false;
                    break;

                } else {
                    src1Copy[index] = Complex128.ZERO;
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the entriesStack
                equal = SemiringProperties.isZeros(src1Copy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a real dense matrix is equal to a sparse complex matrix.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(Matrix A, CooCMatrix B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            double[] entriesCopy = Arrays.copyOf(A.data, A.data.length);

            int rowIndex, colIndex;
            int entriesIndex;

            // Remove all nonZero items from the items of this matrix.
            for(int i=0; i<B.nnz; i++) {
                rowIndex = B.rowIndices[i];
                colIndex = B.colIndices[i];
                int idx = rowIndex*A.numCols + colIndex;

                if(entriesCopy[idx] != B.data[i].re || B.data[i].im != 0) {
                    equal = false;
                    break;
                }

                entriesCopy[idx] = 0;
            }

            if(equal) {
                // Now, if this matrix is equal to the sparse matrix, there should only be zeros left in the entriesStack
                equal = RealProperties.isAllZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a dense complex matrix is equal to a real sparse matrix.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(CMatrix A, CooMatrix B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            Complex128[] entriesCopy = Arrays.copyOf(A.data, A.data.length);

            int rowIndex, colIndex;

            // Remove all nonZero items from the items of this matrix.
            for(int i=0; i<B.nnz; i++) {
                rowIndex = B.rowIndices[i];
                colIndex = B.colIndices[i];
                int idx = rowIndex*A.numCols + colIndex;

                if(!entriesCopy[idx].equals(B.data[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[idx] = Complex128.ZERO;
            }

            if(equal) {
                // Now, if this matrix is equal to the sparse matrix, there should only be zeros left in the entriesStack
                equal = SemiringProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a real dense tensor is equal to a complex sparse tensor.
     * @param A Real dense tensor.
     * @param B Complex sparse tensor.
     * @return True if the two tensors are element-wise equivalent.
     */
    public static boolean tensorEquals(Tensor A, CooCTensor B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            double[] entriesCopy = Arrays.copyOf(A.data, A.data.length);
            int entriesIndex;

            // Remove all nonZero items from the items of this matrix.
            for(int i=0; i<B.nnz; i++) {
                entriesIndex = A.shape.get1DIndex(B.indices[i]);

                if(entriesCopy[entriesIndex] != (B.data[i]).re || (B.data[i]).im != 0) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.get1DIndex(B.indices[i])] = 0;
            }

            if(equal) {
                // Now, if this tensor is equal to the sparse tensor, there should only be zeros left in the entriesStack
                equal = RealProperties.isAllZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a complex dense tensor is equal to a real sparse tensor.
     * @param A Complex dense tensor.
     * @param B Real sparse tensor.
     * @return True if the two tensors are element-wise equivalent.
     */
    public static boolean tensorEquals(CTensor A, CooTensor B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            Complex128[] entriesCopy = new Complex128[A.data.length];
            System.arraycopy(A.data, 0, entriesCopy, 0, A.data.length);
            int entriesIndex;

            // Remove all nonZero items from the items of this matrix.
            for(int i=0; i<B.nnz; i++) {
                entriesIndex = A.shape.get1DIndex(B.indices[i]);

                if(entriesCopy[entriesIndex].re != B.data[i] || entriesCopy[entriesIndex].im != 0) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.get1DIndex(B.indices[i])] = Complex128.ZERO;
            }

            if(equal) {
                // Now, if this tensor is equal to the sparse tensor, there should only be zeros left in the entriesStack
                equal = SemiringProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }
}
