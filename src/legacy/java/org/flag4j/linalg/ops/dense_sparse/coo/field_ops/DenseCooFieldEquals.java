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

package org.flag4j.linalg.ops.dense_sparse.coo.field_ops;

import org.flag4j.arrays.backend.field_arrays.*;
import org.flag4j.linalg.ops.common.semiring_ops.SemiringProperties;
import org.flag4j.numbers.FieldElement;

import java.util.Arrays;

/**
 * This class provides methods for checking the equality of a dense field tensor with a sparse field tensor.
 */
public final class DenseCooFieldEquals {

    private DenseCooFieldEquals() {
        // Hide constructor for utility class. for utility class.
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static <T extends FieldElement<T>> boolean matrixEquals(
            AbstractDenseFieldMatrix<?, ?, T> A,
            AbstractCooFieldMatrix<?, ?, ?, T> B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            T[] entriesCopy = Arrays.copyOf(A.data, A.data.length);

            // Remove all nonZero items from the items of this matrix.
            for(int i=0; i<B.nnz; i++) {
                int rowIndex = B.rowIndices[i];
                int colIndex = B.colIndices[i];
                int idx = rowIndex*A.numCols + colIndex;

                if(!entriesCopy[idx].equals(B.data[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[idx] = A.getZeroElement();
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
     * Checks if a complex dense vector is equal to a complex sparse vector.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero items in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two vectors are equal. Returns false otherwise.
     */
    public static <T extends FieldElement<T>> boolean vectorEquals(
            AbstractDenseFieldVector<?, ?, T> src1, AbstractCooFieldVector<?, ?, ?, ?, T> src2) {
        boolean equal = true;
        final T ZERO = (src1.size > 0) ? src1.data[0].getZero() : null;

        if(src1.size == src2.size) {
            int index;
            T[] src1Copy = (T[]) new FieldElement[src1.size];
            System.arraycopy(src1.data, 0, src1Copy, 0, src1.size);

            for(int i=0; i<src2.size; i++) {
                index = src2.indices[i];

                if(!src1.data[index].equals(src2.data[i])) {
                    equal = false;
                    break;
                } else {
                    src1Copy[index] = ZERO;
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the copy
                equal = SemiringProperties.isZeros(src1Copy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a complex dense tensor is equal to a complex sparse tensor.
     * @param A Complex dense tensor.
     * @param B Complex sparse tensor.
     * @return True if the two tensors are element-wise equivalent.
     */
    public static <T extends FieldElement<T>> boolean tensorEquals(
            AbstractDenseFieldTensor<?, T> A, AbstractCooFieldTensor<?, ?, T> B) {
        if(!A.shape.equals(B.shape)) return false;
        final T ZERO = (A.data.length > 0) ? A.data[0].getZero() : null;

        T[] entriesCopy = (T[]) new FieldElement[A.data.length];
        System.arraycopy(A.data, 0, entriesCopy, 0, A.data.length);
        int entriesIndex;

        // Remove all nonZero items from the items of this matrix.
        for(int i=0; i<B.nnz; i++) {
            entriesIndex = A.shape.get1DIndex(B.indices[i]);

            if(!entriesCopy[entriesIndex].equals(B.data[i]))
                return false;

            entriesCopy[entriesIndex] = ZERO;
        }

        // Now, if this tensor is equal to the sparse tensor, there should only be zeros left in the entriesStack
        return SemiringProperties.isZeros(entriesCopy);
    }
}
