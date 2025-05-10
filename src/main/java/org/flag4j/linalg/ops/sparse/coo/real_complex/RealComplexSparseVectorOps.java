/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo.real_complex;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This class contains low-level implementations of ops on a real sparse tensor and a complex sparse tensor.
 */
public final class RealComplexSparseVectorOps {

    private RealComplexSparseVectorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero data.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVector add(CooVector src, Complex128 a) {
        Complex128[] dest = new Complex128[src.size];
        Arrays.fill(dest, a);

        for(int i = 0; i<src.data.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add(src.data[i]);
        }

        return new CVector(dest);
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero data.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVector add(CooCVector src, double a) {
        Complex128[] dest = new Complex128[src.size];
        ArrayBuilder.fill(dest, a);

        for(int i = 0; i<src.data.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add(src.data[i]);
        }

        return new CVector(dest);
    }


    /**
     * Subtracts a real number from each entry of a sparse vector, including the zero data.
     * @param src Sparse vector to subtract value from.
     * @param a Value to subtract from the {@code src} sparse vector.
     * @return The result of subtracting the specified value from the sparse vector.
     */
    public static CVector sub(CooVector src, Complex128 a) {
        Complex128[] dest = new Complex128[src.size];
        Arrays.fill(dest, a.addInv());

        for(int i = 0; i<src.data.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add(src.data[i]);
        }

        return new CVector(dest);
    }


    /**
     * Computes the element-wise vector addition between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooCVector add(CooCVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Complex128> values = new ArrayList<>(src1.data.length);
        List<Integer> indices = new ArrayList<>(src1.data.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter].add(src2.data[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(new Complex128(src2.data[src2Counter]));
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.data.length) {
            for(int i = src1Counter; i<src1.data.length; i++) {
                values.add(src1.data[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.data.length) {
            for(int i = src2Counter; i<src2.data.length; i++) {
                values.add(new Complex128(src2.data[i]));
                indices.add(src2.indices[i]);
            }
        }

        return CooCVector.unsafeMake(
                src1.size,
                values.toArray(new Complex128[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector subtraction between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooCVector sub(CooCVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Complex128> values = new ArrayList<>(src1.data.length);
        List<Integer> indices = new ArrayList<>(src1.data.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter].sub(src2.data[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(new Complex128(-src2.data[src2Counter]));
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.data.length) {
            for(int i = src1Counter; i<src1.data.length; i++) {
                values.add(src1.data[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.data.length) {
            for(int i = src2Counter; i<src2.data.length; i++) {
                values.add(new Complex128(-src2.data[i]));
                indices.add(src2.indices[i]);
            }
        }

        return CooCVector.unsafeMake(
                src1.size,
                values.toArray(new Complex128[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector subtraction between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooCVector sub(CooVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Complex128> values = new ArrayList<>(src1.data.length);
        List<Integer> indices = new ArrayList<>(src1.data.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(new Complex128(src1.data[src1Counter]).sub(src2.data[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(new Complex128(src1.data[src1Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.data[src2Counter].addInv());
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.data.length) {
            for(int i = src1Counter; i<src1.data.length; i++) {
                values.add(new Complex128(src1.data[i]));
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.data.length) {
            for(int i = src2Counter; i<src2.data.length; i++) {
                values.add(src2.data[i].addInv());
                indices.add(src2.indices[i]);
            }
        }

        return CooCVector.unsafeMake(
                src1.size,
                values.toArray(new Complex128[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector multiplication between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @return The result of the vector multiplication.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooCVector elemMult(CooCVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Complex128> values = new ArrayList<>(src1.data.length);
        List<Integer> indices = new ArrayList<>(src1.data.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                values.add(src1.data[src1Counter].mult(src2.data[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return CooCVector.unsafeMake(
                src1.size,
                values.toArray(Complex128[]::new),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the inner product of a real and complex sparse vector. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static Complex128 inner(CooCVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128 product = Complex128.ZERO;

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src1.data[src1Counter].mult(src2.data[src2Counter]));
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the inner product of a real and complex sparse vector. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static Complex128 inner(CooVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128 product = Complex128.ZERO;

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src2.data[src2Counter].conj().mult(src1.data[src1Counter]));
                src1Counter++;
                src2Counter++;
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the vector outer product between a complex sparse vector and a real sparse vector.
     * @param src1 Entries of the first sparse vector in the outer product.
     * @param src2 Second sparse vector in the outer product.
     * @return The matrix resulting from the vector outer product.
     */
    public static CMatrix outerProduct(CooCVector src1, CooVector src2) {
        Complex128[] dest = new Complex128[src2.size*src1.size];
        Arrays.fill(dest, Complex128.ZERO);

        int destRow;
        int index1;
        int index2;

        for(int i = 0; i<src1.data.length; i++) {
            index1 = src1.indices[i];
            destRow = index1*src1.size;

            for(int j = 0; j<src2.data.length; j++) {
                index2 = src2.indices[j];

                dest[destRow + index2] = src1.data[i].mult(src2.data[j]);
            }
        }

        return new CMatrix(src1.size, src2.size, dest);
    }


    /**
     * Computes the vector outer product between a complex sparse vector and a real sparse vector.
     * @param src1 Entries of the first sparse vector in the outer product.
     * @param src2 Second sparse vector in the outer product.
     * @return The matrix resulting from the vector outer product.
     */
    public static CMatrix outerProduct(CooVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        Complex128[] dest = new Complex128[src2.size*src1.size];
        Arrays.fill(dest, Complex128.ZERO);

        int destRow;
        int index1;
        int index2;

        for(int i = 0; i<src1.data.length; i++) {
            index1 = src1.indices[i];
            destRow = index1*src1.size;

            for(int j = 0; j<src2.data.length; j++) {
                index2 = src2.indices[j];

                dest[destRow + index2] = src2.data[j].mult(src1.data[i]);
            }
        }

        return new CMatrix(src1.size, src2.size, dest);
    }
}
