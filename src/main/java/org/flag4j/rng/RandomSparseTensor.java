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

package org.flag4j.rng;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.*;
import org.flag4j.numbers.Complex128;
import org.flag4j.rng.distributions.Complex128UniformDisk;
import org.flag4j.rng.distributions.RealUniform;
import org.flag4j.util.ArrayJoiner;
import org.flag4j.util.ValidateParameters;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * A utility class for generating random sparse tensors and matrices with customizable sparsity and distributions.
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li>Generate sparse matrices and tensors with specified sparsity levels.</li>
 *   <li>Support for uniform and annular (complex-valued) distributions for non-zero values.</li>
 *   <li>Create symmetric sparse matrices.</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 *     RandomSparseTensor generator = new RandomSparseTensor(12345L);
 *     CooMatrix sparseMatrix = generator.randomCooMatrix(100, 100, 0, 10, 0.95);
 *     CsrMatrix csrMatrix = generator.randomCsrMatrix(new Shape(50, 50), 0, 1, 0.9);
 * }</pre>
 *
 * @see RandomComplex
 * @see RandomDenseTensor
 * @see RandomArray
 */
public class RandomSparseTensor {

    /**
     * Complex pseudorandom number generator.
     */
    private final RandomComplex COMPLEX_RNG;
    /**
     * Generator for random arrays.
     */
    private final RandomArray RAND_ARRAY;


    /**
     * Constructs a new pseudorandom tensor generator with seed set to {@link RandomState#getGlobalSeed()}.
     */
    public RandomSparseTensor() {
        COMPLEX_RNG = RandomState.getDefaultRng();
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }


    /**
     * Constructs a pseudorandom tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the pseudorandom tensor generator.
     */
    public RandomSparseTensor(long seed) {
        COMPLEX_RNG = new RandomComplex(seed);
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }


    /**
     * Computes the total number of non-zero entries required for a sparse vector to have the specified sparsity.
     * @param totalSize Total size of the vector.
     * @param sparsity The desired sparsity. Assumed to be between 0 and 1 (both inclusive).
     * @return The number of non-zero entries required for a vector of the specified {@code size}
     * to have the desired {@code sparsity}.
     */
    private static int nnzFromSparsity(int totalSize, double sparsity) {
        if(sparsity < 0.0 || sparsity > 1.0)
            throw new IllegalArgumentException("sparsity must be between 0.0 and 1.0 but got " + sparsity + ".");
        return BigDecimal.valueOf(totalSize*(1.0-sparsity)).setScale(0, RoundingMode.HALF_UP).intValueExact();
    }


    /**
     * Computes the total number of non-zero entries required for a sparse tensor to have the specified sparsity.
     * @param shape Full shape of the tensor.
     * @param sparsity The desired sparsity. Assumed to be between 0 and 1 (both inclusive).
     * @return The number of non-zero entries required for a vector of the specified {@code size}
     * to have the desired {@code sparsity}.
     */
    private static int nnzFromSparsity(Shape shape, double sparsity) {
        if(sparsity < 0.0 || sparsity > 1.0)
            throw new IllegalArgumentException("sparsity must be between 0.0 and 1.0 but got " + sparsity + ".");
        return new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();
    }


    /**
     * Constructs a COO vector with the specified sparsity filled with pseudo-random values from a uniform
     * distribution in [min, max).
     * @param size Full size of the COO vector.
     * @param min Minimum value of the uniform distribution to sample from (inclusive).
     * @param max Maximum value of the uniform distribution to sample from (exclusive).
     * @param sparsity The sparsity of the COO vector.
     * @return A sparse COO vector whose entries are uniformly distributed in [min, max).
     */
    public CooVector randomCooVector(int size, double min, double max, double sparsity) {
        return randomCooVector(size, min, max, nnzFromSparsity(size, sparsity));
    }


    /**
     * Constructs a COO vector with the specified number of non-zero entries filled with pseudo-random values from a uniform
     * distribution in [min, max).
     * @param size Full size of the COO vector.
     * @param min Minimum value of the uniform distribution to sample from (inclusive).
     * @param max Maximum value of the uniform distribution to sample from (exclusive).
     * @param nnz The number of non-zero value to include in the vector.
     * @return A sparse COO vector whose entries are uniformly distributed in [min, max).
     */
    public CooVector randomCooVector(int size, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz);
        ValidateParameters.ensureLessEq(size, nnz, "nnz");

        double[] data = new double[nnz];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        int[] indices = RAND_ARRAY.randomUniqueIntegers(nnz, 0, size);

        return CooVector.unsafeMake(size, data, indices);
    }


    /**
     * Constructs a COO vector with the specified sparsity entries filled with pseudo-random values uniformly distributed in
     * an annulus centered on the complex plane.
     * @param size Full size of the COO vector.
     * @param min Inner radius of the annulus (inclusive).
     * @param max Outer radius of the annulus (exclusive).
     * @param sparsity The sparsity of the COO vector.
     * @return A sparse COO vector whose entries are uniformly distributed in [min, max).
     */
    public CooCVector randomCooCVector(int size, double min, double max, double sparsity) {
        return randomCooCVector(size, min, max, nnzFromSparsity(size, sparsity));
    }


    /**
     * Constructs a COO vector with the specified number of non-zero entries filled with pseudo-random values uniformly distributed in
     * an annulus centered on the complex plane.
     * @param size Full size of the COO vector.
     * @param min Inner radius of the annulus (inclusive).
     * @param max Outer radius of the annulus (exclusive).
     * @param nnz The number of non-zero value to include in the vector.
     * @return A sparse COO vector whose entries are uniformly distributed in [min, max).
     */
    public CooCVector randomCooCVector(int size, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz);
        ValidateParameters.ensureLessEq(size, nnz, "nnz");

        Complex128[] data = new Complex128[nnz];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        int[] indices = RAND_ARRAY.randomUniqueIntegers(nnz, 0, size);

        return CooCVector.unsafeMake(size, data, indices);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomCooMatrix(new Shape(rows, cols), min, max, sparsity);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(Shape shape, double min, double max, double sparsity) {
        return randomCooMatrix(shape, min, max, nnzFromSparsity(shape, sparsity));
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param nnz Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(int rows, int cols, double min, double max, int nnz) {
        return randomCooMatrix(new Shape(rows, cols), min, max, nnz);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param nnz Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(Shape shape, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz);
        ValidateParameters.ensureRank(shape, 2);
        ValidateParameters.ensureLessEq(shape.totalEntries(), nnz, "nnz");

        double[] data = new double[nnz];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(nnz, 0, shape.getSize(0), 0, shape.getSize(1));

        return new CooMatrix(shape, data, indices[0], indices[1]);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomCooMatrix(new Shape(rows, cols), min, max, sparsity).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(Shape shape, double min, double max, double sparsity) {
        ValidateParameters.ensureRank(shape, 2);
        return randomCooMatrix(shape, min, max, sparsity).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param nnz Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(int rows, int cols, double min, double max, int nnz) {
        return randomCooMatrix(new Shape(rows, cols), min, max, nnz).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param nnz Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(Shape shape, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz);
        ValidateParameters.ensureLessEq(shape.totalEntries(), nnz, "nnz");

        double[] data = new double[nnz];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(nnz, 0, shape.getSize(0), 0, shape.getSize(1));

        return new CooMatrix(shape, data, indices[0], indices[1]).toCsr();
    }


    /**
     * Generates a symmetric {@link CooMatrix COO matrix} filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @param min Minimum value in uniform distribution.
     * @param max Maximum value in uniform distribution.
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     * a value in {@code [0.0, 1.0]}. The true sparsity may slightly differ to ensure the matrix is symmetric.
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code sparsity} is not in the range {@code [0.0, 1.0]}.
     */
    public CooMatrix randomSymmetricCooMatrix(int size, int min, int max, double sparsity) {
        ValidateParameters.ensureInRange(sparsity, 0, 1, "sparsity");
        Shape shape = new Shape(size, size);

        int numEntries = nnzFromSparsity(shape, sparsity);
        numEntries /= 2;

        // Generate half of the random data.
        double[] entries = new double[numEntries];
        RandomArray.randomFill(entries, new RealUniform(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numEntries, 0, shape.getSize(0), 0, shape.getSize(1));

        // Mirror data across diagonal.
        entries = ArrayJoiner.join(entries, entries);
        indices = new int[][]{
                ArrayJoiner.join(indices[0], indices[1]),
                ArrayJoiner.join(indices[1], indices[0])
        };

        CooMatrix randMat = new CooMatrix(shape, entries, indices[0], indices[1]);
        randMat.sortIndices();

        return new CooMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Generates a symmetric {@link CsrMatrix CSR matrix} filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @param min Minimum value in uniform distribution (inclusive).
     * @param max Maximum value in uniform distribution (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     * a value in {@code [0.0, 1.0]}. The true sparsity may slightly differ to ensure the matrix is symmetric.
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code sparsity} is not in the range {@code [0.0, 1.0]}.
     */
    public CsrMatrix randomSymmetricCsrMatrix(int size, int min, int max, double sparsity) {
        return randomSymmetricCooMatrix(size, min, max, sparsity).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have
     * a uniform distribution in the annulus (i.e., washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomCooCMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomCooCMatrix(new Shape(rows, cols), min, max, sparsity);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have
     * a uniform distribution in the annulus (i.e., washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomCooCMatrix(Shape shape, double min, double max, double sparsity) {
        ValidateParameters.ensureInRange(sparsity, 0, 1, "sparsity");
        return randomCooCMatrix(shape, min, max, nnzFromSparsity(shape, sparsity));
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in the annulus (i.e., washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param nnz Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomCooCMatrix(int rows, int cols, double min, double max, int nnz) {
        return randomCooCMatrix(new Shape(rows, cols), min, max, nnz);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in the annulus (i.e., washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Non-zero values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param nnz Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomCooCMatrix(Shape shape, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz);

        Complex128[] entries = new Complex128[nnz];
        RandomArray.randomFill(entries, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(nnz, 0, shape.getSize(0), 0, shape.getSize(1));

        return CooCMatrix.unsafeMake(shape, entries, indices[0], indices[1]);
    }


    /**
     * Generates a Hermitian {@link CooCMatrix complex COO matrix} filled with pseudorandom values uniformly distributed in an annulus
     * centered on the complex plane.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @param min Inner radius of the annulus (inclusive).
     * @param max Outer radius of the annulus (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     * a value in {@code [0.0, 1.0]}. The true sparsity may slightly differ to ensure the matrix is symmetric.
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code sparsity} is not in the range {@code [0.0, 1.0]}.
     */
    public CooCMatrix randomHermitianCooMatrix(int size, int min, int max, double sparsity) {
        ValidateParameters.ensureInRange(sparsity, 0, 1, "sparsity");
        Shape shape = new Shape(size, size);

        int numEntries = nnzFromSparsity(shape, sparsity);
        numEntries /= 2;

        // Generate half of the random data.
        Complex128[] entries = new Complex128[numEntries];
        RandomArray.randomFill(entries, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numEntries, 0, shape.getSize(0), 0, shape.getSize(1));

        // Mirror data across diagonal.
        entries = ArrayJoiner.join(entries, entries);
        indices = new int[][]{
                ArrayJoiner.join(indices[0], indices[1]),
                ArrayJoiner.join(indices[1], indices[0])
        };

        CooCMatrix randMat = new CooCMatrix(shape, entries, indices[0], indices[1]);
        randMat.sortIndices();

        return new CooCMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Generates a Hermitian {@link CsrCMatrix complex CSR matrix} filled with pseudorandom values uniformly distributed in an annulus
     * centered on the complex plane.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @param min Inner radius of the annulus (inclusive).
     * @param max Outer radius of the annulus (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e., the percent of values which are zero. Must be
     * a value in {@code [0.0, 1.0]}. The true sparsity may slightly differ to ensure the matrix is symmetric.
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code sparsity} is not in the range {@code [0.0, 1.0]}.
     */
    public CsrCMatrix randomHermitianCsrMatrix(int size, int min, int max, double sparsity) {
        return randomHermitianCooMatrix(size, min, max, sparsity).toCsr();
    }


    /**
     * Constructs a COO tensor with the specified shape and sparsity filled with pseudo-random values from a uniform
     * distribution in [min, max).
     * @param shape Full shape of the COO tensor.
     * @param min Minimum value of the uniform distribution to sample from (inclusive).
     * @param max Maximum value of the uniform distribution to sample from (exclusive).
     * @param sparsity The sparsity of the COO tensor.
     * @return A sparse COO tensor whose entries are uniformly distributed in [min, max).
     */
    public CooTensor randomCooTensor(Shape shape, double min, double max, double sparsity) {
        return randomCooTensor(shape, min, max, nnzFromSparsity(shape.totalEntriesIntValueExact(), sparsity));
    }


    /**
     * Constructs a COO tensor with the specified shape and number of non-zero entries filled with pseudo-random values from a uniform
     * distribution in [min, max).
     * @param shape Full shape of the COO tensor.
     * @param min Minimum value of the uniform distribution to sample from (inclusive).
     * @param max Maximum value of the uniform distribution to sample from (exclusive).
     * @param nnz The number of non-zero entries in the COO tensor.
     * @return A sparse COO tensor whose entries are uniformly distributed in [min, max).
     */
    public CooTensor randomCooTensor(Shape shape, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz,"nnz");

        double[] data = new double[nnz];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        int[] indices = RAND_ARRAY.randomUniqueIntegers(nnz, 0, shape.totalEntriesIntValueExact());
        int[][] nDIndices = shape.getNdIndices(indices);

        return new CooTensor(shape, data, nDIndices);
    }


    /**
     * Constructs a complex COO tensor with the specified shape and sparsity filled with pseudo-random values uniformly distributed
     * in an annulus centered on the origin of the complex plane.
     * @param shape Full shape of the COO tensor.
     * @param min Minimum radius of the annulus (inclusive).
     * @param max Maximum radius of the annulus (exclusive).
     * @param sparsity The sparsity of the COO tensor.
     * @return A sparse COO tensor whose entries are uniformly distributed in [min, max).
     */
    public CooCTensor randomCooCTensor(Shape shape, double min, double max, double sparsity) {
        return randomCooCTensor(shape, min, max, nnzFromSparsity(shape.totalEntriesIntValueExact(), sparsity));
    }


    /**
     * Constructs a complex COO tensor with the specified shape and number of non-zero entries filled with
     * pseudo-random values uniformly distributed in an annulus centered on the origin of the complex plane.
     * @param shape Full shape of the COO tensor.
     * @param min Minimum radius of the annulus (inclusive).
     * @param max Maximum radius of the annulus (exclusive).
     * @param nnz The number of non-zero entries in the COO tensor.
     * @return A sparse COO tensor whose entries are uniformly distributed in [min, max).
     */
    public CooCTensor randomCooCTensor(Shape shape, double min, double max, int nnz) {
        ValidateParameters.ensureGreaterEq(0, nnz,"nnz");

        Complex128[] data = new Complex128[nnz];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        int[] indices = RAND_ARRAY.randomUniqueIntegers(nnz, 0, shape.totalEntriesIntValueExact());
        int[][] nDIndices = shape.getNdIndices(indices);

        return new CooCTensor(shape, data, nDIndices);
    }
}
