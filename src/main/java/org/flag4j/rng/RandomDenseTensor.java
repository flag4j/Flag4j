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
import org.flag4j.arrays.dense.*;
import org.flag4j.linalg.decompositions.qr.ComplexQR;
import org.flag4j.linalg.decompositions.qr.RealQR;
import org.flag4j.numbers.Complex128;
import org.flag4j.rng.distributions.Complex128BiGaussian;
import org.flag4j.rng.distributions.Complex128UniformDisk;
import org.flag4j.rng.distributions.RealGaussian;
import org.flag4j.rng.distributions.RealUniform;


/**
 * A utility class for generating random dense tensors, matrices, and vectors with customizable distributions.
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li>Generate random tensors, matrices, and vectors with real or complex entries.</li>
 *   <li>Support for configurable uniform and Gaussian distributions.</li>
 *   <li>Create specialized matrices, such as symmetric, orthogonal, unitary, triangular, and positive-definite.</li>
 * </ul>
 *
 * <p>This class relies on the {@code RandomComplex} and {@code RandomArray} classes for pseudorandom number
 * generation and array population.
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 *     RandomDenseTensor rtg = new RandomDenseTensor(12345L);
 *
 *     // Uniformly distributed tensor.
 *     Tensor tensor = rtg.randomTensor(new Shape(3, 3, 5), 0.0, 10.0);
 *
 *     // Random unitary matrix.
 *     CMatrix unitaryMatrix = rtg.randomUnitaryMatrix(5);
 *
 *     // Random symmetric positive-definite matrix.
 *     Matrix symmPosDef = rtg.randomSymPosDef(12);
 * }</pre>
 *
 * @see RandomSparseTensor
 * @see RandomComplex
 * @see RandomArray
 */
public class RandomDenseTensor {

    /**
     * Complex pseudorandom number generator.
     */
    private final RandomComplex COMPLEX_RNG;
    /**
     * Generator for random arrays.
     */
    private final RandomArray RAND_ARRAY;

    
    /**
     * Constructs a new pseudorandom tensor generator with the seed set to {@link RandomState#getGlobalSeed()}
     */
    public RandomDenseTensor() {
        COMPLEX_RNG = new RandomComplex(RandomState.getGlobalSeed());
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }
    

    /**
     * Constructs a pseudorandom tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the pseudorandom tensor generator.
     */
    public RandomDenseTensor(long seed) {
        COMPLEX_RNG = new RandomComplex(seed);
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }


    /**
     * Generates a tensor filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Tensor randomTensor(Shape shape) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, 0, 1));
        return new Tensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param shape Shape of the tensor.
     * @param min Minimum value for the uniform distribution.
     * @param max Maximum value for the uniform distribution.
     * @return A tensor filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Tensor randomTensor(Shape shape, double min, double max) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        return new Tensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     */
    public Tensor randnTensor(Shape shape) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealGaussian(COMPLEX_RNG, 0, 1));
        return new Tensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @param shape Shape of the tensor.
     * @param mean Mean of the normal distribution to sample from.
     * @param std Standard deviation of normal distribution to sample from.
     * @return A tensor filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public Tensor randnTensor(Shape shape, double mean, double std) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealGaussian(COMPLEX_RNG, mean, std));
        return new Tensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom complex values uniformly distributed in the unit disk centered at the origin
     * of the complex plane.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom complex values uniformly distributed in the unit disk centered at the origin of
     * the complex plane.
     */
    public CTensor randomCTensor(Shape shape) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, 0, 1));
        return new CTensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom complex values uniformly distributed in an annulus (i.e., washer) centered at the
     * origin of the complex plane.
     * @param shape Shape of the tensor.
     * @param min Inner radius of annulus.
     * @param max Outer radius of annulus.
     * @return A tensor filled with pseudorandom values with magnitudes uniformly distributed in an annulus (i.e., washer) centered
     * at the origin of the complex plane.
     * @throws IllegalArgumentException If {@code min >= max} or {@code min < 0}.
     */
    public CTensor randomCTensor(Shape shape, double min, double max) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        return new CTensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a bivariate standard Gaussian (normal) distribution with mean
     * zero and standard deviation one along both the real and imaginary axes. The correlation coefficient of the distribution will be
     * zero.
     * @param shape Shape of the tensor.
     * @return A pseudorandom values sampled from a bivariate standard Gaussian (normal) distribution with a correlation coefficient
     * of zero.
     * @see RandomComplex#randnComplex128()
     */
    public CTensor randnCTensor(Shape shape) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, 0, 1, 0, 1));
        return new CTensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a bivariate Gaussian (normal) distribution with
     * specified mean and standard deviation along both the real and imaginary axes. The correlation coefficient of the distribution
     * will be zero.
     * @param shape Shape of the tensor.
     * @param mean Mean of the Gaussian distribution.
     * @param std Standard deviation of the Gaussian distribution.
     * @return A tensor filled with pseudorandom values sampled from a bivariate Gaussian distribution with
     * specified mean and standard deviation.
     * @see RandomComplex#randnComplex128(double, double)
     */
    public CTensor randnCTensor(Shape shape, double mean, double std) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, mean, std, mean, std));
        return new CTensor(shape, data);
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a bivariate Gaussian (normal) distribution.
     * @param shape Shape of the tensor.
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     * @param corrCoeff Correlation coefficient of the bivariate distribution.
     * @return A tensor filled with pseudorandom values sampled from a bivariate Gaussian distribution with
     * specified means, standard deviations and correlation coefficient.
     * @see RandomComplex#randnComplex128(double, double, double, double, double)
     */
    public CTensor randnCTensor(Shape shape, double meanRe, double stdRe, double meanIm, double stdIm, double corrCoeff) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, meanRe, stdRe, meanIm, stdIm, corrCoeff));
        return new CTensor(shape, data);
    }


    /**
     * Generates a vector filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param int size of the vector.
     * @return A vector filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Vector randomVector(int size) {
        double[] data = new double[size];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, 0, 1));
        return new Vector(data);
    }


    /**
     * Generates a vector filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param int size of the vector.
     * @param min Minimum value for the uniform distribution.
     * @param max Maximum value for the uniform distribution.
     * @return A vector filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Vector randomVector(int size, double min, double max) {
        double[] data = new double[size];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        return new Vector(data);
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     * @param int size of the vector.
     * @return A vector filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     */
    public Vector randnVector(int size) {
        double[] data = new double[size];
        RandomArray.randomFill(data, new RealGaussian(COMPLEX_RNG, 0, 1));
        return new Vector(data);
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @param int size of the vector.
     * @param mean Mean of the normal distribution to sample from.
     * @param std Standard deviation of normal distribution to sample from.
     * @return A vector filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public Vector randnVector(int size, double mean, double std) {
        double[] data = new double[size];
        RandomArray.randomFill(data, new RealGaussian(COMPLEX_RNG, mean, std));
        return new Vector(data);
    }


    /**
     * Generates a vector filled with pseudorandom complex values uniformly distributed in the unit disk centered at the origin
     * of the complex plane.
     * @param int size of the vector.
     * @return A vector filled with pseudorandom complex values uniformly distributed in the unit disk centered at the origin of
     * the complex plane.
     */
    public CVector randomCVector(int size) {
        Complex128[] data = new Complex128[size];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, 0, 1));
        return new CVector(data);
    }


    /**
     * Generates a vector filled with pseudorandom complex values uniformly distributed in an annulus (i.e., washer) centered at the
     * origin of the complex plane.
     * @param int size of the vector.
     * @param min Inner radius of annulus.
     * @param max Outer radius of annulus.
     * @return A vector filled with pseudorandom values with magnitudes uniformly distributed in an annulus (i.e., washer) centered
     * at the origin of the complex plane.
     * @throws IllegalArgumentException If {@code min >= max} or {@code min < 0}.
     */
    public CVector randomCVector(int size, double min, double max) {
        Complex128[] data = new Complex128[size];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        return new CVector(data);
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a bivariate standard Gaussian (normal) distribution with mean
     * zero and standard deviation one along both the real and imaginary axes. The correlation coefficient of the distribution will be
     * zero.
     * @param int size of the vector.
     * @return A pseudorandom values sampled from a bivariate standard Gaussian (normal) distribution with a correlation coefficient
     * of zero.
     * @see RandomComplex#randnComplex128()
     */
    public CVector randnCVector(int size) {
        Complex128[] data = new Complex128[size];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, 0, 1, 0, 1));
        return new CVector(data);
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a bivariate Gaussian (normal) distribution with
     * specified mean and standard deviation along both the real and imaginary axes. The correlation coefficient of the distribution
     * will be zero.
     * @param int size of the vector.
     * @param mean Mean of the Gaussian distribution.
     * @param std Standard deviation of the Gaussian distribution.
     * @return A vector filled with pseudorandom values sampled from a bivariate Gaussian distribution with
     * specified mean and standard deviation.
     * @see RandomComplex#randnComplex128(double, double)
     */
    public CVector randnCVector(int size, double mean, double std) {
        Complex128[] data = new Complex128[size];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, mean, std, mean, std));
        return new CVector(data);
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a bivariate Gaussian (normal) distribution.
     * @param int size of the vector.
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     * @param corrCoeff Correlation coefficient of the bivariate distribution.
     * @return A vector filled with pseudorandom values sampled from a bivariate Gaussian distribution with
     * specified means, standard deviations and correlation coefficient.
     * @see RandomComplex#randnComplex128(double, double, double, double, double)
     */
    public CVector randnCVector(int size, double meanRe, double stdRe, double meanIm, double stdIm, double corrCoeff) {
        Complex128[] data = new Complex128[size];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, meanRe, stdRe, meanIm, stdIm, corrCoeff));
        return new CVector(data);
    }

    // -----------------------------------------------------
    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the matrix.
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Matrix randomMatrix(Shape shape) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, 0, 1));
        return new Matrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param shape Shape of the matrix.
     * @param min Minimum value for the uniform distribution.
     * @param max Maximum value for the uniform distribution.
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Matrix randomMatrix(Shape shape, double min, double max) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        return new Matrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     * @param shape Shape of the matrix.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     */
    public Matrix randnMatrix(Shape shape) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealGaussian(COMPLEX_RNG, 0, 1));
        return new Matrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @param shape Shape of the matrix.
     * @param mean Mean of the normal distribution to sample from.
     * @param std Standard deviation of normal distribution to sample from.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public Matrix randnMatrix(Shape shape, double mean, double std) {
        double[] data = new double[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new RealGaussian(COMPLEX_RNG, mean, std));
        return new Matrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom complex values uniformly distributed in the unit disk centered at the origin
     * of the complex plane.
     * @param shape Shape of the matrix.
     * @return A matrix filled with pseudorandom complex values uniformly distributed in the unit disk centered at the origin of
     * the complex plane.
     */
    public CMatrix randomCMatrix(Shape shape) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, 0, 1));
        return new CMatrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom complex values uniformly distributed in an annulus (i.e., washer) centered at the
     * origin of the complex plane.
     * @param shape Shape of the matrix.
     * @param min Inner radius of annulus.
     * @param max Outer radius of annulus.
     * @return A matrix filled with pseudorandom values with magnitudes uniformly distributed in an annulus (i.e., washer) centered
     * at the origin of the complex plane.
     * @throws IllegalArgumentException If {@code min >= max} or {@code min < 0}.
     */
    public CMatrix randomCMatrix(Shape shape, double min, double max) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        return new CMatrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a bivariate standard Gaussian (normal) distribution with mean
     * zero and standard deviation one along both the real and imaginary axes. The correlation coefficient of the distribution will be
     * zero.
     * @param shape Shape of the matrix.
     * @return A pseudorandom values sampled from a bivariate standard Gaussian (normal) distribution with a correlation coefficient
     * of zero.
     * @see RandomComplex#randnComplex128()
     */
    public CMatrix randnCMatrix(Shape shape) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, 0, 1, 0, 1));
        return new CMatrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a bivariate Gaussian (normal) distribution with
     * specified mean and standard deviation along both the real and imaginary axes. The correlation coefficient of the distribution
     * will be zero.
     * @param shape Shape of the matrix.
     * @param mean Mean of the Gaussian distribution.
     * @param std Standard deviation of the Gaussian distribution.
     * @return A matrix filled with pseudorandom values sampled from a bivariate Gaussian distribution with
     * specified mean and standard deviation.
     * @see RandomComplex#randnComplex128(double, double)
     */
    public CMatrix randnCMatrix(Shape shape, double mean, double std) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, mean, std, mean, std));
        return new CMatrix(shape, data);
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a bivariate Gaussian (normal) distribution.
     * @param shape Shape of the matrix.
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     * @param corrCoeff Correlation coefficient of the bivariate distribution.
     * @return A matrix filled with pseudorandom values sampled from a bivariate Gaussian distribution with
     * specified means, standard deviations and correlation coefficient.
     * @see RandomComplex#randnComplex128(double, double, double, double, double)
     */
    public CMatrix randnCMatrix(Shape shape, double meanRe, double stdRe, double meanIm, double stdIm, double corrCoeff) {
        Complex128[] data = new Complex128[shape.totalEntriesIntValueExact()];
        RandomArray.randomFill(data, new Complex128BiGaussian(COMPLEX_RNG, meanRe, stdRe, meanIm, stdIm, corrCoeff));
        return new CMatrix(shape, data);
    }
    // -----------------------------------------------------


    /**
     * Generates a symmetric matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Matrix randomSymmetricMatrix(int size) {
        Matrix randMat = new Matrix(size);

        for(int i=0; i<size; i++) {
            for(int j=0; j<i; j++) {
                randMat.data[i*size+j] = COMPLEX_RNG.nextDouble();
                randMat.data[j*size+i] = randMat.data[i*size+j];
            }

            randMat.data[i*(size+1)] = COMPLEX_RNG.nextDouble(); // Diagonal entry
        }

        return randMat;
    }


    /**
     * <p>Gets a pseudorandom orthogonal matrix.
     *
     * <p>The matrix is generated as if by {@link #randomMatrix(Shape) randomMatrix(new Shape(size, size))}.
     * Then, a {@link RealQR QR} decomposition is computed on this pseudorandom matrix and the {@code Q} matrix from this
     * decomposition is returned.
     *
     * @param size Size of the orthogonal matrix.
     * @return A pseudorandom orthogonal matrix.
     */
    public Matrix randomOrthogonalMatrix(int size) {
        Matrix randMat = randomMatrix(new Shape(size, size));
        return new RealQR().decompose(randMat).getQ();
    }


    /**
     * <p>Gets a pseudorandom unitary matrix.
     *
     * <p>The matrix is generated as if by {@link #randomCMatrix(Shape) randomCMatrix(new Shape(size, size))}. Then, a
     * {@link ComplexQR QR} decomposition is computed on this pseudorandom matrix and the {@code Q} matrix from this decomposition
     * is returned.
     * @param size Size of the unitary matrix.
     * @return A pseudorandom unitary matrix.
     */
    public CMatrix randomUnitaryMatrix(int size) {
        CMatrix randMat = randomCMatrix(new Shape(size, size));
        return new ComplexQR().decompose(randMat).getQ();
    }


    /**
     * Gets a pseudorandom upper triangular matrix of the specified {@code size}. The non-zero entries of the matrix are distributed
     * according to a uniform distribution un [{@code min}, {@code max}).
     * @param size Size if the upper triangular matrix.
     * @param min Lower bound of the uniform distribution (inclusive).
     * @param max Upper bound of the uniform distribution (exclusive).
     * @return A pseudorandom upper triangular matrix of the specified size.
     */
    public Matrix randomTriuMatrix(int size, int min, int max) {
        double[] entries = new double[size*size];

        for(int i=0; i<size; i++) {
            int rowOffset = i*size;
            for(int j=i; j<size; j++)
                entries[rowOffset + j] = COMPLEX_RNG.nextDouble(min, max);
        }

        return new Matrix(new Shape(size, size), entries);
    }


    /**
     * Gets a pseudorandom lower triangular matrix of the specified {@code size}. The non-zero entries of the matrix are distributed
     * according to a uniform distribution un [{@code min}, {@code max}).
     * @param size Size if the lower triangular matrix.
     * @param min Lower bound of the uniform distribution (inclusive).
     * @param max Upper bound of the uniform distribution (exclusive).
     * @return A pseudorandom lower triangular matrix of the specified size.
     */
    public Matrix randomTrilMatrix(int size, int min, int max) {
        double[] entries = new double[size*size];
        double maxMin = max-min;

        for(int i=0; i<size; i++) {
            int rowOffset = i*size;
            for(int j=0; j<=i; j++)
                entries[rowOffset + j] = COMPLEX_RNG.nextDouble(min, max);
        }

        return new Matrix(new Shape(size, size), entries);
    }


    /**
     * Generates a random diagonal matrix whose diagonal entries are uniformly distributed in [{@code min}, {@code max}).
     * @param size Size of the diagonal matrix to construct.
     * @param min Lower bound of the uniform distribution (inclusive).
     * @param max Upper bound of the uniform distribution (exclusive).
     * @return A diagonal matrix with diagonal entries are uniformly distributed in [{@code min}, {@code max}).
     */
    public Matrix randomDiagMatrix(int size, int min, int max) {
        double[] data = new double[size];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        return Matrix.diag(data);
    }


    /**
     * Generates a pseudorandom symmetric positive-definite matrix. This is done as if by
     * <pre>{@code
     *     Matrix D = randomDiagMatrix(size, 0, 1);
     *     Matrix Q = randomOrthogonalMatrix(size);
     *     return Q.T().mult(D).mult(Q);}</pre>
     * @param size Size of the symmetric positive-definite matrix to generate.
     * @return A pseudorandom symmetric positive-definite matrix.
     * @see #randomHermPosDefMatrix(int)
     */
    public Matrix randomSymmPosDefMatrix(int size) {
        Matrix D = randomDiagMatrix(size, 0, 1);
        Matrix Q = randomOrthogonalMatrix(size);
        return Q.T().mult(D).mult(Q);
    }


    /**
     * Generates a pseudorandom symmetric positive-definite matrix. This is done as if by
     * <pre>{@code
     *     Matrix D = randomDiagMatrix(size, 0, 1);
     *     CMatrix U = randomUnitaryMatrix(size);
     *     return U.H().mult(D).mult(U);}</pre>
     * @param size Size of the symmetric positive-definite matrix to generate.
     * @return A pseudorandom symmetric positive-definite matrix.
     * @see #randomSymmPosDefMatrix(int)
     */
    public CMatrix randomHermPosDefMatrix(int size) {
        Matrix D = randomDiagMatrix(size, 0, 1);
        CMatrix U = randomUnitaryMatrix(size);
        return U.H().mult(D).mult(U);
    }
}
