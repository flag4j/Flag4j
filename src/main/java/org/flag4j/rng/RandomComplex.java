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

import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.Complex64;

import java.util.Random;


/**
 * <p>The {@code RandomComplex} class provides methods to generate pseudorandom complex numbers under a variety of
 * distributions and constraints. This class supports both {@link Complex128} and {@link Complex64}.
 *
 * <p>Distributions supported by this class include:
 * <ul>
 *     <li>Uniformly distributed random complex numbers on the unit disk, within annular regions, or rectangular regions
 *         of the complex plane.</li>
 *     <li>Gaussian distributed random complex numbers, including standard circular, elliptical, and
 *         bivariate Gaussian distributions.</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * RandomComplex randomComplex = new RandomComplex();
 *
 * // Generate a random complex number with a magnitude between 0 and 1.
 * Complex128 randomOnDisk = randomComplex.randomComplex128();
 *
 * // Generate a random complex number with a magnitude between 1 and 2.
 * Complex128 randomInAnnulus = randomComplex.randomComplex128(1, 2);
 *
 * // Generate a Gaussian-distributed random complex number centered at (0, 0)
 * //   with standard distributions of 1 along both real and imaginary axis and
 * //   a correlation coefficient between real and imaginary values of 0.5.
 * Complex64 gaussianComplex = randomComplex.randnComplex64(0, 1, 0, 1, 0.5f);
 * }</pre>
 *
 * <p>Note: This class extends {@link Random}. As such instances of {@code RandomComplex} are threadsafe.
 * However, the concurrent use of the same {@code RandomComplex}
 * instance across threads may encounter contention and consequent poor performance.
 *
 * @see java.util.Random
 * @see Complex128
 * @see Complex64
 */
public class RandomComplex extends Random {


    /**
     * Constructs a complex value random number generator. The seed of the random number generator will be set to
     * {@link RandomState#getGlobalSeed()}
     */
    public RandomComplex() {
        super(RandomState.getGlobalSeed());
    }


    /**
     * Constructs a complex value random number generator. Use this constructor with a seed value for reproducible
     * results.
     * @param seed Seed for this random number generator.
     */
    public RandomComplex(long seed) {
        super(seed);
    }


    /**
     * Generates a pseudorandom complex number which is uniformly distributed on the unit disk within the complex plane.
     * @return A pseudorandom complex number uniformly distributed on the unit disk within the complex plane.
     */
    public Complex128 randomComplex128() {
        return randomComplex128(Math.sqrt(nextDouble()));
    }


    /**
     * Generates pseudorandom complex number with specified magnitude which is uniformly distributed around the origin.
     * That is, a pseudorandom complex number which is uniformly distributed around the circle in the complex plane
     * with radius equal to the specified magnitude.
     * @param mag Magnitude of the pseudorandom complex number to generate. Must be non-negative.
     * @return pseudorandom complex number uniformly distributed around the origin with the specified magnitude.
     * @throws IllegalArgumentException If the magnitude is negative.
     */
    public Complex128 randomComplex128(double mag) {
        if(mag < 0) {
            throw new IllegalArgumentException("Magnitude must be non-negative but got " + mag);
        }

        // Simply pick a uniformly random angle in [0, 2pi) radians.
        double theta = 2*Math.PI*nextDouble();

        // Convert to rectangular coordinates.
        return Complex128.fromPolar(mag, theta);
    }


    /**
     * Generates a pseudorandom complex number with magnitude in {@code [min, max)} which is uniformly distributed in
     * the annulus (i.e., washer) with minimum and maximum radii equal to {@code min} and {@code max} respectively.
     *
     * @param min Minimum value for random number. Must be non-negative.
     * @param max Maximum value for random number. Must be larger than or equal to min.
     * @return A pseudorandom complex number with magnitude uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is not positive or if {@code max} is less than {@code min}.
     */
    public Complex128 randomComplex128(double min, double max) {
        if(min < 0) {
            throw new IllegalArgumentException("Min value must be non-negative but got " + min + ".");
        }

        if(min > max) {
            throw new IllegalArgumentException("Max value must be greater than or equal to min but got min=" +
                    min + " and max=" + max + ".");
        }

        double minSq = min*min;

        return randomComplex128(Math.sqrt(nextDouble()*(max*max - minSq) + minSq));
    }


    /**
     * Generates a pseudorandom complex number distributed uniformly in the unit square on the complex plane centered at
     * <span class="latex-inline">(0, 0)</span>.
     * @return A pseudorandom complex number distributed uniformly in the unit square on the complex plane centered at
     * <span class="latex-inline">(0, 0)</span>.
     * @see #randomRectComplex128(double, double, double, double)
     */
    public Complex128 randomRectComplex128() {
        return new Complex128(nextDouble(), nextDouble());
    }


    /**
     * <p>Generates a pseudorandom complex number distributed uniformly in a rectangle on the complex plane. The bounds of the
     * rectangle along each axis are specified by {@code minRe}, {@code maxRe}, {@code minIm}, and {@code maxIm}.
     *
     * @param minRe Minimum value along the real axis (inclusive).
     * @param maxRe Maximum value along the real axis (exclusive).
     * @param minIm Minimum value along the imaginary axis (inclusive).
     * @param maxIm Maximum value along the imaginary axis (exclusive).
     * @return A pseudorandom complex number distributed uniformly in a rectangle on the complex plane.
     */
    public Complex128 randomRectComplex128(double minRe, double maxRe, double minIm, double maxIm) {
        return new Complex128(nextDouble(minRe, maxRe) + minRe, nextDouble(minIm, maxIm));
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a 2D circular standard Gaussian (normal) distribution on the complex
     * plain. The distribution is centered at <span class="latex-inline">(0, 0)</span>
     * with standard deviation <span class="latex-inline">&sigma;=1</span> along both the real and imaginary axes
     * of the complex plane.
     *
     * <p>The 2D circular standard Gaussian distribution centered at <span class="latex-inline">(x<sub>0</sub>, x<sub>0</sub>)</span>
     * with standard deviation along both axes, <span class="latex-inline">&sigma;</span>, is defined as:
     * <span class="latex-replace"><pre>
     *     f(x, y) = 1/(2&pi;) exp[ -1/2 (x<sup>2</sup> + y<sup>2</sup>) ]</pre></span>
     *
     * <!-- LATEX: \[
     * f(x, y) = \frac{1}{2\pi} \exp\left[-\frac{1}{2}(x_2 + y_2)\right]
     * \] -->
     *
     * @param mean Mean of distribution.
     * @param std Standard deviation of distribution. Must be greater than zero.
     * @return A pseudorandom complex number sampled from a 2D circular standard Gaussian (normal) distribution on the complex plain.
     *
     * @see #randnComplex128(double, double)
     * @see #randnComplex128(double, double, double, double)
     * @see #randnComplex128(double, double, double, double, double)
     */
    public Complex128 randnComplex128() {
        return randnComplex128(0, 1, 0, 1);
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a 2D circular Gaussian (normal) distribution on the complex plain.
     * The distribution is centered at ({@code mean}, {@code mean}) with standard deviation {@code std} along
     * both the real and imaginary axes of the complex plane.
     *
     * <p>The 2D circular Gaussian distribution centered at <span class="latex-inline">(x<sub>0</sub>, x<sub>0</sub>)</span>
     * with standard deviations along each axis, <span class="latex-inline">&sigma;</span>, is defined as:
     * <span class="latex-replace"><pre>
     *     f(x, y) = 1/(2&pi;&sigma;<sup>2</sup>) exp[ -1/(2&sigma;<sup>2</sup>) ((x - x<sub>0</sub>)<sup>2</sup> + (y - y<sub>0</sub>)<sup>2</sup>) ]</pre>
     * </span>
     *
     * <!-- LATEX: \[
     * f(x, y) = \frac{1}{2\pi\sigma^2} \exp\left[-\frac{1}{2\sigma^2}\left(\left(x_2 - x_0\right)^2
     * + \left(y - y_0\right)^2\right)\right]
     * \] -->
     *
     * @param mean Mean of distribution.
     * @param std Standard deviation of distribution. Must be greater than zero.
     * @return A pseudorandom complex number sampled from a 2D circular Gaussian (normal) distribution on the complex plain.
     *
     * @see #randnComplex128()
     * @see #randnComplex128(double, double, double, double)
     * @see #randnComplex128(double, double, double, double, double)
     */
    public Complex128 randnComplex128(double mean, double std) {
        return randnComplex128(mean, std, mean, std);
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a 2D elliptical Gaussian (normal) distribution on the complex plain.
     * The distribution is centered at ({@code meanRe}, {@code meanIm}) with standard deviations {@code stdRe} and {@code stdIm} along
     * the real and imaginary axes of the complex plane respectively.
     *
     * <p>The 2D Gaussian distribution centered at (x<sub>0</sub>, y<sub>0</sub>) with standard deviations along each
     * axis, &sigma;<sub>X</sub> and &sigma;<sub>Y</sub>, is defined as:
     * <span class="latex-replace"><pre>
     *     f(x, y) = 1/(2&pi;&sigma;<sub>X</sub>&sigma;<sub>Y</sub>) * exp[ -(x - x<sub>0</sub>)<sup>2</sup>/2&sigma;<sub>X</sub><sup>2</sup> - (y - y<sub>0</sub>)<sup>2</sup>/2&sigma;<sub>Y</sub><sup>2</sup>]</pre>
     * </span>
     *
     * <!-- LATEX: \[
     * f(x, y) = \frac{1}{2\pi\sigma_x \sigma_y} \exp\left[-\left(\frac{x_2 - x_0}{\sigma_x}\right)^2 - \left(\frac{y - y_0}{
     * \sigma_y}\right)^2\right]
     * \] -->
     *
     * @param meanRe Mean of distribution along real axis.
     * @param stdRe Standard deviation of distributions along real axis. Must be greater than zero.
     * @param meanIm Mean of distribution along imaginary axis.
     * @param stdIm Standard deviation of distributions along imaginary axis. Must be greater than zero.
     * @return A pseudorandom complex number sampled from a 2D elliptical Gaussian (normal) distribution on the complex plain.
     *
     * @see #randnComplex128(double, double)
     * @see #randnComplex128()
     * @see #randnComplex128(double, double, double, double, double)
     */
    public Complex128 randnComplex128(double meanRe, double stdRe, double meanIm, double stdIm) {
        return new Complex128(nextGaussian(meanRe, stdRe), nextGaussian(meanIm, stdIm));
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a bivariate Gaussian (normal) distribution on the complex plain.
     * The distribution is centered at ({@code meanRe}, {@code meanIm}) with standard deviations {@code stdRe} and {@code stdIm} along
     * the real and imaginary axes of the complex plane respectively. Further, {@code corrCoeff} is the correlation coefficient
     * between the real and imaginary values.
     *
     * <p>The covariance matrix, <span class="latex-inline"><b>&Sigma;</b></span>, of such a distribution is expressed as:
     * <span class="latex-replace"><pre>
     *   <b>&Sigma;</b> = [  &sigma;<sub>x</sub><sup>2</sup>    &rho;&sigma;<sub>x</sub>&sigma;<sub>y</sub> ]
     *       [ &rho;&sigma;<sub>x</sub>&sigma;<sub>y</sub>     &sigma;<sub>y</sub><sup>2</sup> ]</pre></span>
     * <!-- LATEX: \[
     * \mathbf{\Sigma} = \begin{bmatrix}
     * \sigma_x^2 & \rho\sigma_x\sigma_y \\
     * \rho\sigma_x\sigma_y & \sigma_y^2
     * \end{bmatrix}
     * \] -->
     *
     * where <span class="latex-inline">&rho;</span> is the correlation coefficient,
     * <span class="latex-inline">&sigma;<sub>x</sub></span> is the standard deviation
     * along the real axis, and <span class="latex-inline">&sigma;<sub>y</sub></span>
     * is the standard deviation along the imaginary axis.
     *
     * Let <span class="latex-replace"><b>&mu;</b> = [x<sub>0</sub>  y<sub>0</sub>]<sup>T</sup></span>
     * <!-- LATEX: \( \pmb{\mu} = \begin{bmatrix} x_0 & y_0 \end{bmatrix}^T \) -->
     * be the mean column vector and <span class="latex-replace"><b>z</b> = [x  y]<sup>T</sup></span>
     * <!-- LATEX: \( \mathbf{z} = \begin{bmatrix} x & y \end{bmatrix}^T \) --> also be a
     * column vector. Then, the bivariate Gaussian distribution may be expressed as:
     * <span class="latex-replace"><pre>
     *     f(<b>z</b>) = 1/(2 &pi; det(<b>&Sigma;</b>)<sup>1/2</sup>) exp[-1/2 (<b>z</b> - <b>&mu;</b>)<sup>T</sup> <b>&Sigma;</b><sup>-1</sup> (<b>z</b> - <b>&mu;</b>)]</pre>
     * </span>
     *
     * <!-- LATEX: \[
     * f(\mathbf{z}) = \frac{1}{2\pi \sqrt{\det(\mathbf{\Sigma})}} \exp\left[ -\frac{1}{2} (\mathbf{z} - \pmb{\mu})^T
     * \mathbf{\Sigma}^{-1}(\mathbf{z} - \pmb{\mu})\right]
     * \] -->
     *
     * @param meanRe Mean of distribution along real axis.
     * @param stdRe Standard deviation of distributions along real axis. Must be greater than zero.
     * @param meanIm Mean of distribution along imaginary axis.
     * @param stdIm Standard deviation of distributions along imaginary axis. Must be greater than zero.
     * @param corrCoeff Correlation coefficient between real and imaginary values in the distribution. Must satisfy
     * {@code corrCoeff > -1 && corrCoeff < 1}
     * @return A pseudorandom complex number sampled from a bivariate Gaussian (normal) distribution on the complex plain.
     * @throws IllegalArgumentException If {@code corrCoeff <= -1 || corrCoeff >= 1} or {@code stdRe < 0.0 || stdIm < 0.0}.
     *
     * @see #randnComplex128(double, double)
     * @see #randnComplex128(double, double, double, double)
     * @see #randnComplex128()
     */
    public Complex128 randnComplex128(double meanRe, double stdRe, double meanIm, double stdIm, double corrCoeff) {
        if (stdRe < 0.0 || stdIm < 0.0) {
            throw new IllegalArgumentException("standard deviations must be non-negative " +
                    "but got stdRe=" + stdRe + " and stdIm=" + stdIm + ".");
        }
        if(corrCoeff <= -1.0 || corrCoeff >= 1.0)
            throw new IllegalArgumentException("Correlation coefficient must be in range (-1, 1) but got: " + corrCoeff + ".");

        // Unrolled Cholesky decomposition of the 2x2 covariance matrix.
        double l11 = Math.sqrt(stdRe*stdRe);
        double l21 = corrCoeff*stdRe*stdIm / l11;
        double l22 = Math.sqrt(stdIm*stdIm - l21 * l21);

        // Generate two independent standard normal samples.
        double z1 = nextGaussian();
        double z2 = nextGaussian();

        // Apply unrolled transformation: [x1, x2] = L * [z1, z2]
        double re = meanRe + l11 * z1;
        double im = meanIm + l21 * z1 + l22 * z2;

        return new Complex128(re, im);
    }
    
    
    // --------------------------------------------------------------------------------------------------------------------------


    /**
     * Generates a pseudorandom complex number which is uniformly distributed on the unit disk within the complex plane.
     * @return A pseudorandom complex number uniformly distributed on the unit disk within the complex plane.
     */
    public Complex64 randomComplex64() {
        return randomComplex64((float) Math.sqrt(nextFloat()));
    }


    /**
     * Generates pseudorandom complex number with specified magnitude which is uniformly distributed around the origin.
     * That is, a pseudorandom complex number which is uniformly distributed around the circle in the complex plane
     * with radius equal to the specified magnitude.
     * @param mag Magnitude of the pseudorandom complex number to generate. Must be non-negative.
     * @return pseudorandom complex number uniformly distributed around the origin with the specified magnitude.
     * @throws IllegalArgumentException If the magnitude is negative.
     */
    public Complex64 randomComplex64(float mag) {
        if(mag < 0)
            throw new IllegalArgumentException("Magnitude must be non-negative but got " + mag);

        // Simply pick a uniformly random angle in [0, 2pi) radians.
        float theta = 2f*((float) Math.PI)*nextFloat();

        // Convert to rectangular coordinates.
        return Complex64.fromPolar(mag, theta);
    }


    /**
     * Generates a pseudorandom complex number with magnitude in {@code [min, max)} which is uniformly distributed in
     * the annulus (i.e., washer) with minimum and maximum radii equal to {@code min} and {@code max} respectively.
     *
     * @param min Minimum value for random number. Must be non-negative.
     * @param max Maximum value for random number. Must be larger than or equal to min.
     * @return A pseudorandom complex number with magnitude uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is not positive or if {@code max} is less than {@code min}.
     */
    public Complex64 randomComplex64(float min, float max) {
        if(min < 0) {
            throw new IllegalArgumentException("Min value must be non-negative but got " + min + ".");
        }

        if(min > max) {
            throw new IllegalArgumentException("Max value must be greater than or equal to min but got min=" +
                    min + " and max=" + max + ".");
        }

        return randomComplex64((float) Math.sqrt(nextFloat()*(max*max - min*min) + min*min));
    }



    /**
     * Generates a pseudorandom complex number distributed uniformly in the unit square on the complex plane centered at (0.0, 0.0).
     * @return A pseudorandom complex number distributed uniformly in the unit square on the complex plane centered at (0.0, 0.0).
     * @see #randomCartComplex64(float, float, float, float)
     */
    public Complex64 randomCartComplex64() {
        return new Complex64(nextFloat(), nextFloat());
    }


    /**
     * <p>Generates a pseudorandom complex number distributed uniformly in a rectangle on the complex plane. The bounds of the
     * rectangle along each axis are specified by {@code minRe}, {@code maxRe}, {@code minIm}, and {@code maxIm}.
     *
     * @param minRe Minimum value along the real axis (inclusive).
     * @param maxRe Maximum value along the real axis (exclusive).
     * @param minIm Minimum value along the imaginary axis (inclusive).
     * @param maxIm Maximum value along the imaginary axis (exclusive).
     * @return A pseudorandom complex number distributed uniformly in a rectangle on the complex plane.
     */
    public Complex64 randomCartComplex64(float minRe, float maxRe, float minIm, float maxIm) {
        return new Complex64(nextFloat(minRe, maxRe) + minRe, nextFloat(minIm, maxIm));
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a 2D circular standard Gaussian (normal) distribution on the complex
     * plain. The distribution is centered at <span class="latex-inline">(0, 0)</span>
     * with standard deviation <span class="latex-inline">&sigma;=1</span> along both the real and imaginary axes
     * of the complex plane.
     *
     * <p>The 2D circular standard Gaussian distribution centered at <span class="latex-inline">(x<sub>0</sub>, x<sub>0</sub>)</span>
     * with standard deviation along both axes, <span class="latex-inline">&sigma;</span>, is defined as:
     * <span class="latex-replace"><pre>
     *     f(x, y) = 1/(2&pi;) exp[ -1/2 (x<sup>2</sup> + y<sup>2</sup>) ]</pre></span>
     *
     * <!-- LATEX: \[
     * f(x, y) = \frac{1}{2\pi} \exp\left[-\frac{1}{2}(x_2 + y_2)\right]
     * \] -->
     *
     * @param mean Mean of distribution.
     * @param std Standard deviation of distribution. Must be greater than zero.
     * @return A pseudorandom complex number sampled from a 2D circular standard Gaussian (normal) distribution on the complex plain.
     *
     * @see #randnComplex64(float, float)
     * @see #randnComplex64(float, float, float, float)
     * @see #randnComplex64(float, float, float, float, float)
     */
    public Complex64 randnComplex64() {
        return randnComplex64(0f, 1f, 0f, 1f);
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a 2D circular Gaussian (normal) distribution on the complex plain.
     * The distribution is centered at ({@code mean}, {@code mean}) with standard deviation {@code std} along
     * both the real and imaginary axes of the complex plane.
     *
     * <p>The 2D circular Gaussian distribution centered at <span class="latex-inline">(x<sub>0</sub>, x<sub>0</sub>)</span>
     * with standard deviations along each axis, <span class="latex-inline">&sigma;</span>, is defined as:
     * <span class="latex-replace"><pre>
     *     f(x, y) = 1/(2&pi;&sigma;<sup>2</sup>) exp[ -1/(2&sigma;<sup>2</sup>) ((x - x<sub>0</sub>)<sup>2</sup> + (y - y<sub>0</sub>)<sup>2</sup>) ]</pre>
     * </span>
     *
     * <!-- LATEX: \[
     * f(x, y) = \frac{1}{2\pi\sigma^2} \exp\left[-\frac{1}{2\sigma^2}\left(\left(x_2 - x_0\right)^2
     * + \left(y - y_0\right)^2\right)\right]
     * \] -->
     *
     * @param mean Mean of distribution.
     * @param std Standard deviation of distribution. Must be greater than zero.
     * @return A pseudorandom complex number sampled from a 2D circular Gaussian (normal) distribution on the complex plain.
     *
     * @see #randnComplex64()
     * @see #randnComplex64(float, float, float, float)
     * @see #randnComplex64(float, float, float, float, float)
     */
    public Complex64 randnComplex64(float mean, float std) {
        return randnComplex64(mean, std, mean, std);
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a 2D elliptical Gaussian (normal) distribution on the complex plain.
     * The distribution is centered at ({@code meanRe}, {@code meanIm}) with standard deviations {@code stdRe} and {@code stdIm} along
     * the real and imaginary axes of the complex plane respectively.
     *
     * <p>The 2D Gaussian distribution centered at (x<sub>0</sub>, y<sub>0</sub>) with standard deviations along each
     * axis, &sigma;<sub>X</sub> and &sigma;<sub>Y</sub>, is defined as:
     * <span class="latex-replace"><pre>
     *     f(x, y) = 1/(2&pi;&sigma;<sub>X</sub>&sigma;<sub>Y</sub>) * exp[ -(x - x<sub>0</sub>)<sup>2</sup>/2&sigma;<sub>X</sub><sup>2</sup> - (y - y<sub>0</sub>)<sup>2</sup>/2&sigma;<sub>Y</sub><sup>2</sup>]</pre>
     * </span>
     *
     * <!-- LATEX: \[
     * f(x, y) = \frac{1}{2\pi\sigma_x \sigma_y} \exp\left[-\left(\frac{x_2 - x_0}{\sigma_x}\right)^2 - \left(\frac{y - y_0}{
     * \sigma_y}\right)^2\right]
     * \] -->
     *
     * @param meanRe Mean of distribution along real axis.
     * @param stdRe Standard deviation of distributions along real axis. Must be greater than zero.
     * @param meanIm Mean of distribution along imaginary axis.
     * @param stdIm Standard deviation of distributions along imaginary axis. Must be greater than zero.
     * @return A pseudorandom complex number sampled from a 2D elliptical Gaussian (normal) distribution on the complex plain.
     *
     * @see #randnComplex64(float, float)
     * @see #randnComplex64()
     * @see #randnComplex64(float, float, float, float, float)
     */
    public Complex64 randnComplex64(float meanRe, float stdRe, float meanIm, float stdIm) {
        return new Complex64((float) nextGaussian(meanRe, stdRe), (float) nextGaussian(meanIm, stdIm));
    }


    /**
     * <p>Generates a pseudorandom complex number sampled from a bivariate Gaussian (normal) distribution on the complex plain.
     * The distribution is centered at ({@code meanRe}, {@code meanIm}) with standard deviations {@code stdRe} and {@code stdIm} along
     * the real and imaginary axes of the complex plane respectively. Further, {@code corrCoeff} is the correlation coefficient
     * between the real and imaginary values.
     *
     * <p>The covariance matrix, <span class="latex-inline"><b>&Sigma;</b></span>, of such a distribution is expressed as:
     * <span class="latex-replace"><pre>
     *   <b>&Sigma;</b> = [  &sigma;<sub>x</sub><sup>2</sup>    &rho;&sigma;<sub>x</sub>&sigma;<sub>y</sub> ]
     *       [ &rho;&sigma;<sub>x</sub>&sigma;<sub>y</sub>     &sigma;<sub>y</sub><sup>2</sup> ]</pre></span>
     * <!-- LATEX: \[
     * \mathbf{\Sigma} = \begin{bmatrix}
     * \sigma_x^2 & \rho\sigma_x\sigma_y \\
     * \rho\sigma_x\sigma_y & \sigma_y^2
     * \end{bmatrix}
     * \] -->
     *
     * where <span class="latex-inline">&rho;</span> is the correlation coefficient,
     * <span class="latex-inline">&sigma;<sub>x</sub></span> is the standard deviation
     * along the real axis, and <span class="latex-inline">&sigma;<sub>y</sub></span>
     * is the standard deviation along the imaginary axis.
     *
     * Let <span class="latex-replace"><b>&mu;</b> = [x<sub>0</sub>  y<sub>0</sub>]<sup>T</sup></span>
     * <!-- LATEX: \( \pmb{\mu} = \begin{bmatrix} x_0 & y_0 \end{bmatrix}^T \) -->
     * be the mean column vector and <span class="latex-replace"><b>z</b> = [x  y]<sup>T</sup></span>
     * <!-- LATEX: \( \mathbf{z} = \begin{bmatrix} x & y \end{bmatrix}^T \) --> also be a
     * column vector. Then, the bivariate Gaussian distribution may be expressed as:
     * <span class="latex-replace"><pre>
     *     f(<b>z</b>) = 1/(2 &pi; det(<b>&Sigma;</b>)<sup>1/2</sup>) exp[-1/2 (<b>z</b> - <b>&mu;</b>)<sup>T</sup> <b>&Sigma;</b><sup>-1</sup> (<b>z</b> - <b>&mu;</b>)]</pre>
     * </span>
     *
     * <!-- LATEX: \[
     * f(\mathbf{z}) = \frac{1}{2\pi \sqrt{\det(\mathbf{\Sigma})}} \exp\left[ -\frac{1}{2} (\mathbf{z} - \pmb{\mu})^T
     * \mathbf{\Sigma}^{-1}(\mathbf{z} - \pmb{\mu})\right]
     * \] -->
     *
     * @param meanRe Mean of distribution along real axis.
     * @param stdRe Standard deviation of distributions along real axis. Must be greater than zero.
     * @param meanIm Mean of distribution along imaginary axis.
     * @param stdIm Standard deviation of distributions along imaginary axis. Must be greater than zero.
     * @param corrCoeff Correlation coefficient between real and imaginary values in the distribution. Must satisfy
     * {@code corrCoeff > -1 && corrCoeff < 1}
     * @return A pseudorandom complex number sampled from a bivariate Gaussian (normal) distribution on the complex plain.
     * @throws IllegalArgumentException If {@code corrCoeff <= -1 || corrCoeff >= 1} or {@code stdRe < 0.0 || stdIm < 0.0}.
     *
     * @see #randnComplex64(float, float)
     * @see #randnComplex64(float, float, float, float)
     * @see #randnComplex64()
     */
    public Complex64 randnComplex64(float meanRe, float stdRe, float meanIm, float stdIm, float corrCoeff) {
        if (stdRe < 0.0 || stdIm < 0.0) {
            throw new IllegalArgumentException("standard deviations must be non-negative " +
                    "but got stdRe=" + stdRe + " and stdIm=" + stdIm + ".");
        }
        if(corrCoeff <= -1.0 || corrCoeff >= 1.0)
            throw new IllegalArgumentException("Correlation coefficient must be in range (-1, 1) but got: " + corrCoeff + ".");

        // Unrolled Cholesky decomposition of the 2&times;2 covariance matrix.
        float l11 = (float) Math.sqrt(stdRe*stdRe);
        float l21 = corrCoeff*stdRe*stdIm / l11;
        float l22 = (float) Math.sqrt(stdIm*stdIm - l21 * l21);

        // Generate two independent standard normal samples.
        float z1 = (float) nextGaussian();
        float z2 = (float) nextGaussian();

        // Apply unrolled transformation: [x1, x2] = L * [z1, z2]
        float re = meanRe + l11 * z1;
        float im = meanIm + l21 * z1 + l22 * z2;

        return new Complex64(re, im);
    }
}
