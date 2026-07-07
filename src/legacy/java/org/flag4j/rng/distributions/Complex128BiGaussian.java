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

package org.flag4j.rng.distributions;


import org.flag4j.numbers.Complex128;
import org.flag4j.rng.RandomComplex;

/**
 * A 2D bivariate Gaussian distribution on the complex plane.
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
 * column vector. Then, the PDF of the bivariate Gaussian distribution may be expressed as:
 * <span class="latex-replace"><pre>
 *     f(<b>z</b>) = 1/(2 &pi; det(<b>&Sigma;</b>)<sup>1/2</sup>) exp[-1/2 (<b>z</b> - <b>&mu;</b>)<sup>T</sup> <b>&Sigma;</b><sup>-1</sup> (<b>z</b> - <b>&mu;</b>)]</pre>
 * </span>
 *
 * <!-- LATEX: \[
 * f(\mathbf{z}) = \frac{1}{2\pi \sqrt{\det(\mathbf{\Sigma})}} \exp\left[ -\frac{1}{2} (\mathbf{z} - \pmb{\mu})^T
 * \mathbf{\Sigma}^{-1}(\mathbf{z} - \pmb{\mu})\right]
 * \] -->
 *
 * @see Complex128UniformRect
 * @see Complex128UniformDisk
 * @see RealUniform
 * @see RealGaussian
 */
public class Complex128BiGaussian extends Distribution<Complex128, RandomComplex> {

    /**
     * Mean along real axis of complex plane for the distribution.
     */
    private final double meanRe;
    /**
     * Standard deviation along real axis of complex plane for the distribution.
     */
    private final double stdRe;
    /**
     * Mean along imaginary axis of complex plane for the distribution.
     */
    private final double meanIm;
    /**
     * Standard deviation along real axis of complex plane for the distribution.
     */
    private final double stdIm;
    /**
     * corrCoeff Correlation coefficient between real and imaginary values in the distribution.
     */
    private final double corrCoeff;


    /**
     * Validated standard deviations and correlation coefficient are within proper bounds.
     */
    private void validateParameters() {
        if (stdRe < 0.0 || stdIm < 0.0) {
            throw new IllegalArgumentException("standard deviations must be non-negative " +
                    "but got stdRe=" + stdRe + " and stdIm=" + stdIm + ".");
        }
        if(corrCoeff <= -1.0 || corrCoeff >= 1.0)
            throw new IllegalArgumentException("Correlation coefficient must be in range (-1, 1) but got: " + corrCoeff + ".");
    }


    /**
     * Constructs 2D bivariate Gaussian distribution on the complex plane with a correlation coefficient of zero.
     * To specify a correlation coefficient use {@link #Complex128BiGaussian(RandomComplex, double, double, double, double, double)}.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     */
    public Complex128BiGaussian(RandomComplex rng, double meanRe, double stdRe, double meanIm, double stdIm) {
        super(rng);

        this.meanRe = meanRe;
        this.stdRe = stdRe;
        this.meanIm = meanIm;
        this.stdIm = stdIm;
        this.corrCoeff = 0.0;
    }


    /**
     * Constructs 2D bivariate Gaussian distribution on the complex plane with a specified correlation coefficient.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     * @param corrCoeff Correlation coefficient of the bivariate distribution.
     */
    public Complex128BiGaussian(RandomComplex rng, double meanRe, double stdRe, double meanIm, double stdIm, double corrCoeff) {
        super(rng);

        this.meanRe = meanRe;
        this.stdRe = stdRe;
        this.meanIm = meanIm;
        this.stdIm = stdIm;
        this.corrCoeff = corrCoeff;
    }


    /**
     * Randomly samples this distribution.
     *
     * @return A random value distributed according to this distribution.
     */
    @Override
    public Complex128 sample() {
        // Avoid extra computations when there is no correlation between real and complex components.
        if(corrCoeff == 0.0)
            return rng.randnComplex128(meanRe, stdRe, meanIm, stdIm);
        else
            return rng.randnComplex128(meanRe, stdRe, meanIm, stdIm, corrCoeff);
    }
}
