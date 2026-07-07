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
 * <p>A uniform distribution over an annulus (i.e., washer) on the complex plane.
 *
 * <p>A uniform distribution over an annulus is defined over the region between two concentric circles
 * with inner radius <span class="latex-replace">r<sub>inner</sub></span> <!-- LATEX: \( r_{\text{inner}} \) -->
 * and outer radius <span class="latex-replace">r<sub>outer</sub></span> <!-- LATEX: \( r_{\text{outer}} \) -->
 * and centered at <span class="latex-inline">(x, y)</span>.
 *
 * <p>A complex number <span class="latex-inline">z = x + yi + re<sup>i&theta;</sup></span> is sampled from this distribution by
 * drawing its modulus <span class="latex-inline">r</span> from a uniform distribution
 * <span class="latex-replace">U(r<sub>inner</sub>, r<sub>outer</sub>)</span>
 * <!-- LATEX: \( \mathcal{U}\left(r_{\text{inner}}, r_{\text{outer}}\right) \) -->
 * and its argument <span class="latex-inline">&theta;</span> from a uniform distribution
 * <span class="latex-replace">U(0, 2&pi;)</span> <!-- LATEX: \( \mathcal{U}(0, 2\pi) \) -->.
 *
 * <p>The PDF of the joint distribution for a complex value <span class="latex-inline">z = x + yi + re<sup>i&theta;</sup></span> is
 * given by:
 * <span class="latex-replace"><pre>
 *     f(z) = { 1 / [&pi; (r<sub>outer</sub><sup>2</sup> - r<sub>inner</sub><sup>2</sup>)] for r<sub>inner</sub> &le; |z - (x + yi)| &lt; r<sub>outer</sub>, 0 otherwise.</pre></span>
 *
 * <!-- LATEX: \[ f(z) = \begin{cases}
 * \dfrac{1}{\pi \left(r_{\text{outer}}^2 - r_{\text{inner}}^2\right)} & \text{for } r_{\text{inner}} \leq |z - (x + yi)| < r_{\text{outer}} \\
 * 0 & \text{otherwise}.
 * \end{cases} \] -->
 *
 * <p>This distribution ensures that points are uniformly distributed in within the annulus, meaning that the density function
 * accounts for the increasing circumference as <span class="latex-inline">r</span> increases avoiding "bunching" of points close
 * to the inner radius.
 *
 * @see Complex128UniformRect
 * @see Complex128BiGaussian
 * @see RealUniform
 * @see RealGaussian
 */
public class Complex128UniformDisk extends Distribution<Complex128, RandomComplex> {

    /**
     * Inner radius, <span class="latex-replace">r<sub>inner</sub></span>
     * <!-- LATEX \(r_{\text{inner}}\) -->, of the annulus (inclusive).
     */
    public final double min;
    /**
     * Outer radius, <span class="latex-replace">r<sub>outer</sub></span>
     * <!-- LATEX \(r_{\text{outer}}\) -->, of the annulus (exclusive).
     */
    public final double max;
    /**
     * Real value of the center, <span class="latex-replace">x</span>, of the annulus in the complex plane.
     */
    public final double centerRe;
    /**
     * Imaginary value of the center, <span class="latex-replace">y</span>, of the annulus in the complex plane.
     */
    public final double centerIm;
    /**
     * Amount to shift distribution by to center at ({@code centerRe}, {@code centerIm}).
     */
    private final Complex128 shift;


    /**
     * Validates parameters for the distribution.
     */
    private void validateParameters() {
        if(min < 0) {
            throw new IllegalArgumentException("Inner radius must be non-negative but got min=" + min + ".");
        }
        if(max <= min) {
            throw new IllegalArgumentException("Outer radius must be larger than inner radius but got min="
                    + min + " and max=" + max + "."
            );
        }
    }


    /**
     * Constructs a complex annular uniform distribution centered at the origin of the complex plane.
     * To specify the distribution as a disk, set {@code min=0.0}.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param min Inner radius of the annulus (inclusive).
     * @param max Outer radius of the annulus (exclusive).
     */
    public Complex128UniformDisk(RandomComplex rng, double min, double max) {
        super(rng);

        this.min = min;
        this.max = max;
        this.centerRe = 0;
        this.centerIm = 0;
        shift = new Complex128(centerRe, centerIm);

        // Ensure all parameters are valid.
        validateParameters();
    }


    /**
     * Constructs a complex annular uniform distribution with specified center on the complex plane.
     * To specify the distribution as a disk, set {@code min=0.0}.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param min Inner radius of the annulus (inclusive).
     * @param max Outer radius of the annulus (exclusive).
     * @param centerRe Real value of the center of the annulus in the complex plane.
     * @param centerIm Imaginary value of the center of the annulus in the complex plane.
     */
    public Complex128UniformDisk(RandomComplex rng, double min, double max, double centerRe, double centerIm) {
        super(rng);

        this.min = min;
        this.max = max;
        this.centerRe = centerRe;
        this.centerIm = centerIm;
        shift = new Complex128(centerRe, centerIm);

        // Ensure all parameters are valid.
        validateParameters();
    }


    /**
     * Randomly samples this complex annular uniform distribution centered at ({@link #centerRe}, {@link #centerIm})
     * with inner and outer radius {@link #min} and {@link #max} respectively.
     *
     * @return A random value distributed according to this complex annular uniform distribution.
     */
    @Override
    public Complex128 sample() {
        // No shift needed if distribution is centered at origin.
        if(centerIm == 0 && centerRe == 0)
            return rng.randomComplex128(min, max);
        else
            return rng.randomComplex128(min, max).add(shift);
    }
}
