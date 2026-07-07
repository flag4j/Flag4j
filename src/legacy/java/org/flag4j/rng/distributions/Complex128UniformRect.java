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
 * <p>A rectangular uniform distribution on the complex plane.
 *
 * <p>A rectangular uniform distribution on the complex plane is equivalent to combining two 1D uniform distributions over the real
 * and imaginary axes: <span class="latex-replace">U(a, b)</span> <!-- LATEX: \(\mathcal{U}(a, b) \) -->
 * and <span class="latex-replace">U(c, d)</span> <!-- LATEX: \(\mathcal{U}(c, d) \) --> respectively.
 *
 * <p>The PDF of the joint distribution of the real component <span class="latex-replace">x ~ U(a, b)</span>
 * <!-- LATEX: \( x \sim \mathcal{U}(a, b) \) --> and the complex component <span class="latex-replace">y ~ U(c, d)</span>
 * <!-- LATEX: \( y \sim \mathcal{U}(c, d) \) --> for a complex value <span class="latex-inline">z = x + iy</span> is given by:
 * <span class="latex-replace"><pre>
 *     f(z) = { 1 / [(b - a)(d - c)] for a &le; Re(z) &lt; b and c &le; Im(z) &lt; d,  0 otherwise.</pre></span>
 *
 * <!-- LATEX: \[ f(z) = \begin{cases}
 * \frac{1}{(b - a)(d - c)} & \text{for } \Re(z) \in [a, b), \Im(z) \in [c, d) \\
 * 0 & \text{otherwise}.
 * \end{cases} \] -->
 *
 * @see Complex128UniformDisk
 * @see Complex128BiGaussian
 * @see RealUniform
 * @see RealGaussian
 */
public class Complex128UniformRect extends Distribution<Complex128, RandomComplex> {


    /**
     * Lower bound, <span class="latex-inline">a</span>, of the real component of the uniform distribution (inclusive).
     */
    public final double minRe;
    /**
     * Upper bound, <span class="latex-inline">b</span>, of the real component of the uniform distribution (exclusive).
     */
    public final double maxRe;
    /**
     * Lower bound, <span class="latex-inline">c</span>, of the imaginary component of the uniform distribution (inclusive).
     */
    public final double minIm;
    /**
     * Upper bound, <span class="latex-inline">d</span>, of the imaginary component of the uniform distribution (exclusive).
     */
    public final double maxIm;


    /**
     * Constructs a complex rectangular uniform distribution.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param minRe Lower bound of the real component of the uniform distribution (inclusive).
     * @param maxRe Upper bound of the real component of the uniform distribution (exclusive).
     * @param minIm Lower bound of the imaginary component of the uniform distribution (inclusive).
     * @param maxIm Upper bound of the imaginary component of the uniform distribution (exclusive).
     */
    public Complex128UniformRect(RandomComplex rng, double minRe, double maxRe, double minIm, double maxIm) {
        super(rng);

        this.minRe = minRe;
        this.maxRe = maxRe;
        this.minIm = minIm;
        this.maxIm = maxIm;
    }


    /**
     * Randomly samples this complex rectangular uniform distribution.
     *
     * @return A random value distributed according to this complex rectangular uniform distribution.
     */
    @Override
    public Complex128 sample() {
        return rng.randomRectComplex128(minRe, maxRe, minIm, maxIm);
    }
}
