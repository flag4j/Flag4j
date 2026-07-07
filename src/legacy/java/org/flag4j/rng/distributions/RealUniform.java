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

import java.util.Random;


/**
 * <p>A 1D real uniform distribution <span class="latex-replace">U(a, b)</span> <!-- LATEX: \(\mathcal{U}(a, b)\) -->.
 *
 * <p>The PDF of the uniform distribution <span class="latex-replace">U(a, b)</span> <!-- LATEX: \(\mathcal{U}(a, b)\) -->
 *     over the half open interval <span class="latex-inline">[a, b)</span> where <span class="latex-inline">a &lt; b</span> is given
 *     by:
 * <span class="latex-replace"><pre>
 *     f(x) = { 1 / (b - a) for a &le; x &lt; b,  0 for x &lt; a or x &gt; b.</pre></span>
 *
 * <!-- LATEX: \[ f(x) = \begin{cases}
 * \frac{1}{b - a} & \text{for } x \in [a, b), \\
 * 0 & \text{otherwise}.
 * \end{cases} \] -->
 *
 * @see RealGaussian
 * @see Complex128UniformRect
 * @see Complex128UniformDisk
 */
public class RealUniform extends Distribution<Double, Random> {

    /**
     * Lower bound, <span class="latex-inline">a</span>, of the uniform distribution (inclusive).
     */
    public final double min;
    /**
     * Upper bound, <span class="latex-inline">b</span>, of the uniform distribution (exclusive).
     */
    public final double max;


    /**
     * Constructs a real uniform distribution.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param min Lower bound of the uniform distribution (inclusive).
     * @param max Upper bound of the uniform distribution (exclusive).
     */
    public RealUniform(Random rng, double min, double max) {
        super(rng);

        this.min = min;
        this.max = max;
    }


    /**
     * Randomly samples this uniform distribution.
     *
     * @return A pseudorandomrandom value distributed according to a uniform distribution between {@link #min} (inclusive) and {@link #max}
     * (exclusive).
     */
    @Override
    public Double sample() {
        return rng.nextDouble(min, max);
    }
}
