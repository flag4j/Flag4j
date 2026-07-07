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

import org.flag4j.rng.RandomComplex;

import java.util.Random;


/**
 * <p>A 1D real Gaussian (normal) distribution <span class="latex-replace">N(&mu;, &sigma;<sup>2</sup>)</span>
 * <!-- LATEX:\(\mathcal{N}\left(\mu, \sigma^2\right)\) -->.
 *
 * <p>The PDF of the 1D real Gaussian distribution <span class="latex-replace">N(&mu;, &sigma;<sup>2</sup>)</span>
 * <!-- LATEX:\(\mathcal{N}\left(\mu, \sigma^2\right)\) --> with mean <span class="latex-inline">&mu;</span> and
 * standard deviation <span class="latex-inline">&sigma;</span> is given by:
 * <span class="latex-replace"><pre>
 *     f(x) = 1 / (2&pi;&sigma;<sup>2</sup>)<sup>1/2</sup> exp[ - (x - &mu;)<sup>2</sup> / (2&sigma;<sup>2</sup>) ]</pre></span>
 *
 * <!-- LATEX: \[
 * f(x) = \frac{1}{2\pi\sigma^2} \exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]
 * \] -->
 *
 * @see RealUniform
 * @see Complex128BiGaussian
 */
public class RealGaussian extends Distribution<Double, Random> {


    /**
     * Mean of the Gaussian distribution.
     */
    public final double mean;
    /**
     * Standard deviation of the Gaussian distribution.
     */
    public final double std;


    /**
     * Constructs a real Gaussian distribution.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param mean Mean of the Gaussian distribution.
     * @param std Standard deviation of the Gaussian distribution.
     */
    public RealGaussian(RandomComplex rng, double mean, double std) {
        super(rng);

        this.mean = mean;
        this.std = std;
    }


    /**
     * Randomly samples this Gaussian distribution.
     *
     * @return A pseudorandom value distributed according to a Gaussian distribution with specified {@link #mean} and {@link #std
     * standard deviation}.
     */
    @Override
    public Double sample() {
        return rng.nextGaussian();
    }
}
