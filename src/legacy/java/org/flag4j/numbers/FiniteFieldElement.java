/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
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

package org.flag4j.numbers;


public abstract class FiniteFieldElement<T extends FieldElement<T>> implements FieldElement<T> {
    // TODO: Add an `ExtensionFieldElement` object to represent the rest of the Galois Fields:
    //      We have GF(p) (`PrimeFieldElement`) but we `ExtensionFieldElement` would cover the GF(p^m) with m > 1 case.
    //      This willm require finding irreducible monic polynomials.

    /// The characteristic of the finite field this element belongs to.
    ///
    /// The characteristic of a ring is the smallest number of times the
    /// multiplicative identity must be added to itself to reach the additive identity.
    /// For a finite field <span>GF(p<sup>m</sup>)</span> where p is prime, the characteristic is p.
    /// @return The characteristic of the finite field this element belongs to.
    public abstract int characteristic();

    /// The degree of the finite field this element belongs to.
    ///
    /// For a finite field <span>GF(p<sup>m</sup>)</span> where p is prime and m &ge; 1, the degree is m.
    /// @return The degree of the finite filed this element belongs to.
    public abstract int degree();


    /**
     * <p>Checks if this field element is finite in magnitude.
     * <p>Note: This is <em>always</em> {@code true} for finite fields.
     *
     * @return {@code true}.
     */
    @Override
    public boolean isFinite() {
        return true;
    }


    /**
     * Checks if this field element is infinite in magnitude.
     *
     * <p>Note: This is <em>always</em> {@code false} for finite fields.
     *
     * @return {@code false}.
     */
    @Override
    public boolean isInfinite() {
        return false;
    }


    /**
     * Checks if this field element is NaN in magnitude.
     * <p>Note: This is <em>always</em> {@code false} for finite fields.
     *
     * @return {@code false}.
     */
    @Override
    public boolean isNaN() {
        return false;
    }


    /**
     * Checks if an integer is a prime number. This uses the 6k +/- 1 trial division test.
     * @param n The number to check the primality of.
     * @return {@code true} if {@code n} is prime; {@code false} otherwise.
     */
    protected static boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if ((n & 1) == 0 || n % 3 == 0) return false;

        for (int d = 5; d <= n / d; d += 6) {
            if (n % d == 0 || n % (d + 2) == 0) {
                return false;
            }
        }

        return true;
    }


    /**
     * <p>Computes the multiplicative inverse of an integer modulo a prime number.
     * <p>This method uses the extended Euclidean algorithm to compute the multiplicative inverse:
     *
     * @param a The integer to compute the multiplicative inverse of.
     * @param p The prime number to compute the multiplicative inverse modulo.
     * @return The multiplicative inverse of {@code a} modulo {@code p}.
     */
    protected static int modInverse(int a, int p) {
        a = Math.floorMod(a, p);

        if (a == 0) {
            throw new ArithmeticException("Zero has no multiplicative inverse");
        }

        long oldR = a;
        long r = p;
        long oldS = 1;
        long s = 0;

        while (r != 0) {
            long q = oldR / r;

            long nextR = oldR - q * r;
            oldR = r;
            r = nextR;

            long nextS = oldS - q * s;
            oldS = s;
            s = nextS;
        }

        // oldR is gcd(a, p), which should be 1 for nonzero a in GF(p)...
        if (oldR != 1) {
            throw new ArithmeticException(
                    a + " has no multiplicative inverse modulo " + p
            );
        }

        return (int) Math.floorMod(oldS, p);
    }


    /**
     * Validates that two finite fields are in the same finite field.
     * @param a The first finite field value.
     * @param b The second finite field value.
     */
    protected static <T extends FiniteFieldElement<T>> void ensureSameField(FiniteFieldElement<T> a, FiniteFieldElement<T> b) {
        if (a.characteristic() != b.characteristic() || a.degree() != b.degree()) {
                throw new IllegalArgumentException("Values must be in the same finite field but got fields with characteristics "
                        + a.characteristic() + " and " + b.characteristic() + ".");
        }
    }
}
