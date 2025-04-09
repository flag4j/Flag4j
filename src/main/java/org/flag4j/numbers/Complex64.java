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

package org.flag4j.numbers;

import org.flag4j.io.parsing.ComplexNumberParser;
import org.flag4j.util.ErrorMessages;

import java.math.BigDecimal;
import java.math.RoundingMode;


/**
 * Represents an immutable complex number with single-precision floating point components.
 *
 * <p>This class models a complex number in rectangular (Cartesian) form, defined by its real and imaginary parts,
 * each stored as a 32-bit floating-point number ({@code float}). Instances of {@code Complex64} are immutable
 * and thread-safe.
 *
 * <p>The class provides various operations for complex arithmetic, including addition, subtraction,
 * multiplication, division, exponentiation, logarithms, trigonometric and hyperbolic functions, as well as
 * utilities for comparing and rounding complex numbers.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Complex64 a = new Complex64(2.5f, 3.2f);  // Creates a complex number 2.5 + 3.2i.
 * Complex64 b = new Complex64(1, -4);       // Creates a complex number 1 - 4i.
 * Complex64 sum = a.add(b);                 // Sum of a and b.
 * Complex64 product = a.mult(b);            // Product of a and b.
 * }</pre>
 *
 * <h2>Special Values:</h2>
 * <p>The class defines several constants for common complex numbers:
 * <ul>
 *   <li>{@link #ZERO} - Complex number representing zero (0 + 0i).</li>
 *   <li>{@link #ONE} - Complex number representing one (1 + 0i).</li>
 *   <li>{@link #TWO} - Complex number representing two (2 + 0i).</li>
 *   <li>{@link #THREE} - Complex number representing three (3 + 0i).</li>
 *   <li>{@link #I} - The imaginary unit (0 + 1i).</li>
 *   <li>{@link #NEGATIVE_I} - Negative imaginary unit (0 - 1i).</li>
 *   <li>{@link #NaN} - Complex number representing a Not-a-Number value.</li>
 *   <li>{@link #POSITIVE_INFINITY} - Complex number with infinite real part.</li>
 *   <li>{@link #NEGATIVE_INFINITY} - Complex number with negative infinite real part.</li>
 * </ul>
 *
 * <h2>Mathematical Functions:</h2>
 * <p>The class provides methods for several mathematical functions:
 * <ul>
 *   <li>{@link #sqrt()} - Computes the square root.</li>
 *   <li>{@link #ln(Complex64)} - Computes the natural logarithm.</li>
 *   <li>{@link #exp(Complex64)} - Computes the exponential function.</li>
 *   <li>{@link #pow(Complex64, Complex64)} - Computes the power function.</li>
 *   <li>{@link #sin(Complex64)} - Computes the sine function.</li>
 *   <li>{@link #cos(Complex64)} - Computes the cosine function.</li>
 *   <li>{@link #tan(Complex64)} - Computes the tangent function.</li>
 *   <li>{@link #sinh(Complex64)} - Computes the hyperbolic sine function.</li>
 *   <li>{@link #cosh(Complex64)} - Computes the hyperbolic cosine function.</li>
 *   <li>{@link #tanh(Complex64)} - Computes the hyperbolic tangent function.</li>
 * </ul>
 *
 * @see Complex128
 */
public class Complex64 implements Field<Complex64> {
    private static final long serialVersionUID = 1L;

    // Several constants are provided for convenience.
    /**
     * The complex number with zero imaginary and real parts.
     */
    public static final Complex64 ZERO = new Complex64(0);
    /**
     * The complex number with zero imaginary part and one real part.
     */
    public static final Complex64 ONE = new Complex64(1);
    /**
     * The complex number with zero imaginary part and two real part.
     */
    public static final Complex64 TWO = new Complex64(2);
    /**
     * The complex number with zero imaginary part and three real part.
     */
    public static final Complex64 THREE = new Complex64(3);
    /**
     * The float value closer than any other to the square root of 2
     */
    public static final Complex64 ROOT_TWO = new Complex64((float) Math.sqrt(2));
    /**
     * The float value closer than any other to the square root of 3
     */
    public static final Complex64 ROOT_THREE = new Complex64((float) Math.sqrt(3));
    /**
     * The imaginary unit i.
     */
    public static final Complex64 I = new Complex64(0, 1);
    /**
     * The additive inverse of the imaginary unit, -i.
     */
    public static final Complex64 NEGATIVE_I = new Complex64(0, -1);
    /**
     * Complex number with real part equal to {@link Float#POSITIVE_INFINITY}.
     */
    public static final Complex64 POSITIVE_INFINITY = new Complex64(Float.POSITIVE_INFINITY);
    /**
     * Complex number with real part equal to {@link Float#NEGATIVE_INFINITY}.
     */
    public static final Complex64 NEGATIVE_INFINITY = new Complex64(Float.NEGATIVE_INFINITY);
    /**
     * Complex number with real and imaginary parts equal to {@link Float#NaN}.
     */
    public static final Complex64 NaN = new Complex64(Float.NaN, Float.NaN);

    /**
     * Real component of the complex number.
     */
    public final float re;
    /**
     * Imaginary component of the complex number.
     */
    public final float im;
    /**
     * The magnitude of this complex number. Computed lazily.
     */
    private double mag = -1;
    /**
     * The conjugate of this complex number. Computed lazily.
     */
    private Complex64 conjugate = null;

    /**
     * Constructs a complex number with specified real component and zero imaginary component.
     * @param re Real component of complex number.
     */
    public Complex64(float re) {
        this.re = re;
        this.im = 0;
    }


    /**
     * Constructs a complex number with specified complex and real components.
     * @param re Real component of complex number.
     * @param im Imaginary component of complex number.
     */
    public Complex64(float re, float im) {
        this.re = re;
        this.im = im;
    }


    /**
     * Constructs a complex number from a string of the form {@code "a +/- bi"} where {@code a} and {b} are real values and either may be
     * omitted. i.e. {@code "a", "bi", "a +/- i"}, and {@code "i"} are all also valid.
     * @param num The string representation of a complex number.
     */
    public Complex64(String num) {
        Complex64 complexNum = ComplexNumberParser.parseNumberToComplex64(num);
        this.re = complexNum.re;
        this.im = complexNum.im;
    }


    /**
     * Checks if this complex has zero imaginary part and real part equal to a double.
     * @return True if {@code this.re == b && this.im == 0}. False otherwise.
     */
    public boolean equals(float b) {
        return this.re == b && this.im == 0;
    }


    /**
     * Checks if an object is equal to this Field element.
     * @param b Object to compare to this Field element.
     * @return True if the objects are the same or are both {@link Complex64}'s and have equal real and imaginary parts.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;
        Complex64 bCmp = (Complex64) b;

        return this.re == bCmp.re && this.im == bCmp.im;
    }


    /**
     * Generates the hashcode for this complex number.
     * @return An integer hash for this complex number.
     */
    @Override
    public int hashCode() {
        final int hashPrime1 = 17;
        final int hashPrime2 = 31;

        int hash = hashPrime2*hashPrime1 + Float.hashCode(re);
        hash = hashPrime2*hash + Float.hashCode(im);
        return hash;
    }


    /**
     * Checks if this complex number has zero imaginary part.
     * @return True if this complex number has zero imaginary part. False otherwise.
     * @see #isComplex()
     * @see #isImaginary()
     */
    public boolean isReal() {
        return this.im == 0;
    }


    /**
     * Checks if this complex number has zero real part.
     * @return True if this complex number has zero real part. False otherwise.
     * @see #isComplex()
     * @see #isReal()
     */
    public boolean isImaginary() {
        return this.re == 0;
    }


    /**
     * Checks if this complex number has non-zero imaginary part.
     * @return True if this complex number has non-zero imaginary part. False otherwise.
     * @see #isReal()
     * @see #isImaginary()
     */
    public boolean isComplex() {
        return this.im != 0;
    }


    /**
     * Sums two elements of this field (associative and commutative).
     * @param b Second field element in sum.
     * @return The sum of this element and {@code b}.
     */
    @Override
    public Complex64 add(Complex64 b) {
        return new Complex64(re + b.re, im + b.im);
    }


    /**
     * Sums an elements of this field with a real number.
     * @param b Second element in sum.
     * @return The sum of this element and {@code b}.
     */
    public Complex64 add(float b) {
        return new Complex64(re + b, im);
    }


    /**
     * Computes difference of two elements of this field.
     * @param b Second field element in difference.
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public Complex64 sub(Complex64 b) {
        return new Complex64(re - b.re, im - b.im);
    }


    /**
     * Computes difference of an element of this field and a real number.
     * @param b Second element in difference.
     * @return The difference of this field element and {@code b}.
     */
    public Complex64 sub(float b) {
        return new Complex64(re - b, im);
    }


    /**
     * Computes the sum of all data of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static Complex64 sum(Complex64... numbers) {
        float re = 0;
        float im = 0;

        for(Complex64 num : numbers) {
            re += num.re;
            im += num.im;
        }

        return new Complex64(re, im);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     * @param b Second field element in product.
     * @return The product of this field element and {@code b}.
     */
    @Override
    public Complex64 mult(Complex64 b) {
        return new Complex64(
                this.re*b.re - this.im*b.im,
                this.re*b.im + this.im*b.re
        );
    }


    /**
     * <p>Checks if this value is an additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return True if this value is an additive identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return True if this value is a multiplicative identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public Complex64 getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public Complex64 getOne() {
        return ONE;
    }


    /**
     * Multiplies an element of this field and a real number (associative and commutative).
     *
     * @param b Real number product.
     *
     * @return The product of this field element and {@code b}.
     */
    public Complex64 mult(float b) {
        return new Complex64(re*b, im*b);
    }


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public Complex64 div(Complex64 b) {
        Complex64 quotient;

        if (this.equals(ZERO) && !b.equals(ZERO)) {
            quotient = ZERO;

        } else if(b.isReal() && !b.equals(ZERO)) {
            quotient = new Complex64(re/b.re, im/b.re);

        } else {
            float divisor = b.re*b.re + b.im*b.im;

            quotient = new Complex64(
                    (re*b.re + im*b.im) / divisor,
                    (im*b.re - re*b.im) / divisor);
        }

        return quotient;
    }


    /**
     * Computes the quotient of an elements of this field and a real number.
     * @param b Second element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    public Complex64 div(float b) {
        if(this.equals(ZERO) && b != 0) return ZERO;
        else return new Complex64(re/b, im/b);
    }


    /**
     * Computes the magnitude of this field element.
     * @return The magnitude of this field element.
     */
    @Override
    public double mag() {
        if(mag < 0) mag = Math.hypot(re, im);
        return mag;
    }


    /**
     * Computes the square root of this field element.
     *
     * @implNote Despite {@link Complex64 Complex64's} storing real and imaginary components as 32-bit floats,
     * the computations in this method are performed with 64-bit precision with the result then being down-cast to 32-bits for the
     * final returned value.
     * @return The square root of this field element.
     */
    @Override
    public Complex64 sqrt() {
        if(im == 0) {
            if(re >= 0)
                return new Complex64((float) Math.sqrt(re));
            else
                return new Complex64(0, (float) Math.sqrt(-re));
        } else {
            double mag = mag();
            double sqrtRe = Math.sqrt((mag + re) / 2.0);
            double sqrtIm = Math.sqrt((mag - re) / 2.0);

            if (im > 0)
                return new Complex64((float) sqrtRe, (float) sqrtIm);
            else
                return new Complex64((float) sqrtRe, (float) -sqrtIm);
        }
    }


    /**
     * <p>Squares this magnitude of this complex number.
     *
     * @return The square of the magnitude of this complex number as a real value.
     */
    public float magSquared() {
        return re*re + im*im;
    }


    /**
     * <p>Computes the additive inverse for an element of this field.
     *
     * <p>An element -x is an additive inverse for a field element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this field element.
     */
    public Complex64 addInv() {
        return new Complex64(-re, -im);
    }


    /**
     * Computes the multiplicative inverse of this complex number.
     * @return The multiplicative inverse of this complex number.
     */
    public Complex64 multInv() {
        return ONE.div(this);
    }


    /**
     * Computes the conjugate of this field element.
     * @return The conjugate of this field element.
     */
    @Override
    public Complex64 conj() {
        if(conjugate == null) conjugate = new Complex64(re, -im);
        return conjugate;
    }


    /**
     * Compute a raised to the power of b.
     * and returns a {@link Complex64}.
     * @param a The base.
     * @param b The exponent.
     * @return a to the power of b.
     */
    public static Complex64 pow(float a, Complex64 b) {
        if(a < 0) {
            // Wrap base as complex number and compute using logarithms to avoid NaN in Math.pow method.
            return pow(new Complex64(a), b);
        } else if(b.im == 0) {
            // b is a real number.
            return new Complex64((float) Math.pow(a, b.re));
        } else {
            // Apply a change of base.
            double logA = Math.log(a);
            Complex64 power = new Complex64((float) (logA * b.re), (float) (logA * b.im));
            return exp(power);
        }
    }


    /**
     * Computes {@code a} raised to the power of {@code b}.
     * @param a The base.
     * @param b The exponent.
     * @return {@code a} raised to the power of {@code b}.
     */
    public static Complex64 pow(Complex64 a, Complex64 b) {
        if(a.im == 0 && a.re >= 0) {
            return (b.im == 0)
                    ? new Complex64((float) Math.pow(a.re, b.re))
                    : Complex64.pow(a.re, b);
        } else {
            return exp(b.mult(ln(a)));
        }
    }


    /**
     * Computes {@code a} raised to the power of {@code b}.
     * @param a The base.
     * @param b The exponent.
     * @return {@code a} raised to the power of {@code b}.
     */
    public static Complex64 pow(Complex64 a, float b) {
        return (a.im == 0 && a.re >= 0)
                ? new Complex64((float) Math.pow(a.re, b))
                : exp(ln(a).mult(b));
    }


    /**
     * Computes the exponential function with the given input.
     * @param exponent Input to the exponential function.
     * @return The output of the exponential function.
     */
    public static Complex64 exp(Complex64 exponent) {
        float expRe = (float) Math.exp(exponent.re);
        float re =  expRe*((float) Math.cos(exponent.im));
        float im = expRe*((float) Math.sin(exponent.im));

        return new Complex64(re, im);
    }


    /**
     * Computes the natural logarithm of a float. For non-negative values this function is equivalent to {@link Math#log}.
     * If the number is negative, then it is passed on as a complex value to {@link Complex64#ln(Complex64)}.
     * @param num Input to the complex natural logarithm function.
     * @return The principle value of the complex natural logarithm for the given input.
     */
    public static Complex64 ln(float num) {
        return (num<0)
                ? Complex64.ln(new Complex64(num))
                : new Complex64((float) Math.log(num));
    }


    /**
     * Computes the complex natural logarithm of a complex number. This function is the analytic continuation of
     * the natural logarithm, that is the log base {@link Math#E e}.
     * @param num Input to the complex natural logarithm function.
     * @return The principle value of the complex natural logarithm for the given input.
     */
    public static Complex64 ln(Complex64 num) {
        if(num.isReal() && num.re >=0) {
            return new Complex64((float) Math.log(num.re));
        } else {
            float re = (float) Math.log(Math.sqrt(num.re*num.re + num.im*num.im));
            float im = (float) Math.atan2(num.im, num.re);

            return new Complex64(re, im);
        }
    }


    /**
     * Computes the complex logarithm base 10 of a complex number. Please note, this is <b>NOT</b> the natural logarithm.
     * If the complex natural logarithm is desired see {@link #ln(float)}. To specify a base, see {@link #log(float, Complex64)}
     * or {@link #log(Complex64, Complex64)}. If the argument is non-negative, then this function is equivalent to
     * {@link Math#log(double)}.
     * @param num Input to the complex logarithm base 10 function.
     * @return The principle value of the complex logarithm base 10 for the given input.
     */
    public static Complex64 log(float num) {
        return (num <= 0)
                ? Complex64.log(new Complex64(num))
                : new Complex64((float) Math.log10(num));
    }


    /**
     * Computes the complex logarithm base 10 of a complex number. Please note, this is <b>NOT</b> the complex natural logarithm.
     * If the complex natural logarithm is desired see {@link #ln(Complex64)}. To specify a base, see {@link #log(float, Complex64)}
     * or {@link #log(Complex64, Complex64)}.
     * @param num Input to the complex logarithm base 10 function.
     * @return The principle value of the complex logarithm base 10 for the given input.
     */
    public static Complex64 log(Complex64 num) {
        if(num.isReal() && num.re >=0) {
            return new Complex64((float) Math.log10(num.re));
        } else {
            // Using the change of base formula
            Complex64 numerator = ln(num);
            Complex64 denominator = new Complex64((float) Math.log(10));

            return numerator.div(denominator);
        }
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static Complex64 log(float base, float num) {
        return Complex64.log(new Complex64(base), new Complex64(num));
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static Complex64 log(float base, Complex64 num) {
        return Complex64.log(new Complex64(base), num);
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static Complex64 log(Complex64 base, Complex64 num) {
        // Using the change of base formula
        Complex64 numerator = ln(num);
        Complex64 denominator = ln(base);

        return numerator.div(denominator);
    }


    /**
     * Computes the principle square root of a number. This method wraps {@link Math#sqrt(double)} and returns a {@link Complex64}.
     * @param num Input to square root.
     * @return The principle square root of {@code num}.
     */
    public static Complex64 sqrt(float num) {
        return (num >=0)
                ? new Complex64((float) Math.sqrt(num))
                : new Complex64(0f, (float) Math.sqrt(-num));
    }


    /**
     * Converts a complex number to an equivalent polar from.
     * @return An array of length 2 containing in order, the radius and angle (in radians) if the complex number.
     */
    public float[] toPolar() {
        float[] polar = new float[2];

        // Compute the magnitude and angle of the complex number.
        polar[0] = (float) this.mag();
        polar[1] = (float) Math.atan2(this.im, this.re);

        return polar;
    }


    /**
     * Converts a complex number expressed in polar from to the rectangular form.
     * @param r Radius of complex number.
     * @param theta Angle of the complex number in radians.
     * @return An equivalent complex number in rectangular form.
     */
    public static Complex64 fromPolar(float r, float theta) {
        float re = r*((float) Math.cos(theta));
        float im = r*((float) Math.sin(theta));

        return new Complex64(re, im);
    }


    /**
     * Computes the 2 argument arc-tangent function for a complex number. That is, for a complex number a+bi, atan2(b, a)
     * is computed. This method wraps {@link Math#atan2(double, double)}. <br>
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Float#NaN}
     * is returned.
     */
    public static float atan2(Complex64 num) {
        return (float) Math.atan2(num.im, num.re);
    }


    /**
     * Computes the complex argument function for a complex number.
     * is computed. This method is equivalent to {@link Complex64#atan2(Complex64)}. <br>
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Float#NaN}
     * is returned.
     */
    public static float arg(Complex64 num) {
        return atan2(num);
    }


    /**
     * Computes the trigonometric sine of a complex value.
     * @param num Complex valued input to the sine function. If num is real, then this is an angle in radians.
     * @return The trigonometric sine function evaluated at the specified value.
     */
    public static Complex64 sin(Complex64 num) {
        float re = (float) (Math.sin(num.re)*Math.cosh(num.im));
        float im = (float) (Math.cos(num.re)*Math.sinh(num.im));
        return new Complex64(re, im);
    }


    /**
     * Computes the inverse sine of a complex value. <br>
     * @param num Input to the inverse sine function.
     * @return The inverse sine of the input value.
     */
    public static Complex64 asin(Complex64 num) {
        Complex64 diff = ONE.sub(num.mult(num));
        Complex64 denominator = diff.sqrt();
        return atan(num.div(denominator));
    }


    /**
     * Computes the hyperbolic sine of a complex value.
     * @param num Input to the hyperbolic sine function.
     * @return The hyperbolic sine of the input value.
     */
    public static Complex64 sinh(Complex64 num) {
        Complex64 numerator = exp(num).add(exp(num.addInv()));
        Complex64 denominator = new Complex64(2);
        return numerator.div(denominator);
    }


    /**
     * Computes the trigonometric cosine value of a complex value.
     * @param num Complex valued input to the cosine function. If num is real, then this is an angle in radians.
     * @return The trigonometric cosine function evaluated at the specified value.
     */
    public static Complex64 cos(Complex64 num) {
        float re = (float) (Math.cos(num.re)*Math.cosh(num.im));
        float im = (float) (Math.sin(num.re)*Math.sinh(num.im));
        return new Complex64(re, im);
    }


    /**
     * Computes the inverse cosine of a complex value. <br>
     * @param num Input to the inverse cosine function.
     * @return The inverse cosine of the input value.
     */
    public static Complex64 acos(Complex64 num) {
        Complex64 term = new Complex64((float) (Math.PI/2d));
        return term.sub(asin(num));
    }


    /**
     * Computes the hyperbolic cosine of a complex value.
     * @param num Input to the hyperbolic cosine function.
     * @return The hyperbolic cosine of the input value.
     */
    public static Complex64 cosh(Complex64 num) {
        Complex64 numerator = exp(num).sub(exp(num.addInv()));
        Complex64 denominator = new Complex64(2);
        return numerator.div(denominator);
    }


    /**
     * Computes the trigonometric tangent value of a complex value.
     * @param num Complex valued input to the tangent function. If num is real, then this is an angle in radians.
     * @return The trigonometric tangent function evaluated at the specified value.
     */
    public static Complex64 tan(Complex64 num) {
        Complex64 numerator = new Complex64((float) Math.tan(num.re),
                (float) Math.tanh(num.im));
        Complex64 denominator = new Complex64(1f,
                (float) (-Math.tan(num.re)*Math.tanh(num.im)));
        return numerator.div(denominator);
    }


    /**
     * Computes the inverse tangent of a complex value. <br>
     * @param num Input to the inverse tangent function.
     * @return The inverse tangent of the input value.
     */
    public static Complex64 atan(Complex64 num) {
        Complex64 factor = TWO.multInv();
        Complex64 numerator = I.sub(num);
        Complex64 denominator = I.add(num);
        Complex64 log = Complex64.ln(numerator.div(denominator));
        return factor.mult(log);
    }


    /**
     * Computes the hyperbolic tangent of a complex value.
     * @param num Input to the hyperbolic tangent function.
     * @return The hyperbolic tangent of the input value.
     */
    public static Complex64 tanh(Complex64 num) {
        Complex64 exp = exp(TWO.mult(num));
        Complex64 numerator = exp.sub(ONE);
        Complex64 denominator = exp.add(ONE);
        return numerator.div(denominator);
    }


    /**
     * The complex signum function. Please note, if the value passed to this method is zero, the function will
     * return zero.
     *
     * @param value Value to evaluate the signum function at.
     * @return If the number is zero then this function returns zero. Otherwise, returns the number divided by its magnitude.
     */
    public static Complex64 sgn(Complex64 value) {
        if(value.equals(ZERO)) {
            return value;
        } else {
            float magnitude = (float) value.mag();
            return new Complex64(value.re / magnitude, value.im / magnitude);
        }
    }


    /**
     * Rounds both components of a complex number to the nearest respective integer.
     * @param n The complex number to round.
     * @return A complex number with integer real and imaginary components closest to the real and imaginary
     * components of the parameter n
     * @throws NumberFormatException If n is {@link Float#NaN}, {@link Float#POSITIVE_INFINITY} or
     * {@link Float#NEGATIVE_INFINITY}
     * @see #round(Complex64, int)
     */
    public static Complex64 round(Complex64 n) {
        return round(n, 0);
    }


    /**
     * Rounds number to specified number of decimal places. If the number is complex,
     * both the real and imaginary parts will be rounded.
     *
     * @param n Number to round.
     * @param decimals Number of decimals to round to.
     * @return The number {@code } rounded to the specified
     * 		number of decimals.
     * @throws IllegalArgumentException If decimals is less than zero.
     * @see #round(Complex64)
     */
    public static Complex64 round(Complex64 n, int decimals) {
        if (decimals < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(decimals, "decimals"));

        float real;
        float imaginary;

        if(Double.isFinite(n.re))
            real = BigDecimal.valueOf(n.re).setScale(decimals, RoundingMode.HALF_UP).floatValue();
        else
            real = n.re;

        if(Double.isFinite(n.im))
            imaginary = BigDecimal.valueOf(n.im).setScale(decimals, RoundingMode.HALF_UP).floatValue();
        else
            imaginary = n.im;

        return new Complex64(real, imaginary);
    }


    /**
     * Rounds a complex numbers to zero if its magnitude within the specified tolerance from zero.
     * @param n Number to round.
     * @param tol Max distances in complex plane for which number should be rounded to zero.
     * @return The
     */
    public static Complex64 roundToZero(Complex64 n, float tol) {
        return nearZero(n, tol) ? ZERO : n;
    }


    /**
     * Checks if a number is near zero in magnitude.
     * @param n Number to round.
     * @param tol Tolerance of how close to zero is
     * 		considered "near".
     * @return Returns {@code true} if magnitude of number is less than or equal to
     * 		{@code tol}; {@code false} otherwise.
     * @throws IllegalArgumentException If tol is less than 0.
     */
    public static boolean nearZero(Complex64 n, float tol) {
        if (tol < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(tol, "tol"));

        return n.mag() <= tol;
    }


    /**
     * Compares this element of the field with {@code b}.
     * @param b Second element of the field.
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b} in magnitude.</li>
     *     <li>< 0 if this field element is less than {@code b} in magnitude.</li>
     *     <li>> 0 if this field element is greater than {@code b} in magnitude.</li>
     *     Hence, this method returns zero if and only if the two field elements are equal in magnitude, a negative value if and
     *     only the field element it was called on is less than {@code b} in magnitude and positive if and only if the field
     *     element it was called on is greater than {@code b} in magnitude.
     * </ul>
     */
    public int compareTo(Complex64 b) {
        return Double.compare(this.mag(), b.mag());
    }


    /**
     * Converts this complex number to an equivalent double. This will be the magnitude of the complex number.
     *
     * @return A double value equivalent to this complex number. This will be the magnitude of the complex number.
     */
    @Override
    public double doubleValue() {
        return mag();
    }


    /**
     * Computes the minimum magnitude from an array of complex numbers.
     * @param values Array of values to compute the minimum magnitude from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, then {@code null} is returned.
     */
    public static Complex64 min(Complex64... values) {
        double minAbs = -1;
        double currMag;
        Complex64 min = null;
        if(values.length > 0) minAbs = values[0].mag();

        for(Complex64 value : values) {
            currMag = value.mag();

            if(currMag < minAbs) {
                minAbs = currMag;
                min = value;
            }
        }

        return min;
    }


    /**
     * Computes the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the minimum real component from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, {@link Float#NaN} is
     * returned.
     */
    public static Complex64 minRe(Complex64... values) {
        float min = Float.MAX_VALUE;
        float currMin;

        if(values.length == 0) min = Float.NaN;

        for(Complex64 value : values) {
            currMin = value.re;
            if(currMin < min) min = currMin;
        }

        return new Complex64(min);
    }


    /**
     * Computes the maximum magnitude from an array of complex numbers.
     * @param values Array of values to compute the maximum magnitude from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, then {@code null} is returned.
     */
    public static Complex64 max(Complex64... values) {
        double maxAbs = -1;
        double currMax;
        Complex64 max = null;

        if(values.length > 0) maxAbs = values[0].mag();

        for(Complex64 value : values) {
            currMax = value.mag();

            if(currMax > maxAbs) {
                maxAbs = currMax;
                max = value;
            }
        }

        return max;
    }


    /**
     * Computes the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the minimum real component from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, {@link Float#NaN} is
     * returned.
     */
    public static Complex64 maxRe(Complex64... values) {
        float max = Float.MIN_NORMAL;
        float currMax;

        if(values.length == 0) max = Float.NaN;

        for(Complex64 value : values) {
            currMax = value.re;
            if(currMax > max) max = currMax;
        }

        return new Complex64(max);
    }


    /**
     * Computes the index of the minimum magnitude from an array of complex numbers.
     * @param values Array of values to compute the index of the minimum magnitude from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static int argmin(Complex64... values) {
        double min = -1;
        double currMag;
        int arg = -1;

        if(values.length > 0) min = values[0].mag();

        for(int i=0; i<values.length; i++) {
            currMag = values[i].mag();

            if(currMag < min) {
                min = currMag;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Computes the index of the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the index of the minimum real component from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, -1 is returned.
     */
    public static int argminReal(Complex64... values) {
        float min = Float.MAX_VALUE;
        float currMin;
        int arg = -1;

        if(values.length == 0) min = Float.NaN;

        for(int i=0; i<values.length; i++) {
            currMin = values[i].re;

            if(currMin < min) {
                min = currMin;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Computes the index of the maximum magnitude from an array of complex numbers.
     * @param values Array of values to compute the index of the maximum magnitude from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static int argmax(Complex64... values) {
        double max = -1;
        double currMag;
        int arg = -1;

        if(values.length > 0) max = values[0].mag();

        for(int i=0; i<values.length; i++) {
            currMag = values[i].mag();

            if(currMag > max) {
                max = currMag;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Computes the index of the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the index of the minimum real component from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, -1 is
     * returned.
     */
    public static int argmaxReal(Complex64... values) {
        float max = Float.MIN_NORMAL;
        float currMax;
        int arg = -1;

        if(values.length == 0) max = Float.NaN;

        for (int i=0, size=values.length; i<size; i++) {
            currMax = values[i].re;

            if (currMax > max) {
                max = currMax;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Checks if this complex number is a real valued integer.
     * @return {@code true} if the real component of this number is an integer and the complex component is zero; {@code false} otherwise.
     */
    public boolean isInt() {
        boolean result = !(isInfinite() || isNaN());
        return Math.rint(re)==re && im==0 && result;
    }


    /**
     * Checks if this complex number is a real valued float.
     * @return {@code true} if the complex component is zero; {@code false} otherwise.
     */
    public boolean isFloat() {
        return im==0;
    }


    /**
     * <p>Checks if this field element is finite in magnitude.
     * <p>NOTE: It may be possible for this method to return true and {@link #mag()} to return {@link Float#POSITIVE_INFINITY},
     * {@link Float#NEGATIVE_INFINITY}, or {@link Float#NaN} if both components are finite but computing the magnitude explicitly
     * results in an overflow. 
     * @return True if this field element is finite in magnitude. False otherwise (i.e. infinite, NaN etc.).
     */
    public boolean isFinite() {
        // If both components are finite then the complex number will have finite magnitude.
        return Float.isFinite(re) && Float.isFinite(im);
    }


    /**
     * Checks if this field element is infinite in magnitude.
     * @return True if this field element is infinite in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    public boolean isInfinite() {
        // If either components is infinite then the complex number will have infinite magnitude.
        return Float.isInfinite(re) || Float.isInfinite(im);
    }


    /**
     * Checks if this field element is NaN in magnitude.
     * @return True if this field element is NaN in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    @Override
    public boolean isNaN() {
        return Float.isNaN(re) || Float.isNaN(im);
    }


    /**
     * Gets the real component of this complex number.
     * @return The real component of this complex number.
     */
    public float re() {
        return re;
    }


    /**
     * Gets the imaginary component of this complex number.
     * @return The imaginary component of this complex number.
     */
    public float im() {
        return im;
    }


    /**
     * Gets the length of the string representation of this complex number.
     * @param a Complex number.
     * @return The length of the string representation of the number.
     */
    public static int length(Complex64 a) {
        return a.toString().length();
    }


    /**
     * Converts the complex number to a string representation.
     * @return A string representation of the complex number.
     */
    @Override
    public String toString() {
        if (isNaN()) return "NaN";
        if (isInfinite()) return "Infinity";
        if (isZero()) return "0";

        String realPart = "";
        String imagPart = "";
        String sign;
        double imAbs = Math.abs(im);

        if(re != 0.0) {
            realPart = re % 1 == 0 ? String.valueOf((int) re) : String.valueOf(re);
            sign = im > 0.0 ? "+" : "-";
        } else {
            sign = im > 0.0 ? "" : "-";
        }

        if(imAbs == 1.0)
            imagPart = sign + (im < 0.0 ? "i" : "i");
        else if(im != 0.0)
            imagPart = sign + (im % 1 == 0 ? String.valueOf((int) imAbs) : String.valueOf(imAbs)) + "i";

        return realPart + imagPart;
    }
}
