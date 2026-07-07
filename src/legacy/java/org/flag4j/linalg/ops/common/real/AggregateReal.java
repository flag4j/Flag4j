/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.linalg.ops.common.real;


/**
 * This utility class contains several low-level methods useful for computing aggregation ops on dense/sparse tensors.
 */
public final class AggregateReal {

    private AggregateReal() {
        // Hide default constructor.
        
    }


    /**
     * Computes the sum of all items in this tensor. This can be applied to either real dense or spase tensors.
     * @param entries Entries of the tensor.
     * @return The sum of all items in this tensor.
     */
    public static double sum(double... entries) {
        double sum = 0;
        for(double value : entries)
            sum += value;

        return sum;
    }


    /**
     * Computes the sum of all items in this tensor. This can be applied to either real dense or spase tensors.
     * @param entries Entries of the tensor.
     * @return The sum of all items in this tensor.
     */
    public static double prod(double... entries) {
        double sum = 0;
        for(double value : entries)
            sum *= value;

        return sum;
    }
}
