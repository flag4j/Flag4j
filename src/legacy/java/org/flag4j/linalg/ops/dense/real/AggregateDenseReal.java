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

package org.flag4j.linalg.ops.dense.real;


/**
 * This class contains several low-level methods useful for computing aggregation ops on dense tensors.
 */
public final class AggregateDenseReal {

    private AggregateDenseReal() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the index of the minimum value in this tensor. If the minimum value occurs at more than one index,
     * the index of the first occurrence is taken.
     * @param entries Entries of the tensor.
     * @return The index of the minimum value in this tensor. If there are zero items in the array, -1 is returned.
     */
    public static int argmin(double[] entries) {
        double currMin = Double.MAX_VALUE;
        int currMinIndex = -1;

        for(int i=0; i<entries.length; i++) {
            if(entries[i] < currMin) {
                currMin = entries[i]; // Update current minimum.
                currMinIndex = i;
            }
        }

        return currMinIndex;
    }


    /**
     * Computes the index of the minimum absolute value in this tensor. If the minimum absolute value occurs at more than one index,
     * the index of the first occurrence is taken.
     * @param entries Entries of the tensor.
     * @return The index of the minimum absolute value in this tensor. If there are zero items in the array, -1 is returned.
     */
    public static int argminAbs(double[] entries) {
        double currMin = Double.MAX_VALUE;
        int currMinIndex = -1;

        for(int i=0; i<entries.length; i++) {
            double abs = Math.abs(entries[i]);

            if(abs < currMin) {
                currMin = abs; // Update current minimum.
                currMinIndex = i;
            }
        }

        return currMinIndex;
    }


    /**
     * Computes the index of the maximum value in this tensor. If the maximum value occurs at more than one index,
     * the index of the first occurrence is taken.
     * @param entries Entries of the tensor.
     * @return The index of the maximum value in this tensor. If there are zero items in the array, -1 is returned.
     */
    public static int argmax(double[] entries) {
        double currMax = Double.MIN_VALUE;
        int currMaxIndex = -1;

        for(int i=0, size=entries.length; i<size; i++) {
            if(entries[i] > currMax) {
                currMax = entries[i]; // Update current minimum.
                currMaxIndex = i;
            }
        }

        return currMaxIndex;
    }


    /**
     * Computes the index of the maximum absolute value in this tensor. If the maximum absolute value occurs at more than one index,
     * the index of the first occurrence is taken.
     * @param entries Entries of the tensor.
     * @return The index of the maximum absolute value in this tensor. If there are zero items in the array, -1 is returned.
     */
    public static int argmaxAbs(double[] entries) {
        double currMax = 0;
        int currMaxIndex = -1;

        for(int i=0, size=entries.length; i<size; i++) {
            double abs = Math.abs(entries[i]);
            if(abs > currMax) {
                currMax = abs; // Update current minimum.
                currMaxIndex = i;
            }
        }

        return currMaxIndex;
    }



    /**
     * Computes the index of the first entry in a tensor which is equal to the specified {@code value}.
     * @param entries Entries of this tensor.
     * @param value Value to find index of.
     * @return The index of the first entry in {@code items} which is equal to {@code value}. If no such value exists,
     * then -1 is returned.
     */
    public static int argEq(double[] entries, double value) {
        int currIndex = -1;

        for(int i=0; i<entries.length; i++) {
            if(entries[i]==value) {
                currIndex = i;
                break;
            }
        }

        return currIndex;
    }


    /**
     * Computes the index of the first entry in a tensor which is equal to the specified {@code value}.
     * @param entries Entries of this tensor.
     * @param value Value to find index of.
     * @return The index of the first entry in {@code items} which is equal to {@code value}. If no such value exists,
     * then -1 is returned.
     */
    public static int argEq(int[] entries, int value) {
        int currIndex = -1;

        for(int i=0; i<entries.length; i++) {
            if(entries[i]==value) {
                currIndex = i;
                break;
            }
        }

        return currIndex;
    }
}
