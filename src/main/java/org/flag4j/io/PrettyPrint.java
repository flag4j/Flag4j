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

package org.flag4j.io;


import org.flag4j.arrays.Shape;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for formatting arrays as &#x2605;&#x2605;<em>pretty</em>&#x2605;&#x2605; human-readable strings.
 */
public final class PrettyPrint {


    private PrettyPrint() {
        // Hide default constructor in utility class.
    }


    /**
     * Formats array as an abbreviated array so that no more than {@code maxEntries} is actually printed.
     * @param arr Array to format.
     * @param maxEntries The maximum number of data to print.
     * @param padding The amount of padding to use between each entry.
     * @param precision The number of decimal places to print for each value.
     * @param centering Flag indicating if each value should be centered within the padding.
     * @return A string representing the abbreviated and formatted array.
     */
    public static String abbreviatedArray(double[] arr, int maxEntries, int padding, int precision, boolean centering) {
        ValidateParameters.ensureNonNegative(maxEntries, padding, precision);
        StringBuilder result = new StringBuilder("[");
        String value;
        int width;

        // Get data up until the stopping point.
        for(int i=0; i<maxEntries-1 && i<arr.length-1; i++) {
            value = StringUtils.ValueOfRound(arr[i], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Add last value
        if(maxEntries < arr.length) {
            width = padding + 3;
            value = "...";
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        value = StringUtils.ValueOfRound(arr[arr.length-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }


    /**
     * Formats array as an abbreviated array so that no more than {@code maxEntries} is actually printed.
     * @param arr Array to format.
     * @param maxEntries The maximum number of data to print.
     * @param padding The amount of padding to use between each entry.
     * @param precision The number of decimal places to print for each value.
     * @param centering Flag indicating if each value should be centered within the padding.
     * @return A string representing the abbreviated and formatted array.
     */
    public static <T> String abbreviatedArray(T[] arr, int maxEntries, int padding, int precision,
                                                               boolean centering) {
        ValidateParameters.ensureNonNegative(maxEntries, padding, precision);
        StringBuilder result = new StringBuilder("[");
        String value;
        int width;

        // Get data up until the stopping point.
        for(int i=0; i<maxEntries-1 && i<arr.length-1; i++) {
            value = StringUtils.ValueOfRound(arr[i], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Add last value
        if(maxEntries < arr.length) {
            width = padding + 3;
            value = "...";
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        value = StringUtils.ValueOfRound(arr[arr.length-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }


    /**
     * Formats array as an abbreviated array so that no more than {@code maxCols} is actually printed.
     * @param arr Array to format.
     * @param maxRows Maximum number of rows to print.
     * @param maxCols The maximum number of columns to print.
     * @param padding The amount of padding to use between each entry.
     * @param centering Flag indicating if each value should be centered within the padding.
     * @param offset Offset for array rows after the first.
     * @param centering Flag indicating if each value should be centered within the padding.
     * @return A string representing the abbreviated and formatted array.
     */
    public static String abbreviatedArray(int[][] arr, int maxRows, int maxCols, int padding, int offset, boolean centering) {
        ValidateParameters.ensureNonNegative(maxRows, maxCols, padding);
        StringBuilder result = new StringBuilder("[ ");
        String offsetPad = " ".repeat(offset);
        String value;
        int width;

        // Get data up until the stopping point.
        for(int i=0, stop=arr.length-1; i<maxRows-1 && i<stop; i++) {
            value = abbreviatedArray(arr[i], maxCols, padding, centering);
            width = padding + value.length();
            if(i > 0) result.append(offsetPad);
            result.append(String.format("%-" + width + "s\n", value));
        }

        // Add last value.
        if(maxRows < arr.length) {
            width = padding + 3;
            value = "...";
            result.append(offsetPad).append(" ").append(String.format("%-" + width + "s\n", value));
        }

        value = abbreviatedArray(arr[arr.length-1], maxCols, padding, centering);
        width = padding + value.length();
        result.append(offsetPad).append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }


    /**
     * Formats array as an abbreviated array so that no more than {@code maxEntries} is actually printed.
     * @param arr Array to format.
     * @param maxEntries The maximum number of data to print.
     * @param padding The amount of padding to use between each entry.
     * @param precision The number of decimal places to print for each value.
     * @param centering Flag indicating if each value should be centered within the padding.
     * @return A string representing the abbreviated and formatted array.
     */
    public static String abbreviatedArray(int[] arr, int maxEntries, int padding, boolean centering) {
        ValidateParameters.ensureNonNegative(maxEntries, padding);
        StringBuilder result = new StringBuilder("[");
        String value;
        int width;

        // Get data up until the stopping point.
        for(int i=0; i<maxEntries-1 && i<arr.length-1; i++) {
            value = String.valueOf(arr[i]);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Add last value
        if(maxEntries < arr.length) {
            width = padding + 3;
            value = "...";
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(arr.length > 0) {
            value = String.valueOf(arr[arr.length-1]);
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        return result.append("]").toString();
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles up until stopping index.
     * The length of the last element is always considered.
     * @param src Array for which to compute the max string representation length of a double in.
     * @param stopIndex Stopping index for finding max length.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLengthRounded(double[] src, int stopIndex) {
        return maxStringLengthRounded(src, 0, stopIndex, 1, src.length - 1);
    }


    /**
     * Computes the maximum length of the string representation of an element in an array up until some stopping index.
     * The length of the last element is always considered.
     * @param src Array for which to compute the max string representation length of a double in.
     * @param stopIndex Stopping index for finding max length.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static <T> int maxStringLengthRounded(T[] src, int stopIndex) {
        return maxStringLengthRounded(src, 0, stopIndex, 1, src.length - 1);
    }


    /**
     * Computes the maximum length of the string representation of an element in an array over a specified range with elements spaced
     * by some specified {@code stride}.
     * @param src Array to find maximum length string representation
     * @param startIdx Staring index to search for maximum length string (inclusive).
     * @param stopIdx Stopping index to search for maximum length string (exclusive).
     * @param stride The gap between consecutive elements within {@code src} to check.
     * @param finalIdx The final index to consider for the maximum length string. The length of the string representation of the
     * element at this index in {@code src} is <em>always</em> considered.
     * @return The maximum string representation length of elements within {@code src} between indices {@code startIdx} (inclusive)
     * and {@code stopIdx} spaced by {@code stride} and a final element at index {@code finalIdx}.
     * @param <T> Type of elements within the array.
     */
    public static <T> int maxStringLengthRounded(double[] src, int startIdx, int stopIdx, int stride, int finalIdx) {
        int maxLength = -1;
        int precision = PrintOptions.getPrecision();

        if(startIdx != stopIdx) {
            ValidateParameters.validateArrayIndices(src.length, startIdx, stopIdx-1);
            ValidateParameters.ensurePositive(stride);
            ValidateParameters.ensureGreaterEq(startIdx, stopIdx);

            for(int i=startIdx; i<stopIdx; i+=stride)
                maxLength = Math.max(maxLength, lengthRounded(src[i], precision));
        }

        // Always consider the element at finalIdx.
        maxLength = Math.max(maxLength, lengthRounded(src[finalIdx], precision));

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of an element in an array over a specified range with elements spaced
     * by some specified {@code stride}.
     * @param src Array to find maximum length string representation
     * @param startIdx Staring index to search for maximum length string (inclusive).
     * @param stopIdx Stopping index to search for maximum length string (exclusive).
     * @param stride The gap between consecutive elements within {@code src} to check.
     * @param finalIdx The final index to consider for the maximum length string. The length of the string representation of the
     * element at this index in {@code src} is <em>always</em> considered.
     * @return The maximum string representation length of elements within {@code src} between indices {@code startIdx} (inclusive)
     * and {@code stopIdx} spaced by {@code stride} and a final element at index {@code finalIdx}.
     * @param <T> Type of elements within the array.
     */
    public static <T> int maxStringLengthRounded(T[] src, int startIdx, int stopIdx, int stride, int finalIdx) {
        int maxLength = -1;
        int precision = PrintOptions.getPrecision();

        if(startIdx != stopIdx) {
            ValidateParameters.validateArrayIndices(src.length, startIdx, stopIdx-1);
            ValidateParameters.ensurePositive(stride);
            ValidateParameters.ensureGreaterEq(startIdx, stopIdx);

            for(int i=startIdx; i<stopIdx; i+=stride)
                maxLength = Math.max(maxLength, lengthRounded(src[i], precision));
        }

        // Always consider the element at finalIdx.
        maxLength = Math.max(maxLength, lengthRounded(src[finalIdx], precision));

        return maxLength;
    }


    /**
     * Converts a matrix into a "pretty" string using parameters set in the {@link PrintOptions} class.
     * @param shape Shape of the matrix. Must be rank 2.
     * @param data Entries of the matrix.
     * @return This matrix represented as a "pretty" string.
     * @param <T> Type of an individual entry of the matrix.
     */
    public static <T> String matrixToString(Shape shape, double[] data) {
        ValidateParameters.ensureRank(shape, 2);
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        if (data.length == 0) {
            result.append("[]"); // No data in this matrix.
        } else {
            int numRows = shape.getSize(0);
            int numCols = shape.getSize(1);

            int maxRows = PrintOptions.getMaxRows();
            int maxCols = PrintOptions.getMaxColumns();

            int rowStopIndex = Math.min(maxRows - 1, numRows - 1);
            int rowStopOffset = rowStopIndex*numCols;
            boolean truncatedRows = maxRows < numRows;

            int colStopIndex = Math.min(maxCols - 1, numCols - 1);
            int lastIdxOffset = (numRows-1)*numCols;
            boolean truncatedCols = maxCols < numCols;

            // Build list of column indices to print
            List<Integer> columnsToPrint = new ArrayList<>();
            for (int j = 0; j < colStopIndex; j++)
                columnsToPrint.add(j);

            if (truncatedCols) columnsToPrint.add(-1); // Use -1 to indicate '...'.
            columnsToPrint.add(numCols - 1); // Always include the last column.

            // Compute maximum widths for each column
            List<Integer> maxWidths = new ArrayList<>();
            for (Integer colIndex : columnsToPrint) {
                int maxWidth;
                if (colIndex == -1)
                    maxWidth = 3; // Width for '...'.
                else {
                    maxWidth = PrettyPrint.maxStringLengthRounded(
                            data, colIndex, rowStopOffset + colIndex, numCols,
                            lastIdxOffset + colIndex);
                }

                maxWidths.add(maxWidth);
            }

            // Build the rows up to the stopping index.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(shape, data, i, columnsToPrint, maxWidths));
                result.append("\n");
            }

            if (truncatedRows) {
                // Print a '...' row to indicate truncated rows.
                int totalWidth = maxWidths.stream().mapToInt(w -> w + PrintOptions.getPadding()).sum();
                String value = "...";

                if (PrintOptions.useCentering())
                    value = StringUtils.center(value, totalWidth);

                if (rowStopIndex != 0)
                    result.append(" ");

                result.append(String.format("[%-" + totalWidth + "s]\n", value));
            }

            // Append the last row.
            result.append(rowToString(shape, data, numRows - 1, columnsToPrint, maxWidths));
        }

        result.append("]");

        return result.toString();
    }


    /**
     * Converts a matrix into a "pretty" string using parameters set in the {@link PrintOptions} class.
     * @param shape Shape of the matrix. Must be rank 2.
     * @param data Entries of the matrix.
     * @return This matrix represented as a "pretty" string.
     * @param <T> Type of an individual entry of the matrix.
     */
    public static <T> String matrixToString(Shape shape, T[] data) {
        ValidateParameters.ensureRank(shape, 2);
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        if (data.length == 0) {
            result.append("[]"); // No data in this matrix.
        } else {
            int numRows = shape.getSize(0);
            int numCols = shape.getSize(1);

            int maxRows = PrintOptions.getMaxRows();
            int maxCols = PrintOptions.getMaxColumns();

            int rowStopIndex = Math.min(maxRows - 1, numRows - 1);
            int rowStopOffset = rowStopIndex*numCols;
            boolean truncatedRows = maxRows < numRows;

            int colStopIndex = Math.min(maxCols - 1, numCols - 1);
            int lastIdxOffset = (numRows-1)*numCols;
            boolean truncatedCols = maxCols < numCols;

            // Build list of column indices to print
            List<Integer> columnsToPrint = new ArrayList<>();
            for (int j = 0; j < colStopIndex; j++)
                columnsToPrint.add(j);

            if (truncatedCols) columnsToPrint.add(-1); // Use -1 to indicate '...'.
            columnsToPrint.add(numCols - 1); // Always include the last column.

            // Compute maximum widths for each column
            List<Integer> maxWidths = new ArrayList<>();
            for (Integer colIndex : columnsToPrint) {
                int maxWidth;
                if (colIndex == -1)
                    maxWidth = 3; // Width for '...'.
                else {
                    maxWidth = PrettyPrint.maxStringLengthRounded(
                            data, colIndex, rowStopOffset + colIndex, numCols,
                            lastIdxOffset + colIndex);
                }

                maxWidths.add(maxWidth);
            }

            // Build the rows up to the stopping index.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(shape, data, i, columnsToPrint, maxWidths));
                result.append("\n");
            }

            if (truncatedRows) {
                // Print a '...' row to indicate truncated rows.
                int totalWidth = maxWidths.stream().mapToInt(w -> w + PrintOptions.getPadding()).sum();
                String value = "...";

                if (PrintOptions.useCentering())
                    value = StringUtils.center(value, totalWidth);

                if (rowStopIndex != 0)
                    result.append(" ");

                result.append(String.format("[%-" + totalWidth + "s]\n", value));
            }

            // Append the last row.
            result.append(rowToString(shape, data, numRows - 1, columnsToPrint, maxWidths));
        }

        result.append("]");

        return result.toString();
    }


    /**
     * Converts a row of a matrix to a "pretty" string using the parameters set in the {@link PrettyPrint} class.
     * @param shape Shape of the matrix.
     * @param data Entries of the matrix.
     * @param rowIndex Index of the row to convert to a "pretty" string.
     * @param columnsToPrint A list of columns that should be pretend for this row.
     * @param maxWidths A list of the maximum width for each column to print.
     * @return The specified row of the matrix represented as a "pretty" string.
     * @param <T> Type of an individual entry of the matrix.
     */
    private static <T> String rowToString(Shape shape, double[] data, int rowIndex, List<Integer> columnsToPrint,
                                          List<Integer> maxWidths) {
        StringBuilder sb = new StringBuilder();

        // Start the row with appropriate bracket.
        sb.append(rowIndex > 0 ? " [" : "[");
        int rowOffset = rowIndex*shape.getSize(1);
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean useCentering = PrintOptions.useCentering();

        // Loop over the columns to print.
        for (int i = 0; i < columnsToPrint.size(); i++) {
            int colIndex = columnsToPrint.get(i);
            String value;
            int width = padding + maxWidths.get(i);

            if (colIndex == -1) // Placeholder for truncated columns.
                value = "...";
            else
                value = StringUtils.ValueOfRound(data[rowOffset + colIndex], precision);

            if (useCentering)
                value = StringUtils.center(value, width);

            sb.append(String.format("%-" + width + "s", value));
        }

        // Close the row.
        sb.append("]");

        return sb.toString();
    }


    /**
     * Converts a row of a matrix to a "pretty" string using the parameters set in the {@link PrettyPrint} class.
     * @param shape Shape of the matrix.
     * @param data Entries of the matrix.
     * @param rowIndex Index of the row to convert to a "pretty" string.
     * @param columnsToPrint A list of columns that should be pretend for this row.
     * @param maxWidths A list of the maximum width for each column to print.
     * @return The specified row of the matrix represented as a "pretty" string.
     * @param <T> Type of an individual entry of the matrix.
     */
    private static <T> String rowToString(Shape shape, T[] data, int rowIndex, List<Integer> columnsToPrint, List<Integer> maxWidths) {
        StringBuilder sb = new StringBuilder();

        // Start the row with appropriate bracket.
        sb.append(rowIndex > 0 ? " [" : "[");
        int rowOffset = rowIndex*shape.getSize(1);
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean useCentering = PrintOptions.useCentering();

        // Loop over the columns to print.
        for (int i = 0; i < columnsToPrint.size(); i++) {
            int colIndex = columnsToPrint.get(i);
            String value;
            int width = padding + maxWidths.get(i);

            if (colIndex == -1) // Placeholder for truncated columns.
                value = "...";
            else
                value = StringUtils.ValueOfRound(data[rowOffset + colIndex], precision);

            if (useCentering)
                value = StringUtils.center(value, width);

            sb.append(String.format("%-" + width + "s", value));
        }

        // Close the row.
        sb.append("]");

        return sb.toString();
    }


    /**
     * Compute the length of the string representation of the specified field value rounded to {@code precision} if possible.
     * @param value Field value to round. If the value cannot be rounded the length of the object unchanged will be returned.
     * @param precision The precision to round {@code value} to.
     * @return The length of the string representation of the specified field value rounded to {@code precision} if possible. If the
     * {@code value} cannot be rounded, then the length of the full string is returned.
     */
    private static <T> int lengthRounded(double value, int precision) {
        return StringUtils.ValueOfRound(value, precision).length();
    }


    /**
     * Compute the length of the string representation of the specified field value rounded to {@code precision} if possible.
     * @param value Field value to round. If the value cannot be rounded the length of the object unchanged will be returned.
     * @param precision The precision to round {@code value} to.
     * @return The length of the string representation of the specified field value rounded to {@code precision} if possible. If the
     * {@code value} cannot be rounded, then the length of the full string is returned.
     */
    private static <T> int lengthRounded(T value, int precision) {
        return StringUtils.ValueOfRound(value, precision).length();
    }
}
