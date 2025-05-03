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

package org.flag4j.io;

import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * Printing and formating options for tensors, matrices, and vectors.
 */
public final class PrintOptions {

    /**
     * Default padding between elements when printing.
     */
    public static final int DEFAULT_PADDING = 2;
    /**
     * Default maximum number of rows to print.
     */
    public static final int DEFAULT_MAX_ROWS = 10;
    /**
     * Default maximum number of columns to print.
     */
    public static final int DEFAULT_MAX_COLS = 10;
    /**
     * Default precision (i.e., number of decimals) to use when printing.
     */
    public static final int DEFAULT_PRECISION = 8;
    /**
     * Default flag for centering elements when printing.
     */
    public static final boolean DEFAULT_CENTER = true;

    private PrintOptions() {
        // Hide default constructor for utility class.
    }


    /**
     * <p>
     * Padding to apply to each size of an element. Padding is divided up to either side of the element.
     * For instance, if {@code padding=4} then two spaces will be placed on either side of each element.
     * Further, since this is done per element, each element will have 4 spaces between them (in addition to any
     * additional padding from column alignment within a matrix).
     * 
     *
     * <p>
     * If negative, zero padding will be used.
     * 
     *
     * <p>
     * Default Value: {@link #DEFAULT_PADDING}
     * 
     */
    private static int padding = DEFAULT_PADDING;


    /**
     * <p>
     * Maximum number of rows to print from a matrix.
     * If a matrix has more rows than this value, rows with
     * indices larger than this value, except for the last row,
     * will not be printed. If zero or negative all rows will be printed.
     * The last row is always printed
     * 
     *
     * <p>
     * Default Value: {@link #DEFAULT_MAX_ROWS}
     * 
     */
    private static int maxRows = DEFAULT_MAX_ROWS;


    /**
     * Maximum number of columns to print from a matrix/vector.
     * If a matrix has more columns than this value, columns with
     * indices equal to or larger than this value, except for the last column,
     * will <b>not</b> be printed. If zero or negative all columns will be printed.
     * The last column is always printed.<br><br>
     *
     * Default Value: {@link #DEFAULT_MAX_COLS}
     */
    private static int maxColumns = DEFAULT_MAX_COLS;


    /**
     * Precision of the printed matrix values. i.e., the number of decimal places printed.
     * If negative, max precision is used.
     * <br><br>
     * Default Value: {@link #DEFAULT_PRECISION}.
     */
    private static int precision = DEFAULT_PRECISION;


    /**
     * A flag which indicates if each value should be centered within its column.
     * <br><br>
     * Default value: {@link #DEFAULT_PRECISION}.
     */
    private static boolean center = DEFAULT_CENTER;


    /**
     * Sets the centering flag.
     * @param center Flag for centering values within its column.
     */
    public static void setCentering(boolean center) {
        // TODO: Could take an enum and allow for left, right, and center justified.
        PrintOptions.center = center;
    }


    /**
     * Gets the centering flag.
     * @return The current value of the center flag.
     */
    public static boolean useCentering() {
        return center;
    }


    /**
     * Sets the printing precision for which values in a matrix/vector
     * @param precision The precision to use.
     */
    public static void setPrecision(int precision) {
        if(precision < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(precision, "precision"));

        PrintOptions.precision = precision;
    }


    /**
     * Gets the current printing precision.
     * @return The current printing precision.
     */
    public static int getPrecision() {
        return precision;
    }


    /**
     * Gets the current maximum number of columns to print.
     * @return The current maximum number of columns to print.
     */
    public static int getMaxColumns() {
        return maxColumns;
    }


    /**
     * Sets the maximum number of columns to print.
     * @param maxColumns Maximum number of columns to print.
     */
    public static void setMaxColumns(int maxColumns) {
        if(maxColumns < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(maxColumns, "maxColumns"));

        PrintOptions.maxColumns = maxColumns;
    }


    /**
     * Gets the maximum number of rows to print.
     * @return The maximum number of rows to print.
     */
    public static int getMaxRows() {
        return maxRows;
    }


    /**
     * Set the maximum number of rows to print.
     * @param maxRows The new maximum number of rows to print.
     */
    public static void setMaxRows(int maxRows) {
        if(maxRows < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(maxRows, "maxRows"));

        PrintOptions.maxRows = maxRows;
    }


    /**
     * Set the maximum number of rows and columns to print.
     * @param maxRows The new maximum number of rows to print.
     * @param maxCols The maximum number of columns to print.
     */
    public static void setMaxRowsCols(int maxRows, int maxCols) {
        ValidateParameters.ensureAllGreaterEq(1, maxRows, maxCols);

        PrintOptions.maxRows = maxRows;
        PrintOptions.maxColumns = maxCols;
    }


    /**
     * Set the maximum number of rows and columns to print.
     * @param maxRowCols The new maximum number of rows and columns to print.
     */
    public static void setMaxRowsCols(int maxRowCols) {
        ValidateParameters.ensureGreaterEq(1, maxRowCols);

        PrintOptions.maxRows = maxRowCols;
        PrintOptions.maxColumns = maxRowCols;
    }


    /**
     * Gets the current padding amount for columns.
     * @return The current padding amount for columns.
     */
    public static int getPadding() {
        return padding;
    }


    /**
     * Sets the minimum padding amount for the columns.
     * @param padding New padding amount for the columns.
     */
    public static void setPadding(int padding) {
        if(padding < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(padding, "padding"));

        PrintOptions.padding = padding;
    }


    /**
     * Resets all print options to their default values.
     */
    public static void resetAll() {
        padding = DEFAULT_PADDING;
        maxColumns = DEFAULT_MAX_COLS;
        maxRows = DEFAULT_MAX_ROWS;
        precision = DEFAULT_PRECISION;
        center = DEFAULT_CENTER;
    }
}
