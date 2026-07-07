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

import org.flag4j.util.exceptions.Flag4jParsingException;

import java.util.StringJoiner;
import java.util.StringTokenizer;

/**
 * <p>A items record for storing the tokens of a
 * <a href="https://math.nist.gov/MatrixMarket/formats.html">Matrix Market Exchange Format</a> file header.
 * Valid tokens for each modifier are defined by the following enums:
 * <ul>
 *     <li>{@link MMObject} - Valid object types in the Matrix Market file.</li>
 *     <li>{@link MMFormat} - Valid formats in the Matrix Market file.</li>
 *     <li>{@link MMField} - Valid field types in the Matrix Market file.</li>
 *     <li>{@link MMSymmetry} - Valid symmetries in the Matrix Market file.</li>
 * </ul>
 *
 * @param object The type of object the Matrix Market file contains (e.g. matrix or vector).
 * @param format The format of the items the Matrix Market file contains (e.g. array (dense) or coordinate (sparse COO)).
 * @param field The field the elements belong to in the Matrix Market file (e.g. real, integer, or complex).
 * @param symmetry The symmetry of the items in the Matrix Market file (e.g. general, symmetric, skew-symmetric, Hermitian).
 * @param comments Comments to prepend to file after the main header.
 * @see MatrixMarketReader
 * @see org.flag4j.io.MatrixMarketWriter
 */
public record MatrixMarketHeader(
        MMObject object,
        MMFormat format,
        MMField field,
        MMSymmetry symmetry,
        String... comments) {

    /**
     * Header prefix for Matrix Market Exchange format.
     */
    private static final String MM_HEADER_PREFIX = "%%MatrixMarket";
    /**
     * Comment prefix for Matrix Market Exchange format.
     */
    private static final String MM_COMMENT_PREFIX = "\n% ";

    /**
     * Parses a string representing the header of a Matrix Market file and stores in a {@link MatrixMarketHeader} object.
     * @param header String containing the header, and only the header, of the Matrix Market file.
     * @return A {@link MatrixMarketHeader} object containing information about the Matrix Market object type, format, field type, and
     * symmetry.
     */
    public static MatrixMarketHeader parseHeader(String header) {
        StringTokenizer headerTokens = new StringTokenizer(header);

        if (!headerTokens.nextToken().equalsIgnoreCase("%%matrixmarket")) {
            throw new Flag4jParsingException("Invalid Matrix Market MMFormat file header. Did not find %%MatrixMarket.");
        }

        if(headerTokens.countTokens() < 3) {
            throw new Flag4jParsingException("Invalid Matrix Market file header. " +
                    "Did not specify both <object> and <format>.");
        }

        MMObject mmObject = MMObject.valueOf(headerTokens.nextToken().toUpperCase());
        MMFormat mmFormat = null;
        MMField mmField = null;
        MMSymmetry mmSymmetry = null;

        // If new token types are added, this will need to be updated.
        if(mmObject == MMObject.MATRIX) {
            if(headerTokens.countTokens() != 3) {
                throw new Flag4jParsingException("Invalid Matrix Market file header. " +
                        "Must specify exactly three qualifiers for a matrix: " +
                        "<format>, <field>, and <symmetry>.");
            }

            mmFormat = MMFormat.valueOf(headerTokens.nextToken().toUpperCase());
            mmField = MMField.valueOf(headerTokens.nextToken().toUpperCase());
            mmSymmetry = MMSymmetry.valueOf(headerTokens.nextToken().toUpperCase());
        }

        return new MatrixMarketHeader(mmObject, mmFormat, mmField, mmSymmetry);
    }


    /**
     * The String representation of this Matrix Market Exchange Format header.
     * @return This Matrix Market Exchange Format header as a string.
     */
    @Override
    public String toString() {
        StringJoiner joiner = new StringJoiner(" ")
                .add(MM_HEADER_PREFIX)
                .add(object.toString().toLowerCase())
                .add(format.toString().toLowerCase())
                .add(field.toString().toLowerCase())
                .add(symmetry.toString().toLowerCase());

        if(comments != null) {
            StringJoiner commentsJoiner = new StringJoiner(MM_COMMENT_PREFIX).add(MM_COMMENT_PREFIX);
            for(String comment : comments)
                commentsJoiner.add(comment);
            joiner.merge(commentsJoiner);
        }

        return joiner.add("\n%").toString();
    }


    /**
     * Enum containing valid object types in a Matrix Market file.
     */
    public enum MMObject {
        // TODO: Future extensions: VECTOR.
        MATRIX,
    }


    /**
     * Enum containing valid formats in a Matrix Market file.
     */
    public enum MMFormat {
        ARRAY,
        COORDINATE,
    }


    /**
     * Enum containing valid field types in a Matrix Market file.
     */
    public enum MMField {
        REAL,
        INTEGER,
        COMPLEX,
        PATTERN
    }


    /**
     * Enum containing valid symmetry in a Matrix Market file.
     */
    public enum MMSymmetry {
        // TODO: Future extensions: SKEW_SYMMETRIC.
        GENERAL,
        SYMMETRIC,
        HERMITIAN
    }
}

