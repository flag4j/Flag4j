/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.StringJoiner;

/**
 * <p>The {@code TensorWriter} class provides static methods for serializing and saving tensors, matrices, and vectors to files.
 * Supported formats include:
 * <ul>
 *   <li>Binary serialization using {@link ObjectOutputStream}</li>
 *   <li>Comma-separated values (CSV) format for dense real, complex, and field matrices</li>
 *   <li><a href="https://math.nist.gov/MatrixMarket/formats.html">Matrix Market Exchange Format</a>
 *   for dense and sparse (COO or CSR) matrices (real or complex). Both COO and CSR matrices are written in
 *  the <em>coordinate (COO)</em> format. Symmetry is not considered and all matrices will be written as general matrices
 *  regardless of the symmetry.</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Write a real matrix to a Matrix Market Exchange Format file
 * Matrix matrix1 = new Matrix(...);
 * TensorWriter.toMatrixMarket("matrix1.mtx", matrix1);
 *
 * // Write a complex COO matrix to a Matrix Market Exchange Format file
 * CooCMatrix cooMatrix = new CooCMatrix(...);
 * TensorWriter.toMatrixMarket("cooMatrix.mtx", cooMatrix, "this is a comment");
 *
 * // Write a real matrix to a CSV file
 * Matrix matrix2 = new Matrix(...);
 * TensorWriter.toCsv("matrix2.csv", matrix2);
 * TensorWriter.toCsv("matrix2.csv", matrix2, ";");  // Specify delimiter.
 *
 * // Serialize a tensor to a binary file
 * AbstractNDArray tensor = new AbstractNDArray(...);
 * TensorWriter.write("tensor.ser", tensor);
 * }</pre>
 */
public final class TensorWriter {

    private TensorWriter() {
        // Hide default constructor in utility class.
    }

    /**
     * Writes a tensor to a file using a {@link ObjectOutputStream}. If the write operation fails it will
     * terminate but no exception will be thrown. To determine if the write was successful, see the return value: if
     * the write was successful, true is returned; if the write fails,then false is returned.
     * is returned.
     * @param fileName File name, including extension, of the file to write the serialized matrix to.
     * @param src Source object to write to the specified file.
     * @throws IOException If any I/O error occurs when attempting to write to file.
     */
    public static void write(String fileName, AbstractNDArray<?, ?, ?> src) throws IOException {
        boolean successfulWrite = true;

        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName))) {
            out.writeObject(src);
        }
    }


    /**
     * <p>Writes a matrix to a
     * <a href="https://math.nist.gov/MatrixMarket/formats.html">Matrix Market Exchange Format</a> file.
     * <p>If {@code src} is a CSR matrix, it will be converted to a COO matrix and saved as a coordinate matrix. All matrices are
     * stored in general form regardless of the symmetry.
     * <p>Supported matrices:
     * <ul>
     *   <li>{@code Matrix} (real dense)</li>
     *   <li>{@code CMatrix} (complex dense)</li>
     *   <li>{@code CooMatrix} (real sparse COO)</li>
     *   <li>{@code CsrMatrix} (real sparse CSR, converted to COO)</li>
     *   <li>{@code CooCMatrix} (complex sparse COO)</li>
     *   <li>{@code CsrCMatrix} (complex sparse CSR, converted to COO)</li>
     * </ul>
     *
     * @param fileName Path of the file to write to.
     * @param src Matrix to write to file.
     * @param comments Comments to prepend to file. Each comment will be written to its own line.
     * May be {@code null} or length zero (i.e., nothing passed); in this case the parameter will be ignored.
     * @throws IOException If an I/O error occurs.
     * @throws IllegalArgumentException If {@code src} is not a supported matrix type for writing to a
     * Matrix Market Exchange Format file.
     */
    public static void toMatrixMarket(String fileName, MatrixMixin<?, ?, ?, ?> src, String... comments) throws IOException {
        MatrixMarketWriter.write(fileName, src, comments);
    }


    /**
     * Writes a matrix to a CSV file. Matrix must be an instance of either {@link Matrix} or {@link CMatrix}.
     * Uses "," as the default delimiter. To specify the delimiter use {@link #toCsv(String, MatrixMixin, String)}.
     * @param fileName Name of the file to save matrix to.
     * @param src Matrix to save to CSV file.
     * @throws IOException If any I/O error occurs when attempting to write to file.
     * @see #toCsv(String, MatrixMixin, String)
     */
    public static void toCsv(String fileName, MatrixMixin<?, ?, ?, ?> src) throws IOException {
        toCsv(fileName, src, ",");
    }


    /**
     * Writes a matrix to a CSV file. Matrix must be an instance of either {@link Matrix} or {@link CMatrix}.
     * @param filePath Path of the file to save matrix to.
     * @param src Matrix to save to CSV file.
     * @throws IOException If any I/O error occurs when attempting to write to file.
     * @throws IllegalArgumentException If {@code !(src instanceof Matrix || src instanceof CMatrix)}.
     * @see #toCsv(String, MatrixMixin)
     */
    public static void toCsv(String filePath, MatrixMixin<?, ?, ?, ?> src, String delimiter) throws IOException {
        if(!(src instanceof Matrix || src instanceof CMatrix)) {
            throw new IllegalArgumentException("Unsupported matrix; cannot write matrix type to CSV: "
                    + src.getClass().getName());
        }
        int numRows = src.numRows();
        int numCols = src.numCols();

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(filePath))) {
            for (int i = 0; i < numRows; i++) {
                StringJoiner rowJoiner = new StringJoiner(delimiter);

                for (int j = 0; j < numCols; j++)
                    rowJoiner.add(src.get(i, j).toString());

                writer.write(rowJoiner.toString());
                writer.newLine();
            }
        }
    }
}
