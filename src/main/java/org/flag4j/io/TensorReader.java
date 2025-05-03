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

import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;

import java.io.*;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * The TensorReader class provides several static methods for reading serialized
 * tensors, matrices, and vectors from a file.
 */
public final class TensorReader {

    private TensorReader() {
        // Hide default constructor in utility class.
    }


    /**
     * <p>Reads a serialized {@link AbstractTensor} from a specified file using a {@link ObjectInputStream}.
     * <p>The object returned from this tensor will likely need to be cast to the desired type:
     * <pre>{@code
     *      Matrix matrix;
     *      try {
     *          matrix = (Matrix) TensorReader.read("some_file.ser")
     *      } catch(IOException | ClassNotFoundException | ClassCastException e) {
     *          // Handel exception.
     *      }
     * }</pre>
     *
     * @param filePath Path of the file containing the serialized tensor.
     * @return The deserialized tensor from the specified file.
     * @throws IOException If an I/O exception occurs while reading the object.
     * @throws ClassNotFoundException Class of a serialized object cannot be found.
     */
    public static AbstractTensor<?, ?, ?> read(String filePath) throws IOException, ClassNotFoundException {
        Object object = null;
        File file = new File(filePath);

        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            object = in.readObject();
            return (AbstractTensor<?, ?, ?>) object;
        } catch(ClassCastException e) {
            throw new IOException("Attempted to load a non-Flag4j tensor object: " + object.getClass().getName(), e);
        }
    }


    /**
     * <p>Reads a CSV file into a matrix of the specified type. This method uses {@code ","} as the default delimiter in the CSV file.
     * If another delimiter is desired, specify using {@link #fromCsv(String, String, Class)}.
     *
     * <p>Usage example:
     * <pre>{@code
     *      Matrix realMatrix;
     *      try {
     *          realMatrix = (Matrix) TensorRead.fromCsv("real_data.csv", Matrix.class);
     *      } catch (IOException e) {
     *          // Handel exception.
     *      }
     *
     *      CMatrix complexMatrix;
     *      try {
     *          complexMatrix = (CMatrix) TensorRead.fromCsv("complex_data.csv", CMatrix.class);
     *      } catch (IOException e) {
     *          // Handel exception.
     *      }
     * }</pre>
     *
     * @param fileName   Path to the CSV file.
     * @param matrixType The class object of the desired matrix to read into. Must be either
     * {@code Matrix.class} or {@code CMatrix.class}.
     * @return A MatrixMixin object containing the data from the CSV file.
     * @throws IOException              If an I/O error occurs while reading the file.
     * @throws IllegalArgumentException If the CSV file is malformed or the {@code matrixType} isn't supported.
     * @see #fromCsv(String, String, Class)
     */
    public static MatrixMixin<?, ?, ?, ?> fromCsv(
            String fileName, Class<? extends MatrixMixin<?, ?, ?, ?>> matrixType) throws IOException {
        return fromCsv(fileName, ",", matrixType);
    }


    /**
     * <p>Reads a CSV file into a matrix of the specified type.
     *
     * <p>Usage example:
     * <pre>{@code
     *      Matrix realMatrix;
     *      try {
     *          realMatrix = (Matrix) TensorRead.fromCsv("real_data.csv", ",", Matrix.class);
     *      } catch (IOException e) {
     *          // Handel exception.
     *      }
     *
     *      CMatrix complexMatrix;
     *      try {
     *          complexMatrix = (CMatrix) TensorRead.fromCsv("complex_data.csv", ",", CMatrix.class);
     *      } catch (IOException e) {
     *          // Handel exception.
     *      }
     * }</pre>
     *
     * @param fileName   Path to the CSV file.
     * @param delimiter  Delimiter used in the CSV file.
     * @param matrixType The class object of the desired matrix to read into. Must be either {@code Matrix.class} or {@code CMatrix
     * .class}.
     * @return A MatrixMixin object containing the data from the CSV file.
     * @throws IOException              If an I/O error occurs while reading the file.
     * @throws IllegalArgumentException If the CSV file is malformed or the matrixType isn't supported.
     * #fromCsv(String,  Class)
     */
    public static MatrixMixin<?, ?, ?, ?> fromCsv(
            String fileName, String delimiter,
            Class<? extends MatrixMixin<?, ?, ?, ?>> matrixType) throws IOException {

        if (matrixType == Matrix.class) {
            // Parse dense double matrix.
            Pair<Shape, List<Double>> data = fromCsv(fileName, delimiter, Double::parseDouble);
            double[] arr = ArrayConversions.fromDoubleList(data.second());
            Constructor<? extends MatrixMixin<?, ?, ?, ?>> constructor = null;
            try {
                constructor = matrixType.getConstructor(Shape.class, double[].class);
                return constructor.newInstance(data.first(), arr);
            } catch(ReflectiveOperationException e) {
                throw new IllegalArgumentException("Unable to construct matrix type: "
                        + matrixType.getName(), e);
            }
        } else if (matrixType == CMatrix.class) {
            // Parse Complex128 matrix.
            Pair<Shape, List<Complex128>> data = fromCsv(fileName, delimiter, Complex128::new);
            Complex128[] arr = data.second().toArray(new Complex128[0]);
            Constructor<? extends MatrixMixin<?, ?, ?, ?>> constructor = null;

            try {
                constructor = matrixType.getConstructor(Shape.class, Complex128[].class);
                return constructor.newInstance(data.first(), arr);
            } catch(ReflectiveOperationException e) {
                throw new IllegalArgumentException("Unable to construct matrix type: "
                        + matrixType.getName(), e);
            }
        } else {
            throw new IllegalArgumentException("Unsupported matrix type; cannot read from CSV file: " + matrixType.getName());
        }
    }


    /**
     * Loads a dense matrix from a CSV file.
     *
     * @param fileName  Path to the CSV file.
     * @param delimiter Delimiter used in the CSV file (e.g., "," or ";"). May be regex.
     * @param parseFunction Function used to parse an individual entry/token from the CSV file to the desired object type.
     * @return A {@link Pair pair} containing the {@link Shape shape} of the matrix and the data from the CSV file in a list.
     * @throws IOException If an I/O error occurs while reading the file.
     * @throws IllegalArgumentException If the CSV file is malformed (e.g., inconsistent row lengths).
     */
    private static <T> Pair<Shape, List<T>> fromCsv(
            String fileName, String delimiter, Function<String, T> parseFunction) throws IOException {
        if (fileName == null || delimiter == null || delimiter.isEmpty())
            throw new IllegalArgumentException("File path and delimiter must not be null or empty.");

        List<T> data = new ArrayList<>();
        int numCols = -1;
        int numRows = 0;

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(fileName))) {
            String line;

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(delimiter, -1); // Include trailing empty tokens.
                if (numCols == -1)
                    numCols = tokens.length; // Determine number of columns from the first row.
                else if (tokens.length != numCols)
                    throw new IllegalArgumentException("Malformed CSV file: inconsistent number of columns.");

                // Add values for current row.
                for (String token : tokens) {
                    try {
                        data.add(parseFunction.apply(token));
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("Invalid numeric value in CSV: " + token, e);
                    }
                }

                numRows++;
            }
        }

        return new Pair<>(new Shape(numRows, numCols), data);
    }
}
