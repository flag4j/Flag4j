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

package org.flag4j.linalg.transformations;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.ValidateParameters;

/**
 * Utility class containing static for generating view matrices.
 */
public final class View {

    private View() {
        // Hide default constructor for utility class.
    }


    /**
     * Constructs a view matrix for a camera at world position {@code cameraPos}, looking at the point {@code center} where {@code up}
     * is the direction of the upwards vector in world coordinates. This function is identical to glm's lookAt function.
     * @param cameraPos Camera position in world coordinates. Must have length 3.
     * @param center The point the camera is looking at. Must have length 3.
     * @param up The vector in the direction fo the worlds 'upward' vector. Must have length 3.
     * @return A view matrix for a camera at world position {@code cameraPos}, looking at the point {@code center} where {@code up}
     * is the direction of the upwards vector in world coordinates.
     * @throws IllegalArgumentException If any of the argument vectors do not have length 3.
     */
    public static Matrix lookAt(Vector cameraPos, Vector center, Vector up) {
        ValidateParameters.ensureAllEqual(3, cameraPos.size, center.size, up.size);

        Vector f = center.sub(cameraPos).normalize();
        Vector u = up.normalize();
        Vector s = f.cross(u).normalize();
        u = s.cross(f);

        Matrix view = new Matrix(4);

        view.data[0] = s.data[0];
        view.data[4] = s.data[1];
        view.data[8] = s.data[2];

        view.data[1] = u.data[0];
        view.data[5] = u.data[1];
        view.data[9] = u.data[2];

        view.data[2] = -f.data[0];
        view.data[6] = -f.data[1];
        view.data[10] = -f.data[2];

        view.data[12] = -s.inner(cameraPos);
        view.data[13] = -u.inner(cameraPos);
        view.data[14] = f.inner(cameraPos);

        view.data[15] = 1.0;

        return view;
    }
}
