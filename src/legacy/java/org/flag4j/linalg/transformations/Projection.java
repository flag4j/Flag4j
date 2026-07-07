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
import org.flag4j.util.ValidateParameters;

/**
 * A utility class for generating projection matrices used in 3D graphics transformations.
 *
 * <h2>Supported Projections:</h2>
 * <ul>
 *   <li>Support for creating perspective projection matrices with specified field-of-view (FOV),
 *       aspect ratio, and near/far clipping planes.</li>
 *   <li>Support for creating orthogonal projection matrices with customizable boundaries
 *       (xMin, xMax, yMin, yMax) and clipping planes.</li>
 * </ul>
 *
 * <h2>Examples:</h2>
 * <pre>{@code
 * // Create a perspective projection matrix.
 * Matrix perspective = Projection.getPerspective(
 *      Math.toRadians(60), 16.0/9.0, 0.1, 100.0
 * );
 *
 * // Create an orthogonal projection matrix.
 * Matrix orthogonal = Projection.getOrthogonal(
 *      -10, 10, -10, 10, 0.1, 100.0
 * );
 * }</pre>
 */
public final class Projection {

    private Projection() {
        // Hide constructor for utility class. for utility class.
    }

    // TODO: Consider having an object-oriented class for projections similar to scipy.spatial.transform.Rotation


    /**
     * Creates a <span class="latex-inline">4&times;4</span> perspective projection matrix to transform a 3D point
     * represented in homogeneous
     * coordinates.
     *
     * @param fov FieldElement of view in radians (this is the fov in both the {@code x} and {@code y} directions).
     *            For distinct field of views see {@link #getPerspective(double, double, double, double, double)}.
     * @param aspectRatio Aspect ratio of the image plane to project to (i.e., {@code width/height}).
     * @param nearClip The distance from the camera to the near clipping plane.
     * @param farClip The distance from the camera to the far clipping plane.
     * @return The perspective projection matrix based on the provided attributes.
     * @throws IllegalArgumentException If {@code aspectRatio}  is not positive.
     * @throws AssertionError If {@code nearClip!=farClip}.
     */
    public static Matrix getPerspective(double fov, double aspectRatio, double nearClip, double farClip) {
        ValidateParameters.ensureAllGreaterEq(0, aspectRatio);
        if(nearClip >= farClip)
            throw new IllegalArgumentException("nearClip cannot be equal to or greater than farClip.");

        Matrix perspective = new Matrix(4);

        double cotHalfFov = 1.0/Math.tan(fov/2.0);

        perspective.data[0] = cotHalfFov/aspectRatio;
        perspective.data[5] = cotHalfFov;
        perspective.data[10] = -(farClip + nearClip)/(farClip - nearClip);
        perspective.data[11] = -1.0;
        perspective.data[14] = -2.0*farClip*nearClip/(farClip - nearClip);

        return perspective;
    }


    /**
     * Creates a <span class="latex-inline">4&times;4</span> perspective projection matrix to transform a
     * 3D point represented in homogeneous coordinates.
     *
     * @param fovX FieldElement of view, in radians, in the {@code x} direction.
     * @param fovY FieldElement of view, in radians, in the {@code y} direction.
     * @param aspectRatio Aspect ratio of the image plane to project to (i.e., {@code width/height}).
     * @param nearClip The distance from the camera to the near clipping plane.
     * @param farClip The distance from the camera to the far clipping plane.
     * @return The perspective projection matrix based on the provided attributes.
     * @throws IllegalArgumentException If {@code aspectRatio}  is not positive.
     * @throws AssertionError If {@code nearClip!=farClip}.
     */
    public static Matrix getPerspective(double fovX, double fovY, double aspectRatio, double nearClip, double farClip){
        ValidateParameters.ensureAllGreaterEq(0, aspectRatio);
        if(nearClip >= farClip)
            throw new IllegalArgumentException("nearClip cannot be equal to or greater than farClip.");

        Matrix perspective = new Matrix(4);

        // Convert the field of views to radians.
        double fovXRad = Math.toRadians(fovX);
        double fovYRad = Math.toRadians(fovY);

        perspective.data[0] = 1.0/(aspectRatio*Math.tan(fovXRad/2.0));
        perspective.data[5] = 1.0/(Math.tan(fovYRad/2.0));
        perspective.data[10] = -(farClip + nearClip)/(farClip - nearClip);
        perspective.data[11] = -1.0;
        perspective.data[14] = -2.0*farClip*nearClip/(farClip - nearClip);

        return perspective;
    }


    /**
     * Creates a <span class="latex-inline">4&times;4</span> orthogonal projection matrix to project a 3D point in homogeneous coordinates
     * onto the specified 2D coordinate grid (i.e., image plane).
     * This is an orthographic projection meaning the distance from the virtual camera will not affect the projection.
     * @param xMin Minimum {@code x} value of image plane to project to.
     * @param xMax Maximum {@code x} value of image plane to project to.
     * @param yMin Minimum {@code y} value of image plane to project to.
     * @param yMax Maximum {@code y} value of image plane to project to.
     * @param nearClip Distance from camera to near clipping plane.
     * @param farClip Distance from camera to far clipping plane.
     * @return The orthogonal projection for the specified parameters.
     */
    public static Matrix getOrthogonal(double xMin, double xMax, double yMin, double yMax,
                                          double nearClip, double farClip){
        if(xMin>=xMax)
            throw new IllegalArgumentException("xMin cannot be greater than or equal to xMax.");
        if(yMin>=yMax)
            throw new IllegalArgumentException("yMin cannot be greater than or equal to yMax.");
        if(nearClip >= farClip)
            throw new IllegalArgumentException("nearClip cannot be equal to or greater than farClip.");
        Matrix ortho = new Matrix(4);

        ortho.data[0] = 2.0/(xMax - xMin);
        ortho.data[5] = 2.0/(yMax - yMin);
        ortho.data[10] = -2.0/(farClip-nearClip);
        ortho.data[12] = -(xMax + xMin)/(xMax - xMin);
        ortho.data[13] = -(yMax + yMin)/(yMax - yMin);
        ortho.data[14] = -(farClip + nearClip)/(farClip - nearClip);
        ortho.data[15] = 1.0;

        return ortho;
    }


    /**
     * Creates a <span class="latex-inline">4&times;4</span> orthogonal projection matrix to project a 3D point in homogeneous coordinates
     * onto the specified 2D coordinate grid (i.e., image plane). Here, the minimum {@code x} and {@code y} values are
     * taken to be zero.
     * This is an orthographic projection meaning the distance from the virtual camera will not affect the projection.
     * @param xMax Maximum {@code x} value of image plane to project to.
     * @param yMax Maximum {@code y} value of image plane to project to.
     * @param nearClip Distance from camera to near clipping plane.
     * @param farClip Distance from camera to far clipping plane.
     * @return The orthogonal projection for the specified parameters.
     */
    public static Matrix getOrthogonal(double xMax, double yMax,
                                          double nearClip, double farClip){
        if(nearClip >= farClip)
            throw new IllegalArgumentException("nearClip cannot be equal to or greater than farClip.");
        Matrix ortho = new Matrix(4);

        ortho.data[0] = 2.0/xMax;
        ortho.data[5] = 2.0/yMax;
        ortho.data[10] = -2.0/(farClip - nearClip);
        ortho.data[12] = -1.0;
        ortho.data[13] = -1.0;
        ortho.data[14] = -(farClip + nearClip)/(farClip - nearClip);
        ortho.data[15] = 1.0;

        return ortho;
    }



    /**
     * Creates a <span class="latex-inline">4&times;4</span> orthogonal projection matrix to project a 2D point in an orthographic
     * viewing region.
     * Equivalent to {@link #getOrthogonal(double, double, double, double, double, double)} with {@code nearClip = -1} and
     * {@code farClip = 1}.
     * @param xMin Minimum {@code x} value of image plane to project to.
     * @param xMax Maximum {@code x} value of image plane to project to.
     * @param yMin Minimum {@code y} value of image plane to project to.
     * @param yMax Maximum {@code y} value of image plane to project to.
     * @return The orthogonal projection for the specified parameters.
     */
    public static Matrix getOrthogonal2D(double xMin, double xMax, double yMin, double yMax){
        if(xMin>=xMax)
            throw new IllegalArgumentException("xMin cannot be greater than or equal to xMax.");
        if(yMin>=yMax)
            throw new IllegalArgumentException("yMin cannot be greater than or equal to yMax.");

        Matrix ortho = new Matrix(4);

        ortho.data[0] = 2.0/(xMax-xMin);
        ortho.data[5] = 2.0/(yMax-yMin);
        ortho.data[10] = -1.0;
        ortho.data[12] = -(xMax + xMin)/(xMax - xMin);
        ortho.data[13] = -(yMax + yMin)/(yMax - yMin);
        ortho.data[14] = 0.0;
        ortho.data[15] = 1.0;

        return ortho;
    }


    /**
     * Creates a <span class="latex-inline">4&times;4</span> orthogonal projection matrix to project a
     * 2D point in an orthographic viewing region. The
     * minimum {@code x} and {@code y} values are assumed to be zero. Equivalent to
     * {@link #getOrthogonal(double, double, double, double)} with {@code nearClip=-1} and
     * {@code farClip = 1}.
     * @param xMax Maximum {@code x} value of image plane to project to.
     * @param yMax Maximum {@code y} value of image plane to project to.
     * @return The orthogonal projection for the specified parameters.
     */
    public static Matrix getOrthogonal2D(double xMax, double yMax){
        if(xMax < 0 || yMax < 0)
            throw new IllegalArgumentException("xMax and yMax must be non-negative.");

        Matrix ortho = new Matrix(4);

        ortho.data[0] = 2.0/xMax;
        ortho.data[5] = 2.0/yMax;
        ortho.data[10] = -1.0;
        ortho.data[12] = -1.0;
        ortho.data[13] = -1.0;
        ortho.data[14] = 0.0;
        ortho.data[15] = 1.0;

        return ortho;
    }
}
