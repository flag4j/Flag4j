/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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


/**
 * <p>The {@code Rotation} class provides utility methods to compute rotation matrices for 2D and 3D rotations.
 * These matrices are designed to perform rotations of vectors in a counterclockwise direction when viewed from a
 * positive axis perspective in a right-handed coordinate system.
 *
 * <p>This class supports:
 * <ul>
 *   <li>2D rotation matrices for rotating column vectors by a specified angle in radians.</li>
 *   <li>3D rotation matrices for rotating about the <span class="latex-inline">x</span>-axis,
 *   <span class="latex-inline">y</span>-axis, and <span class="latex-inline">z</span>-axis.</li>
 *   <li>3D rotation matrices for yaw-pitch-roll rotations.</li>
 *   <li>3D rotation matrices for arbitrary axis rotations.</li>
 *   <li>3D rotation matrices for proper Euler angle rotations.</li>
 * </ul>
 *
 * <p>Rotation matrices have the following properties:
 * <ul>
 *   <li>A rotation matrix is orthogonal: <span class="latex-inline">R<sup>-1</sup> = R<sup>T</sup></span>.</li>
 *   <li>Rotations preserve the length of vectors (orthogonal transformations).</li>
 *   <li>The inverse/transpose rotation matrix undoes the rotation:
 *   <span class="latex-inline">x = R<sup>T</sup>Rx = RR<sup>T</sup>x</span></li>
 *   <li>Successive rotations can be composed through matrix multiplication (rotation order is from right to left).</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 *         // Rotate a 2D vector by 45 degrees.
 *         double theta = Math.toRadians(45.0);
 *         Matrix rotation2D = Rotation.rotate2D(theta);
 *         Vector vector2D = new Vector(1, 0);  // A vector along the <span class="latex-inline">x</span>-axis
 *         Vector rotatedVector2D = rotation2D.mult(vector2D);
 *
 *         // Rotate a 3D vector about the <span class="latex-inline">x</span>-axis by 90 degrees.
 *         double thetaX = Math.toRadians(90.0);
 *         Matrix rotationX3D = Rotation.rotateX3D(thetaX);
 *         Vector vector3D = new Vector(0, 1, 0);  // A vector along the <span class="latex-inline">y</span>-axis
 *         Vector rotatedVector3D = rotationX3D.mult(vector3D);
 *
 *         // Perform a yaw-pitch-roll rotation in 3D.
 *         double yaw = Math.toRadians(0.0);
 *         double pitch = Math.toRadians(45.0);
 *         double roll = Math.toRadians(60.0);
 *         Matrix yawPitchRoll = Rotation.rotate3D(yaw, pitch, roll);
 *         vector3D = new Vector(1, 1, 1);  // An arbitrary 3D vector
 *         Vector rotatedVector = yawPitchRoll.mult(vector3D);
 *
 *         // Rotate a 3D vector about an arbitrary axis.
 *         Vector axis = new Vector(1, 1, 0);  // An arbitrary axis.
 *         double angle = Math.toRadians(45.0);
 *         Matrix arbitraryAxisRotation = Rotation.rotate3D(angle, axis);
 *         Vector arbitraryRotatedVector = arbitraryAxisRotation.mult(vector3D);
 *
 *         // Perform a rotation using proper Euler angles.
 *         double alpha = Math.PI / 6.0;  // Rotation about <span class="latex-inline">z</span>-axis
 *         double beta = Math.PI / 4.0;   // Rotation about <span class="latex-inline">x</span>-axis
 *         double gamma = Math.PI / 3.0;  // Rotation about <span class="latex-inline">z</span>-axis again
 *         Matrix eulerRotation = Rotation.rotateEuler3D(alpha, beta, gamma);
 *         Vector eulerRotatedVector = eulerRotation.mult(vector3D);
 *
 *         // Perform multiple rotations.
 *         double thetaY = Math.toRadians(90.0);
 *         double thetaZ = Math.toRadians(-30.0);
 *         Matrix rotationY3D = Rotation.rotateY3D(thetaY);
 *         Matrix rotationZ3D = Rotation.rotateZ3D(thetaZ);
 *         vector3D = new Vector(0, 1, 0);
 *         // First rotate by thetaY then by thetaZ.
 *         Matrix combinedRotation3D = rotationZ3D.mult(rotationY3D);
 *         Vector multiRotatedVector3D = combinedRotation3D.mult(vector3D);
 * }</pre>
 *
 * <p><strong>Note:</strong> Methods involving sequential rotations, such as yaw-pitch-roll or Euler angles, are
 * susceptible to gimbal lock, where two axes align, causing a loss of one degree of rotational freedom. For
 * applications requiring continuous rotations, consider alternative representations such as quaternions.
 */
public final class Rotation {

    private Rotation() {
        // Hide default constructor for utility class.
    }

    // TODO: Consider having an object-oriented class for rotations similar to scipy.spatial.transform.Rotation


    /**
     * <p>Constructs a rotation matrix, <span class="latex-inline">R(&theta;)</span>, which rotates 2D column vectors
     * <span class="latex-inline">&theta;</span> radians. When <span class="latex-inline">&theta; > 0</span> the rotation is
     * counterclockwise.
     *
     * <p>A 2D rotation matrix <span class="latex-inline">R(&theta;)</span>, rotates a 2D column vector 
     * <span class="latex-inline">x</span>, <span class="latex-inline">&theta;</span> radians by means of
     * the following matrix-vector multiplication:
     * <span class="latex-display"><pre>
     *     x' = R(&theta;)x</pre></span>
     * The following holds <span class="latex-inline">R(-&theta;) = R(&theta;)<sup>-1</sup> = R(&theta;)<sup>T</sup></span>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <span class="latex-eq-align"><pre>
     *     x = R(&theta;)R(&theta;)<sup>T</sup>x
     *       = R(&theta;)<sup>T</sup>R(&theta;)x
     *       = Ix</pre></span>
     *
     * @param theta The radians to rotate a 2D vector by.
     * @return A rotation matrix which rotates (counterclockwise) 2D column vectors {@code theta} radians.
     */
    public static Matrix rotate2D(double theta) {
        if(theta == 0) return Matrix.I(2);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        return new Matrix(2, 2, c, -s, s, c);
    }


    /**
     * <p>Constructs a matrix which rotates 3D column vectors about the <span class="latex-inline">x</span>-axis
     * <span class="latex-inline">&theta;</span> radians.
     * The rotation appears counterclockwise when the <span class="latex-inline">x</span>-axis points toward the observer,
     * <span class="latex-inline">&theta;</span> and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <span class="latex-inline">R<sub>x</sub>(&theta;)</span>, rotates a 3D column vector x about the
     * <span class="latex-inline">x</span>-axis <span class="latex-inline">&theta;</span>
     * radians by
     * means of the following matrix-vector multiplication:
     * <span class="latex-display"><pre>
     *     x' = R<sub>x</sub>(&theta;)x</pre></span>
     * The following holds
     * <span class="latex-inline">R<sub>x</sub>(-&theta;) = R<sub>x</sub>(&theta;)<sup>-1</sup>
     * = R<sub>x</sub>(&theta;)<sup>T</sup></span>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <span class="latex-eq-align"><pre>
     *     x = R<sub>x</sub>(&theta;)R<sub>x</sub>(&theta;)<sup>T</sup>x
     *       = R<sub>x</sub>(&theta;)<sup>T</sup>R<sub>x</sub>(&theta;)x
     *       = Ix</pre></span>
     *
     *
     * @param theta The radians to rotate a 3D vector about the <span class="latex-inline">x</span>-axis by.
     * @return matrix which rotates 3D column vectors about the <span class="latex-inline">x</span>-axis {@code theta} radians.
     */
    public static Matrix rotateX3D(double theta) {
        if(theta == 0) return Matrix.I(3);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        return new Matrix(3, 3,
                1, 0, 0,
                0, c, -s,
                0, s, c
        );
    }


    /**
     * <p>Constructs a matrix which rotates 3D column vectors about the <span class="latex-inline">y</span>-axis
     * <span class="latex-inline">&theta;</span> radians. The rotation appears
     * counterclockwise when the <span class="latex-inline">y</span>-axis points toward the observer,
     * <span class="latex-inline">&theta; > 0</span> and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <span class="latex-inline">R<sub>y</sub>(&theta;)</span>, rotates a 3D column vector x about the
     * <span class="latex-inline">y</span>-axis <span class="latex-inline">&theta;</span> radians by
     * means of the following matrix-vector multiplication:
     * <span class="latex-display"><pre>
     *     x' = R<sub>y</sub>(&theta;)x</pre></span>
     * The following holds
     * <span class="latex-inline">R<sub>y</sub>(-&theta;) = R<sub>y</sub>(&theta;)<sup>-1</sup>
     * = R<sub>y</sub>(&theta;)<sup>T</sup></span>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <span class="latex-eq-align"><pre>
     *     x = R<sub>y</sub>(&theta;)R<sub>y</sub>(&theta;)<sup>T</sup>x
     *       = R<sub>y</sub>(&theta;)<sup>T</sup>R<sub>y</sub>(&theta;)x
     *       = Ix</pre></span>
     *
     * @param theta The radians to rotate a 3D vector about the <span class="latex-inline">y</span>-axis by.
     * @return matrix which rotates 3D column vectors about the <span class="latex-inline">y</span>-axis {@code theta} radians.
     */
    public static Matrix rotateY3D(double theta) {
        if(theta == 0) return Matrix.I(3);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        return new Matrix(3, 3,
                c, 0, s,
                0, 1, 0,
                -s, 0, c
        );
    }


    /**
     * <p>Constructs a matrix which rotates 3D column vectors about the <span class="latex-inline">z</span>-axis
     * <span class="latex-inline">&theta;</span> radians. The rotation appears
     * counterclockwise when the <span class="latex-inline">z</span>-axis points toward the observer,
     * <span class="latex-inline">&theta; > 0</span> and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, R<sub>z</sub>(&theta;), rotates a 3D column vector x about the
     * <span class="latex-inline">z</span>-axis <span class="latex-inline">&theta;</span> radians by
     * means of the following matrix-vector multiplication:
     * <span class="latex-display"><pre>
     *     x' = R<sub>z</sub>(&theta;)x</pre></span>
     * The following holds <span class="latex-inline">R<sub>z</sub>(-&theta;) = R<sub>z</sub>(&theta;)<sup>-1</sup>
     * = R<sub>z</sub>(&theta;)<sup>T</sup></span>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <span class="latex-eq-align"><pre>
     *     x = R<sub>z</sub>(&theta;)R<sub>z</sub>(&theta;)<sup>T</sup>x
     *       = R<sub>z</sub>(&theta;)<sup>T</sup>R<sub>z</sub>(&theta;)x
     *       = Ix</pre></span>
     *
     * @param theta The radians to rotate a 3D vector about the <span class="latex-inline">z</span>-axis by.
     * @return matrix which rotates 3D column vectors about the <span class="latex-inline">z</span>-axis {@code theta} radians.
     */
    public static Matrix rotateZ3D(double theta) {
        if(theta == 0) return Matrix.I(3);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        return new Matrix(3, 3,
                c, -s, 0,
                s, c, 0,
                0, 0, 1
        );
    }


    /**
     * <p>Constructs a 3D rotation matrix, <span class="latex-eq-align">R(&alpha;, &beta;, &gamma;)</span>, representing a rotation
     * with yaw, pitch, and roll angles
     * <span class="latex-inline">&alpha;</span>, <span class="latex-inline">&beta;</span>, and
     * <span class="latex-inline">&gamma;</span> respectively. This is equivalent to rotating by
     * <span class="latex-inline">&alpha;</span>
     * radians about the <span class="latex-inline">x</span>-axis,
     * &beta; radians about the <span class="latex-inline">y</span>-axis, and
     * <span class="latex-inline">&gamma;</span> radians about the
     * <span class="latex-inline">z</span>-axis in that order. Each of the three rotations appear
     * counterclockwise when the axis about which they occur points toward the observer,
     * the rotation angle is positive, and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <span class="latex-inline">R(&alpha;, &beta;, &gamma;)</span>, rotates a 3D column vector
     * <span class="latex-inline">x</span>,
     * <span class="latex-inline">&gamma;</span>, <span class="latex-inline">&beta;</span>, and
     * <span class="latex-inline">&alpha;</span> radians about the <span class="latex-inline">x</span>-,
     * <span class="latex-inline">y</span>-, and
     * <span class="latex-inline">z</span>-axes in that order by means of
     * the following matrix
     * multiplication:
     * <span class="latex-eq-align"><pre>
     *     x' = R(&alpha;, &beta;, &gamma;)x
     *        = R<sub>z</sub>(&gamma;)R<sub>y</sub>(&beta;)R<sub>x</sub>(&alpha;)x</pre></span>
     *
     * <p><strong>Note:</strong> This method is susceptible to gimbal lock, a phenomenon where two of the rotation axes align,
     * causing a loss of one degree of rotational freedom. Gimbal lock occurs when the second rotation in the sequence aligns
     * the axes, such as when the pitch angle (for yaw-pitch-roll) or the second Euler angle (for proper Euler angles) is
     * <span class="latex-inline">&plusmn;90&deg;</span>. To avoid gimbal lock, consider using rotation representations that do not
     * rely on sequential rotations.
     *
     * @param yaw Radians to rotate about the vertical (yaw) axis (i.e., the <span class="latex-inline">z</span>-axis).
     * @param pitch Radians to rotate about the lateral (pitch) axis (i.e., the <span class="latex-inline">y</span>-axis).
     * @param roll Radians to rotate about the longitudinal (roll) axis (i.e., the <span class="latex-inline">x</span>-axis).
     * @return a rotation matrix representing a rotation with yaw, pitch, and roll angles <span class="latex-inline">&alpha;</span>,
     * <span class="latex-inline">&beta;</span>, and <span class="latex-inline">&gamma;</span> respectively.
     */
    public static Matrix rotate3D(double yaw, double pitch, double roll) {
        if(yaw == 0.0 && pitch == 0.0 && roll == 0.0) return Matrix.I(3);

        double ca = Math.cos(yaw);
        double sa = Math.sin(yaw);
        double cb = Math.cos(pitch);
        double sb = Math.sin(pitch);
        double cy = Math.cos(roll);
        double sy = Math.sin(roll);

        return new Matrix(3, 3,
                ca*cb,  ca*sb*sy - sa*cy,   ca*sb*cy + sa*sy,
                sa*cb,  sa*sb*sy + ca*cy,   sa*sb*cy - ca*sy,
                -sb,    cb*sy,              cb*cy
        );
    }


    /**
     * <p>Constructs a 3D rotation matrix, <span class="latex-inline">R<sub>u</sub>(&theta;)</span>, which representing a rotation of
     * <span class="latex-inline">&theta;</span> radians about
     * an axis unit vector u. The rotation is a counterclockwise rotation when u points towards the observer,
     * <span class="latex-inline">&theta; > 0</span>,
     * and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <span class="latex-inline">R<sub>u</sub>(&theta;)</span>, rotates a 3D column vector x,
     * <span class="latex-inline">&theta;</span> radians about the vector u by means of the following matrix multiplication:
     * <span class="latex-display"><pre>
     *     x' = R<sub>u</sub>(&theta;)x</pre></span>
     *
     * @param theta The radians to rotate about the vector <span class="latex-inline">u</span>.
     * @param axis The axis vector <span class="latex-inline">u</span> to rotate about. This vector will be normalized so it need not be a unit vector.
     * Must satisfy {@code axis.size == 3}.
     * @return A rotation matrix representing a rotation of {@code theta} radians about the axis specified
     * by {@code axis}.
     */
    public static Matrix rotate3D(double theta, Vector axis) {
        if(axis.size != 3)
            throw new IllegalArgumentException("Axis vector size must be size 3 but got " + axis.size + ".");

        if(theta == 0.0) return Matrix.I(3);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        double cInv = 1.0 - c;

        // Ensure vector is a unit vector.
        axis = axis.normalize();
        double ux = axis.data[0];
        double uy = axis.data[1];
        double uz = axis.data[2];

        // Construct using Rodrigues' formula: R = I + sin(t)*K + (1 - cos(t))*K^2 where
        //  where K is skew-symmetric with zeros on the diagonal:
        //      [  0  -uz  uy ]
        //  K = [  uz  0  -ux ]
        //      [ -uy  ux  0  ]
        double r01 = ux*uy*cInv;
        double r02 = ux*uz*cInv;
        double r12 = uy*uz*cInv;

        return new Matrix(3, 3,
                ux*ux*cInv + c, r01 - uz*s,     r02 + uy*s,
                r01 + uz*s,     uy*uy*cInv + c, r12 - ux*s,
                r02 - uy*s,     r12 + ux*s,     uz*uz*cInv + c
        );
    }


    /**
     * <p>Constructs a 3D rotation matrix, <span class="latex-inline">R<sub>E</sub>(&alpha;, &beta;, &gamma;)</span>,
     * representing a rotation described
     * by proper Euler angles <span class="latex-inline">(&alpha;, &beta;, &gamma;)</span>. This is equivalent to
     * performing a rotation about the
     * <span class="latex-inline">z</span>-axis by &alpha;
     * radians, then about the <span class="latex-inline">x</span>-axis by &beta; radians, then about the
     * <span class="latex-inline">z</span>-axis again by &gamma; radians.
     *
     * <p>A 3D rotation matrix, <span class="latex-inline">R<sub>E</sub>(&alpha;, &beta;, &gamma;)</span>, rotates a 3D column vector x,
     * according to the Euler angles <span class="latex-inline">(&alpha;, &beta;, &gamma;)</span> by means of the following
     * matrix multiplication:
     * <span class="latex-eq-align"><pre>
     *     x' = R<sub>E</sub>(&alpha;, &beta;, &gamma;)x
     *        = R<sub>z</sub>(&gamma;)R<sub>x</sub>(&beta;)R<sub>z</sub>(&alpha;)x</pre></span>
     *
     * <p><strong>Note:</strong> This method is susceptible to gimbal lock, a phenomenon where two of the rotation axes align,
     * causing a loss of one degree of rotational freedom. Gimbal lock occurs when the second rotation in the sequence aligns
     * the axes, such as when the pitch angle (for yaw-pitch-roll) or the second Euler angle (for proper Euler angles) is
     * <span class="latex-inline">&plusmn;90&deg;</span>. To avoid gimbal lock, consider using rotation representations that do not
     * rely on sequential rotations.
     *
     * @param alpha Radians of first rotation about the <span class="latex-inline">z</span>-axis.
     * @param beta Radians of second rotation about the <span class="latex-inline">x</span>-axis.
     * @param gamma Radians of third rotation about the <span class="latex-inline">z</span>-axis.
     * @return Constructs a rotation matrix representing a rotation described by proper Euler angles
     * <span class="latex-inline">(&alpha;, &beta;, &gamma;)</span>.
     */
    public static Matrix rotateEuler(double alpha, double beta, double gamma) {
        if(alpha == 0.0 && beta == 0.0 && gamma == 0.0) return Matrix.I(3);

        double ca = Math.cos(alpha);
        double sa = Math.sin(alpha);
        double cb = Math.cos(beta);
        double sb = Math.sin(beta);
        double cy = Math.cos(gamma);
        double sy = Math.sin(gamma);

        return new Matrix(3, 3,
                cy*ca - sy*sa*cb,   -sy*cb*ca - sa*cy,  sy*sb,
                sy*ca + sa*cy*cb,   cy*cb*ca - sy*sa,   -sb*cy,
                sb*sa,              sb*ca,              cb
        );
    }
}
