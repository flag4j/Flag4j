package org.flag4j.linalg.transformations;

import org.flag4j.CustomAssertions;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.Flag4jConstants;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class RotationTests {

    @Test
    void rotate2DTests() {
        double theta;
        Shape expShape;
        double[] expData;
        Matrix exp;

        // ---------------------- sub-case 1 ----------------------
        theta = Math.toRadians(90);
        expShape = new Shape(2, 2);
        expData = new double[]{
                0.0, -1.0000000000000002,
                1.0000000000000002, 0.0};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate2D(theta), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 2 ----------------------
        theta = 0;
        expShape = new Shape(2, 2);
        expData = new double[]{1.0, 0.0, 0.0, 1.0};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate2D(theta), 2.0*Flag4jConstants.EPS_F64);
        

        // ---------------------- sub-case 3 ----------------------
        theta = Math.toRadians(-25.2);
        expShape = new Shape(2, 2);
        expData = new double[]{
                0.9048270524660196, 0.42577929156507266,
                -0.42577929156507266, 0.9048270524660196};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate2D(theta), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 4 ----------------------
        theta = Math.toRadians(6725.2);
        expShape = new Shape(2, 2);
        expData = new double[]{
                -0.41945208244618426, 0.9077774785329054,
                -0.9077774785329054, -0.41945208244618426};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate2D(theta), 2.0*Flag4jConstants.EPS_F64);
    }


    @Test
    void singleRotate3DTests() {
        double theta;
        Shape expXShape, expYShape, expZShape;
        double[] expXData, expYData, expZData;
        Matrix expX, expY, expZ;

        // ---------------------- sub-case 1 ----------------------
        theta = Math.toRadians(90);
        expXShape = new Shape(3, 3);
        expXData = new double[]{
                1.0, 0.0, 0.0,
                0.0, 6.123233995736766E-17, -1.0,
                0.0, 1, 6.123233995736766E-17};
        expX = new Matrix(expXShape, expXData);
        expYShape = new Shape(3, 3);
        expYData = new double[]{
                0.0, 0.0, 1.0000000000000002,
                0.0, 1.0000000000000002, 0.0,
                -1.0000000000000002, 0.0, 0.0};
        expY = new Matrix(expYShape, expYData);
        expZShape = new Shape(3, 3);
        expZData = new double[]{
                0.0, -1.0000000000000002, 0.0,
                1.0000000000000002, 0.0, 0.0,
                0.0, 0.0, 1.0000000000000002};
        expZ = new Matrix(expZShape, expZData);

        CustomAssertions.assertEquals(expX, Rotation.rotateX3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expY, Rotation.rotateY3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expZ, Rotation.rotateZ3D(theta), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 2 ----------------------
        theta = Math.toRadians(0.0);
        expXShape = new Shape(3, 3);
        expXData = new double[]{
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};
        expX = new Matrix(expXShape, expXData);
        expYShape = new Shape(3, 3);
        expYData = new double[]{
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};
        expY = new Matrix(expYShape, expYData);
        expZShape = new Shape(3, 3);
        expZData = new double[]{
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};
        expZ = new Matrix(expZShape, expZData);

        CustomAssertions.assertEquals(expX, Rotation.rotateX3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expY, Rotation.rotateY3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expZ, Rotation.rotateZ3D(theta), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 3 ----------------------
        theta = Math.toRadians(-23.224);
        expXShape = new Shape(3, 3);
        expXData = new double[]{
                1.0, -0.0, 0.0,
                0.0, 0.9189702446232713, 0.39432688152983575,
                -0.0, -0.39432688152983575, 0.9189702446232713};
        expX = new Matrix(expXShape, expXData);
        expYShape = new Shape(3, 3);
        expYData = new double[]{
                0.9189702446232713, -0.0, -0.39432688152983575,
                0.0, 1.0, -0.0,
                0.39432688152983575, 0.0, 0.9189702446232713};
        expY = new Matrix(expYShape, expYData);
        expZShape = new Shape(3, 3);
        expZData = new double[]{
                0.9189702446232713, 0.39432688152983575, 0.0,
                -0.39432688152983575, 0.9189702446232713, -0.0,
                -0.0, 0.0, 1.0};
        expZ = new Matrix(expZShape, expZData);

        CustomAssertions.assertEquals(expX, Rotation.rotateX3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expY, Rotation.rotateY3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expZ, Rotation.rotateZ3D(theta), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 4 ----------------------
        theta = Math.toRadians(-15623.224);
        expXShape = new Shape(3, 3);
        expXData = new double[]{
                0.9999999999999999, 0.0, 0.0,
                0.0, -0.8009822191115815, 0.5986881364008194,
                0.0, -0.5986881364008194, -0.8009822191115815};
        expX = new Matrix(expXShape, expXData);
        expYShape = new Shape(3, 3);
        expYData = new double[]{
                -0.8009822191115815, 0.0, -0.5986881364008194,
                0.0, 0.9999999999999999, 0.0,
                0.5986881364008194, 0.0, -0.8009822191115815};
        expY = new Matrix(expYShape, expYData);
        expZShape = new Shape(3, 3);
        expZData = new double[]{
                -0.8009822191115815, 0.5986881364008194, 0.0,
                -0.5986881364008194, -0.8009822191115815, 0.0,
                0.0, 0.0, 0.9999999999999999};
        expZ = new Matrix(expZShape, expZData);

        CustomAssertions.assertEquals(expX, Rotation.rotateX3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expY, Rotation.rotateY3D(theta), 2.0*Flag4jConstants.EPS_F64);
        CustomAssertions.assertEquals(expZ, Rotation.rotateZ3D(theta), 2.0*Flag4jConstants.EPS_F64);
    }


    @Test
    void yawPitchRollTests() {
        double yaw, pitch, roll;
        Shape expShape;
        double[] expData;
        Matrix exp;

        // ---------------------- sub-case 1 ----------------------
        yaw = Math.toRadians(90);
        pitch = Math.toRadians(45);
        roll = Math.toRadians(-30);
        expShape = new Shape(3, 3);
        expData = new double[]{
                0.0, -0.8660254037844387, -0.4999999999999999,
                0.7071067811865476, -0.3535533905932737, 0.6123724356957946,
                -0.7071067811865476, -0.35355339059327373, 0.6123724356957946};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate3D(yaw, pitch, roll), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 2 ----------------------
        yaw = 0;
        pitch = 0;
        roll = 0;
        expShape = new Shape(3, 3);
        expData = new double[]{
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate3D(yaw, pitch, roll), 2.0*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 3 ----------------------
        yaw = Math.toRadians(-294.23);
        pitch = Math.toRadians(360);
        roll = Math.toRadians(14.3);
        expShape = new Shape(3, 3);
        expData = new double[]{
                0.4104005626044148, -0.883650635471688, 0.22523972261673023,
                0.9119053559520196, 0.39768460134190775, -0.10136853378414892,
                2.220446049250313e-16, 0.2469990127227429, 0.9690157314068695};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate3D(yaw, pitch, roll), 2.0*Flag4jConstants.EPS_F64);
    }


    @Test
    void rotateAboutVecTests() {
        double deg;
        Shape expShape, axisShape;
        double[] expData, axisData;
        Matrix exp;
        Vector axis;

        // ---------------------- sub-case 1 ----------------------
        deg = Math.toRadians(90);
        axisShape = new Shape(3);
        axisData = new double[]{0, 1, 0};
        axis = new Vector(axisShape, axisData);
        expShape = new Shape(3, 3);
        expData = new double[]{0.0, 0.0, 0.9999999999999998, 0.0, 0.9999999999999998, 0.0, -0.9999999999999998, 0.0, 0.0};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate3D(deg, axis), 2*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 2 ----------------------
        deg = Math.toRadians(-83.151);
        axisShape = new Shape(3);
        axisData = new double[]{5.0, 13.0, 34.2223};
        axis = new Vector(axisShape, axisData);
        expShape = new Shape(3, 3);
        expData = new double[]{0.13538205618433236, 0.9615505688971, -0.23893974620595243, -0.8776801020892495,
                0.22828472710995135, 0.4213831056947335, 0.4597274597627448, 0.15266494955292687, 0.8748394572252074};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotate3D(deg, axis), 1.0e-12);

        // ---------------------- sub-case 3 ----------------------
        assertThrows(IllegalArgumentException.class, ()->Rotation.rotate3D(45, new Vector(1, 1, 1, 1)));
        assertThrows(IllegalArgumentException.class, ()->Rotation.rotate3D(45, new Vector(1, 1)));
    }


    @Test
    void eulerAngleTests() {
        double a, b, c;
        Shape expShape;
        double[] expData;
        Matrix exp;

        // ---------------------- sub-case 1 ----------------------
        a = Math.toRadians(30);
        b = Math.toRadians(40);
        c = Math.toRadians(50);
        expShape = new Shape(3, 3);
        expData = new double[]{0.26325835480968673, -0.8295983733257066, 0.49240387650610407, 0.9096158864219905, 0.04341204441673252, -0.41317591116653474, 0.3213938048432696, 0.5566703992264194, 0.7660444431189781};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotateEuler(a, b, c), 2*Flag4jConstants.EPS_F64);

        // ---------------------- sub-case 2 ----------------------
        a = Math.toRadians(4.125);
        b = Math.toRadians(0);
        c = Math.toRadians(-93125.2);
        expShape = new Shape(3, 3);
        expData = new double[]{-0.4836643306997947, -0.8752535719485638, -0.0, 0.8752535719485638, -0.4836643306997947, 0.0, 0.0, -0.0, 1.0};
        exp = new Matrix(expShape, expData);

        CustomAssertions.assertEquals(exp, Rotation.rotateEuler(a, b, c), 2*Flag4jConstants.EPS_F64);
    }
}
