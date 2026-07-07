package org.flag4j.linalg.transformations;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ProjectionTests {

    double[][] expEntries;
    Matrix expProjection;


    @Test
    void perspectiveTestCase() {
        // -------------------------- sub-case 1 --------------------------
        expEntries = new double[][]{
                {-1.0223770614041234, 0, 0, 0},
                {0, -1.0223770614041234, 0, 0},
                {0, 0, -1.0066889632107023, -1},
                {0, 0, -0.20066889632107024, 0}
        };
        expProjection = new Matrix(expEntries);

        assertEquals(expProjection, Projection.getPerspective(55, 1, 0.1, 30));

        // -------------------------- sub-case 2 --------------------------
        expEntries = new double[][]{
                {0.40088936609321757, 0, 0, 0},
                {0, 0.6173696237835551, 0, 0},
                {0, 0, -1.0001200072004321, -1},
                {0, 0, -0.6000360021601296, 0}
        };
        expProjection = new Matrix(expEntries);

        assertEquals(expProjection, Projection.getPerspective(90, 1.54, 0.3, 5000));

        // -------------------------- sub-case 3 --------------------------
        expEntries = new double[][]{
                {0.6493506493506495, 0, 0, 0},
                {0, 1.5696855771174902, 0, 0},
                {0, 0, -1.0001200072004321, -1},
                {0, 0, -0.6000360021601296, 0}
        };
        expProjection = new Matrix(expEntries);

        assertEquals(expProjection, Projection.getPerspective(90, 65, 1.54, 0.3, 5000));
    }


    @Test
    void orthogonalTestCase() {
        // -------------------------- sub-case 1 --------------------------
        expEntries = new double[][]{
                {0.0392156862745098, 0, 0, 0},
                {0, 0.2857142857142857, 0, 0},
                {0, 0, -0.020100502512562814, 0},
                {-0.9607843137254902, -0.42857142857142855, -1.0100502512562815, 1}
        };
        expProjection = new Matrix(expEntries);

        assertEquals(expProjection, Projection.getOrthogonal(-1, 50, -2, 5, 0.5, 100));

        // -------------------------- sub-case 2 --------------------------
        expProjection = Projection.getOrthogonal(0, 5.4, 0, 80.1, 0.1, 10);
        assertEquals(expProjection, Projection.getOrthogonal(5.4, 80.1, 0.1, 10));

        // -------------------------- sub-case 3 --------------------------
        expProjection = Projection.getOrthogonal(-1, 50, -2, 5, -1, 1).round(10);
        assertEquals(expProjection, Projection.getOrthogonal2D(-1, 50, -2, 5).round(10));

        // -------------------------- sub-case 4 --------------------------
        expProjection = Projection.getOrthogonal(0, 50, 0, 5, -1, 1).round(10);
        assertEquals(expProjection, Projection.getOrthogonal2D(50, 5).round(10));
    }
}
