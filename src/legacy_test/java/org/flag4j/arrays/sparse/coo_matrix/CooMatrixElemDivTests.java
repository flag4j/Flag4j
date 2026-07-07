package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseCooMatOps;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixElemDivTests {
    Shape aShape, expShape;
    double[] aEntries;
    CooMatrix A;
    int[] aRowIndices, aColIndices, expRowIndices, expColIndices;

    @Test
    void realSparseRealDenseElemDivTest() {
        double[][] bEntries;
        Matrix B;
        double[] expEntries;
        CooMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.4438057635382914, 0.1584073685290338, 0.6422718458358927, 0.7588801607997614, 0.21918998350329344};
        aRowIndices = new int[]{0, 1, 2, 3, 4};
        aColIndices = new int[]{3, 1, 1, 0, 0};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.66839, 0.35581, 0.69694, 0.95199, 0.46608},
                {0.9729, 0.84525, 0.6302, 0.97971, 0.81272},
                {0.34341, 0.53687, 0.77891, 0.74275, 0.9868},
                {0.96654, 0.59343, 0.96977, 0.06438, 0.24247},
                {0.32838, 0.71464, 0.29847, 0.86053, 0.886}};
        B = new Matrix(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new double[]{0.466187421651794, 0.18740889503582825, 1.196326570372516, 0.785151324104291, 0.6674888345919162};
        expRowIndices = new int[]{0, 1, 2, 3, 4};
        expColIndices = new int[]{3, 1, 1, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealDenseSparseMatrixOps.elemDiv(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.022415659127439036, 0.4427456213990437, 0.5909502892607543};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 1, 3};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.49565, 0.99507, 0.70204, 0.32258, 0.96505},
                {0.83145, 0.97469, 0.41509, 0.15008, 0.0034},
                {0.9897, 0.80495, 0.22398, 0.2536, 0.43284}};
        B = new Matrix(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.06948868227242556, 0.45424249904999925, 3.9375685585071585};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{3, 1, 3};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealDenseSparseMatrixOps.elemDiv(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.2628603919772923, 0.4832584151042525, 0.09732142956805734};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 2, 3};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.84134, 0.19162, 0.82944},
                {0.9047, 0.82863, 0.43141},
                {0.14977, 0.09193, 0.67211},
                {0.26796, 0.36743, 0.90342},
                {0.12804, 0.1211, 0.4879}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(Exception.class, ()->RealDenseSparseMatrixOps.elemDiv(A, finalB));
    }


    @Test
    void realSparseComplexDenseElemDivTest() {
        Complex128[][] bEntries;
        CMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.8290713116268288, 0.40772922471694184, 0.5345871783654969, 0.11148753066679329, 0.5148538501414794};
        aRowIndices = new int[]{2, 3, 3, 4, 4};
        aColIndices = new int[]{3, 1, 2, 0, 2};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.07522+0.44582i"), new Complex128("0.11863+0.44475i"), new Complex128("0.76382+0.99673i"), new Complex128("0.25827+0.27253i"), new Complex128("0.27447+0.86933i")},
                {new Complex128("0.60642+0.1579i"), new Complex128("0.00916+0.65482i"), new Complex128("0.21379+0.45586i"), new Complex128("0.57548+0.77109i"), new Complex128("0.86723+0.37164i")},
                {new Complex128("0.34112+0.85033i"), new Complex128("0.13631+0.07062i"), new Complex128("0.20108+0.96518i"), new Complex128("0.11581+0.18583i"), new Complex128("0.18305+0.70993i")},
                {new Complex128("0.865+0.86245i"), new Complex128("0.19539+0.33287i"), new Complex128("0.24912+0.37305i"), new Complex128("0.96705+0.28008i"), new Complex128("0.53257+0.20213i")},
                {new Complex128("0.40525+0.42922i"), new Complex128("0.36987+0.93569i"), new Complex128("0.59264+0.22915i"), new Complex128("0.62475+0.79414i"), new Complex128("0.96436+0.13689i")}};
        B = new CMatrix(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("2.0026125615957087-3.2134141466309516i"), new Complex128("0.5347454659906242-0.9110022174333338i"), new Complex128("0.6618212614627141-0.9910582112582911i"), new Complex128("0.1296581033313363-0.1373272081724273i"), new Complex128("0.7557564604588403-0.2922205603977849i")};
        expRowIndices = new int[]{2, 3, 3, 4, 4};
        expColIndices = new int[]{3, 1, 2, 0, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealComplexDenseCooMatOps.elemDiv(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.41089341705897764, 0.11457154954599069, 0.28303647128436815};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{3, 0, 4};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.98548+0.48213i"), new Complex128("0.83127+0.64831i"), new Complex128("0.82835+0.40503i"), new Complex128("0.28981+0.60902i"), new Complex128("0.95984+0.08349i")},
                {new Complex128("0.79999+0.11706i"), new Complex128("0.13687+0.54202i"), new Complex128("0.24906+0.26877i"), new Complex128("0.37736+0.59863i"), new Complex128("0.47597+0.40385i")},
                {new Complex128("0.36492+0.32894i"), new Complex128("0.7193+0.20864i"), new Complex128("0.93825+0.24218i"), new Complex128("0.13836+0.20312i"), new Complex128("0.988+0.64793i")}};
        B = new CMatrix(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.30963978929385033 - 0.4912011529175791i"), new Complex128("0.17321860123092025-0.1561397749887617i"), new Complex128("0.20032133981018468-0.1313706535457621i")};
        expRowIndices = new int[]{1, 2, 2};
        expColIndices = new int[]{3, 0, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealComplexDenseCooMatOps.elemDiv(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.12160331491060905, 0.6189704537780147, 0.641396804512954};
        aRowIndices = new int[]{0, 0, 2};
        aColIndices = new int[]{0, 4, 2};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.95006+0.77287i"), new Complex128("0.08063+0.53201i"), new Complex128("0.18516+0.10631i")},
                {new Complex128("0.85183+0.29688i"), new Complex128("0.34511+0.51713i"), new Complex128("0.70082+0.95626i")},
                {new Complex128("0.24499+0.41864i"), new Complex128("0.57098+0.12641i"), new Complex128("0.39938+0.24247i")},
                {new Complex128("0.10381+0.29217i"), new Complex128("0.39046+0.34261i"), new Complex128("0.68262+0.87753i")},
                {new Complex128("0.22802+0.4769i"), new Complex128("0.19684+0.11355i"), new Complex128("0.56578+0.29147i")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(Exception.class, ()-> RealComplexDenseCooMatOps.elemDiv(A, finalB));
    }
}
