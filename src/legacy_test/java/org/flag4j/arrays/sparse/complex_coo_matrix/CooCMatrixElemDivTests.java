package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatrixOps;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixElemDivTests {

    Shape aShape, expShape;
    Complex128[] aEntries;
    CooCMatrix A;
    int[] aRowIndices, aColIndices, expRowIndices, expColIndices;

    @Test
    void complexSparseRealDenseElemDivTest() {
        double[][] bEntries;
        Matrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.0766368471796478+0.9492280625872803i"), new Complex128("0.1930391038649386+0.7617047445283123i"), new Complex128("0.9067549010330563+0.31336061871125553i"), new Complex128("0.21893877466645717+0.5152634601707257i"), new Complex128("0.5764595328602934+0.39285513632570546i")};
        aRowIndices = new int[]{1, 1, 2, 2, 4};
        aColIndices = new int[]{3, 4, 2, 4, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.23814, 0.33623, 0.01234, 0.0361, 0.69305},
                {0.05945, 0.93939, 0.9087, 0.88784, 0.2656},
                {0.28246, 0.03969, 0.22864, 0.63789, 0.75131},
                {0.71335, 0.53126, 0.37861, 0.12162, 0.81109},
                {0.94294, 0.42372, 0.3333, 0.80281, 0.54418}};
        B = new Matrix(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("0.08631830868134778 + 1.0691431593387102i"), new Complex128("0.7268038549131725+2.8678642489770794i"), new Complex128("3.9658629331396793+1.3705415443984235i"), new Complex128("0.2914093711869364+0.6858200478773417i"), new Complex128("0.7180522575208248 + 0.4893500782572532i")};
        expRowIndices = new int[]{1, 1, 2, 2, 4};
        expColIndices = new int[]{3, 4, 2, 4, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealFieldDenseCooMatrixOps.elemDiv(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.9720212972219383+0.18426402442963263i"), new Complex128("0.5764899656357376+0.31327386474221364i"), new Complex128("0.7410673521266785+0.11934630071637853i")};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{2, 4, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.85568, 0.5295, 0.24867, 0.12305, 0.26644},
                {0.1553, 0.12725, 0.94087, 0.94976, 0.49794},
                {0.50322, 0.00905, 0.20847, 0.20631, 0.80573}};
        B = new Matrix(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("1.033109034427645+0.19584429775594145i"), new Complex128("1.1577498606975491+0.6291397854002764i"), new Complex128("0.9197465058104807+0.14812195241132703i")};
        expRowIndices = new int[]{1, 1, 2};
        expColIndices = new int[]{2, 4, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealFieldDenseCooMatrixOps.elemDiv(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.2167893175598996+0.2996212001672518i"), new Complex128("0.1511818245619858+0.20927498440208603i"), new Complex128("0.20209823912923397+0.8997366291135686i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 0, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.49778, 0.77635, 0.99512},
                {0.62501, 0.5288, 0.23735},
                {0.48371, 0.44386, 0.18931},
                {0.76458, 0.46405, 0.59421},
                {0.11186, 0.73465, 0.40008}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(Exception.class, ()-> RealFieldDenseCooMatrixOps.elemDiv(A, finalB));
    }


    @Test
    void complexSparseComplexDenseElemDivTest() {
        Complex128[][] bEntries;
        CMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.3713399183921321+0.2732583086126841i"), new Complex128("0.8097640641001572+0.47645109131560925i"), new Complex128("0.08879023807148412+0.9698316824688531i"), new Complex128("0.7172788370662367+0.08708074327944348i"), new Complex128("0.7462626232525152+0.9494035470157722i")};
        aRowIndices = new int[]{0, 0, 2, 2, 3};
        aColIndices = new int[]{3, 4, 1, 4, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.35248+0.5024i"), new Complex128("0.02582+0.58262i"), new Complex128("0.59393+0.23783i"), new Complex128("0.17298+0.47328i"), new Complex128("0.44915+0.51353i")},
                {new Complex128("0.75648+0.8851i"), new Complex128("0.06829+0.57647i"), new Complex128("0.58224+0.23523i"), new Complex128("0.58503+0.23666i"), new Complex128("0.60039+0.62148i")},
                {new Complex128("0.61849+0.46678i"), new Complex128("0.01722+0.36464i"), new Complex128("0.24588+0.26521i"), new Complex128("0.6891+0.4162i"), new Complex128("0.34601+0.02556i")},
                {new Complex128("0.76476+0.60393i"), new Complex128("0.19538+0.9464i"), new Complex128("0.21713+0.11611i"), new Complex128("0.05808+0.43596i"), new Complex128("0.886+0.27363i")},
                {new Complex128("0.33299+0.75751i"), new Complex128("0.23273+0.95777i"), new Complex128("0.55063+0.04563i"), new Complex128("0.60138+0.57274i"), new Complex128("0.55587+0.06641i")}};
        B = new CMatrix(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("0.7623073843560689 - 0.505992197105771i"), new Complex128("1.3070771264452088-0.43364627725214056i"), new Complex128("2.6652516607566423-0.11763548835359462i"), new Complex128("2.0802398157395676+0.09800240914753947i"), new Complex128("2.363824038562478-1.3968522871199343i")};
        expRowIndices = new int[]{0, 0, 2, 2, 3};
        expColIndices = new int[]{3, 4, 1, 4, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DenseCooFieldMatrixOps.elemDiv(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.43891010689776977+0.8643182182497601i"), new Complex128("0.2344331678075624+0.1339568190472047i"), new Complex128("0.2972763774902456+0.49046124106584676i")};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{2, 0, 2};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.5042+0.27989i"), new Complex128("0.00354+0.66911i"), new Complex128("0.60815+0.39143i"), new Complex128("0.22379+0.58241i"), new Complex128("0.32315+0.07285i")},
                {new Complex128("0.50014+0.2802i"), new Complex128("0.05302+0.58335i"), new Complex128("0.8714+0.1005i"), new Complex128("0.87381+0.91641i"), new Complex128("0.86372+0.90357i")},
                {new Complex128("0.87835+0.89733i"), new Complex128("0.17624+0.22454i"), new Complex128("0.41889+0.85406i"), new Complex128("0.31796+0.83282i"), new Complex128("0.37757+0.9195i")}};
        B = new CMatrix(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.6099648314642674+0.9215248481611215i"), new Complex128("0.20683571771116144-0.05879555476353598i"), new Complex128("0.6005271815432809-0.053534349359038766i")};
        expRowIndices = new int[]{1, 2, 2};
        expColIndices = new int[]{2, 0, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DenseCooFieldMatrixOps.elemDiv(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.5864427355413825+0.6325572833610345i"), new Complex128("0.015557949933594273+0.3938021787260496i"), new Complex128("0.04797168412999786+0.8363898618137131i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 3, 0};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.15042+0.26227i"), new Complex128("0.6688+0.97353i"), new Complex128("0.63097+0.51755i")},
                {new Complex128("0.06431+0.70709i"), new Complex128("0.00367+0.79614i"), new Complex128("0.44468+0.00696i")},
                {new Complex128("0.53063+0.76949i"), new Complex128("0.79508+0.16329i"), new Complex128("0.62538+0.31855i")},
                {new Complex128("0.86006+0.83223i"), new Complex128("0.47554+0.06541i"), new Complex128("0.28913+0.16755i")},
                {new Complex128("0.38596+0.9124i"), new Complex128("0.92621+0.74368i"), new Complex128("0.15599+0.76305i")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(Exception.class, ()-> DenseCooFieldMatrixOps.elemDiv(A, finalB));
    }
}
