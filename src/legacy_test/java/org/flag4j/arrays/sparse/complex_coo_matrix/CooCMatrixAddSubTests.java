package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseCooMatOps;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixAddSubTests {
    Shape aShape;
    Complex128[] aEntries;
    int[] aRowIndices, aColIndices;
    CooCMatrix A;
    Shape bShape, expShape;
    int[] bRowIndices, bColIndices, expRowIndices, expColIndices;

    @Test
    void complexSparseRealSparseSubTest() {
        double[] bEntries;
        CooMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.16907660311363415+0.6444051570772598i"), new Complex128("0.2685636944700537+0.277720286141387i"), new Complex128("0.5371330041634212+0.8363534476695493i"), new Complex128("0.7584479587013249+0.1811320912553298i"), new Complex128("0.24728154455051643+0.8252113996910411i")};
        aRowIndices = new int[]{0, 1, 2, 3, 4};
        aColIndices = new int[]{2, 0, 2, 1, 1};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.19188475719480103, 0.4647475368649855, 0.29775891146914446, 0.3211976671246123, 0.17140653454433696};
        bRowIndices = new int[]{0, 2, 2, 2, 4};
        bColIndices = new int[]{3, 1, 2, 4, 0};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("0.16907660311363415+0.6444051570772598i"), new Complex128("-0.19188475719480103"), new Complex128("0.2685636944700537+0.277720286141387i"), new Complex128("-0.4647475368649855"), new Complex128("0.23937409269427679+0.8363534476695493i"), new Complex128("-0.3211976671246123"), new Complex128("0.7584479587013249+0.1811320912553298i"), new Complex128("-0.17140653454433696"), new Complex128("0.24728154455051643+0.8252113996910411i")};
        expRowIndices = new int[]{0, 0, 1, 2, 2, 2, 3, 4, 4};
        expColIndices = new int[]{2, 3, 0, 1, 2, 4, 1, 0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.04597036221184858+0.27569095742387373i"), new Complex128("0.5253705770713939+0.8652830858877205i"), new Complex128("0.22466851327235537+0.48755081428740665i")};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{0, 3, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.672094469418651, 0.004058773106305424, 0.8859863043452371};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 3, 0};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("-0.672094469418651"), new Complex128("0.04597036221184858+0.27569095742387373i"), new Complex128("0.5213118039650885+0.8652830858877205i"), new Complex128("-0.8859863043452371"), new Complex128("0.22466851327235537+0.48755081428740665i")};
        expRowIndices = new int[]{0, 1, 1, 2, 2};
        expColIndices = new int[]{0, 0, 3, 0, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new Complex128[]{new Complex128("0.7773495524901912+0.9670609434569698i"), new Complex128("0.8143837489702114+0.6264702725849607i"), new Complex128("0.8703120541209147+0.3764970320612634i"), new Complex128("0.10844960651850521+0.10089333349129115i"), new Complex128("0.030157519141515654+0.049893464833488554i"), new Complex128("0.9105679984224888+0.8197605473728841i"), new Complex128("0.4716928344990372+0.14839364769347763i"), new Complex128("0.060272082265836135+0.6869053336232293i"), new Complex128("0.8562050927896339+0.9202699550583396i")};
        aRowIndices = new int[]{0, 1, 3, 3, 3, 4, 6, 7, 8};
        aColIndices = new int[]{2, 1, 1, 3, 4, 4, 0, 1, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.536356449591203, 0.30915669933809475, 0.8877352025892702};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{2, 3, 1};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }


    @Test
    void complexSparseComplexSparseSubTest() {
        Complex128[] bEntries;
        CooCMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.6910245510195184+0.9077474303843585i"), new Complex128("0.6823193196673835+0.9686934950350857i"), new Complex128("0.7016095528990995+0.003958926929736095i"), new Complex128("0.7245077304133463+0.5737158012835359i"), new Complex128("0.2691984171075853+0.67074179839617i")};
        aRowIndices = new int[]{1, 1, 4, 4, 4};
        aColIndices = new int[]{3, 4, 0, 1, 2};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new Complex128[]{new Complex128("0.750831194736842+0.014946996944480873i"), new Complex128("0.4056010561730603+0.8747315110548242i"), new Complex128("0.36210393820761777+0.3658252932429157i"), new Complex128("0.6939774995467313+0.8474628180314803i"), new Complex128("0.7736819449768878+0.549693340140008i")};
        bRowIndices = new int[]{0, 0, 1, 3, 4};
        bColIndices = new int[]{1, 4, 0, 0, 4};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("-0.750831194736842-0.014946996944480873i"), new Complex128("-0.4056010561730603-0.8747315110548242i"), new Complex128("-0.36210393820761777-0.3658252932429157i"), new Complex128("0.6910245510195184+0.9077474303843585i"), new Complex128("0.6823193196673835+0.9686934950350857i"), new Complex128("-0.6939774995467313-0.8474628180314803i"), new Complex128("0.7016095528990995+0.003958926929736095i"), new Complex128("0.7245077304133463+0.5737158012835359i"), new Complex128("0.2691984171075853+0.67074179839617i"), new Complex128("-0.7736819449768878-0.549693340140008i")};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 3, 4, 4, 4, 4};
        expColIndices = new int[]{1, 4, 0, 3, 4, 0, 0, 1, 2, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.16309615141403278+0.2516389353539852i"), new Complex128("0.5450934022190043+0.8214327358769891i"), new Complex128("0.9755707249007003+0.24732294705340552i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 3, 0};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new Complex128[]{new Complex128("0.5974064951428303+0.25567994696631857i"), new Complex128("0.09729813080087268+0.25124075582765404i"), new Complex128("0.5183760577380403+0.015612286010666998i")};
        bRowIndices = new int[]{0, 2, 2};
        bColIndices = new int[]{2, 0, 3};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("-0.4343103437287975-0.0040410116123333895i"), new Complex128("0.5450934022190043+0.8214327358769891i"), new Complex128("0.9755707249007003+0.24732294705340552i"), new Complex128("-0.09729813080087268-0.25124075582765404i"), new Complex128("-0.5183760577380403-0.015612286010666998i")};
        expRowIndices = new int[]{0, 0, 1, 2, 2};
        expColIndices = new int[]{2, 3, 0, 0, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new Complex128[]{new Complex128("0.37615963269009955+0.46648638371680784i"), new Complex128("0.4389228048293522+0.49919855367244914i"), new Complex128("0.3641970005860071+0.21364944579375622i"), new Complex128("0.6768728860174499+0.11293844550895626i"), new Complex128("0.10555547511624319+0.9974250471254223i"), new Complex128("0.48695465079671685+0.22713039083257602i"), new Complex128("0.24233326574055303+0.6509670389804286i"), new Complex128("0.8285754970538919+0.6734183712117148i"), new Complex128("0.9793874081511778+0.9016976212187522i")};
        aRowIndices = new int[]{0, 1, 2, 3, 5, 5, 8, 8, 8};
        aColIndices = new int[]{0, 1, 1, 4, 0, 2, 2, 3, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new Complex128[]{new Complex128("0.49283950580425695+0.5978997646247968i"), new Complex128("0.4798061204000892+0.6805791759797895i"), new Complex128("0.3296042362931275+0.40587377179411577i")};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 3, 4};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }


    @Test
    void complexSparseRealDenseSubTest() {
        double[][] bEntries;
        Matrix B;
        Complex128[][] expEntries;
        CMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.8424268253354057+0.21146667805661112i"), new Complex128("0.2304861070248274+0.10979159677958883i"), new Complex128("0.3056659052818361+0.6921336614085905i"), new Complex128("0.5195959873475188+0.41206170034352796i"), new Complex128("0.03582224948045554+0.17683330297161015i")};
        aRowIndices = new int[]{0, 0, 0, 2, 4};
        aColIndices = new int[]{0, 1, 3, 4, 0};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.4769, 0.55535, 0.74008, 0.4981, 0.60883},
                {0.42825, 0.24429, 0.71298, 0.44908, 0.11728},
                {0.60191, 0.47307, 0.02511, 0.81523, 0.15956},
                {0.5857, 0.85135, 0.89822, 0.82077, 0.64763},
                {0.10706, 0.06424, 0.70094, 0.27209, 0.71348}};
        B = new Matrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("0.3655268253354057+0.21146667805661112i"), new Complex128("-0.3248638929751726+0.10979159677958883i"), new Complex128("-0.74008"), new Complex128("-0.19243409471816386+0.6921336614085905i"), new Complex128("-0.60883")},
                {new Complex128("-0.42825"), new Complex128("-0.24429"), new Complex128("-0.71298"), new Complex128("-0.44908"), new Complex128("-0.11728")},
                {new Complex128("-0.60191"), new Complex128("-0.47307"), new Complex128("-0.02511"), new Complex128("-0.81523"), new Complex128("0.36003598734751874+0.41206170034352796i")},
                {new Complex128("-0.5857"), new Complex128("-0.85135"), new Complex128("-0.89822"), new Complex128("-0.82077"), new Complex128("-0.64763")},
                {new Complex128("-0.07123775051954447+0.17683330297161015i"), new Complex128("-0.06424"), new Complex128("-0.70094"), new Complex128("-0.27209"), new Complex128("-0.71348")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, RealComplexDenseCooMatOps.sub(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.3159084198324813+0.7906484152681891i"), new Complex128("0.942719674202597+0.961516632226317i"), new Complex128("0.39271189079390745+0.011647415847762632i")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{3, 2, 2};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.88463, 0.00213, 0.95681, 0.10726, 0.57349},
                {0.23503, 0.83957, 0.28726, 0.74247, 0.04663},
                {0.00192, 0.96867, 0.35521, 0.485, 0.51841}};
        B = new Matrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("-0.88463"), new Complex128("-0.00213"), new Complex128("-0.95681"), new Complex128("0.2086484198324813+0.7906484152681891i"), new Complex128("-0.57349")},
                {new Complex128("-0.23503"), new Complex128("-0.83957"), new Complex128("0.6554596742025969+0.961516632226317i"), new Complex128("-0.74247"), new Complex128("-0.04663")},
                {new Complex128("-0.00192"), new Complex128("-0.96867"), new Complex128("0.03750189079390742+0.011647415847762632i"), new Complex128("-0.485"), new Complex128("-0.51841")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, RealComplexDenseCooMatOps.sub(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new Complex128[]{new Complex128("0.9952829598305839+0.8050493064451766i"), new Complex128("0.709025797231945+0.5951590091259715i"), new Complex128("0.06758149251369971+0.08300547071660325i"), new Complex128("0.7448367789778203+0.9129414768429843i"), new Complex128("0.8328088447829204+0.17370844362000515i"), new Complex128("0.7298735049313018+0.5143837509697955i"), new Complex128("0.9841202093082072+0.8351781796300937i"), new Complex128("0.8865663154413503+0.40606422345406057i"), new Complex128("0.9921185011384177+0.332737970659427i")};
        aRowIndices = new int[]{2, 3, 4, 5, 5, 5, 6, 8, 8};
        aColIndices = new int[]{1, 3, 4, 0, 2, 3, 1, 1, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.22054, 0.57962, 0.11093, 0.97711, 0.05861},
                {0.79201, 0.24955, 0.41488, 0.55444, 0.19949},
                {0.70022, 0.49334, 0.50787, 0.63341, 0.33605}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(Exception.class, ()-> RealComplexDenseCooMatOps.sub(A, finalB));
    }


    @Test
    void complexSparseComplexDenseSubTest() {
        Complex128[][] bEntries;
        CMatrix B;
        Complex128[][] expEntries;
        CMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.21432657368028107+0.02285719095217109i"), new Complex128("0.5121521258714489+0.06946412645312638i"), new Complex128("0.7335832759673879+0.8072923006459559i"), new Complex128("0.9995745132631427+0.8601025880139203i"), new Complex128("0.7433339540509724+0.991526108868919i")};
        aRowIndices = new int[]{0, 1, 2, 2, 2};
        aColIndices = new int[]{4, 4, 1, 2, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.15344+0.85327i"), new Complex128("0.15998+0.24614i"), new Complex128("0.02935+0.61796i"), new Complex128("0.34734+0.34036i"), new Complex128("0.72925+0.62188i")},
                {new Complex128("0.26567+0.87823i"), new Complex128("0.49959+0.06567i"), new Complex128("0.91521+0.21248i"), new Complex128("0.49163+0.40293i"), new Complex128("0.66516+0.68606i")},
                {new Complex128("0.43925+0.13461i"), new Complex128("0.43469+0.27304i"), new Complex128("0.31367+0.71719i"), new Complex128("0.45697+0.10558i"), new Complex128("0.6393+0.77987i")},
                {new Complex128("0.9078+0.06041i"), new Complex128("0.52368+0.26074i"), new Complex128("0.72075+0.83193i"), new Complex128("0.58351+0.02258i"), new Complex128("0.8177+0.85571i")},
                {new Complex128("0.32123+0.17085i"), new Complex128("0.72278+0.87353i"), new Complex128("0.85108+0.44586i"), new Complex128("0.87101+0.73296i"), new Complex128("0.45473+0.7i")}};
        B = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("-0.15344-0.85327i"), new Complex128("-0.15998-0.24614i"), new Complex128("-0.02935-0.61796i"), new Complex128("-0.34734-0.34036i"), new Complex128("-0.5149234263197189-0.5990228090478289i")},
                {new Complex128("-0.26567-0.87823i"), new Complex128("-0.49959-0.06567i"), new Complex128("-0.91521-0.21248i"), new Complex128("-0.49163-0.40293i"), new Complex128("-0.1530078741285511-0.6165958735468736i")},
                {new Complex128("-0.43925-0.13461i"), new Complex128("0.2988932759673879+0.5342523006459559i"), new Complex128("0.6859045132631427+0.14291258801392026i"), new Complex128("0.28636395405097237+0.885946108868919i"), new Complex128("-0.6393-0.77987i")},
                {new Complex128("-0.9078-0.06041i"), new Complex128("-0.52368-0.26074i"), new Complex128("-0.72075-0.83193i"), new Complex128("-0.58351-0.02258i"), new Complex128("-0.8177-0.85571i")},
                {new Complex128("-0.32123-0.17085i"), new Complex128("-0.72278-0.87353i"), new Complex128("-0.85108-0.44586i"), new Complex128("-0.87101-0.73296i"), new Complex128("-0.45473-0.7i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DenseCooFieldMatrixOps.sub(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.8641201436785124+0.464077542772543i"), new Complex128("0.051767898713606764+0.8739906137569583i"), new Complex128("0.3479464354902717+0.5164113253872332i")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{3, 3, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.64542+0.78504i"), new Complex128("0.78863+0.16445i"), new Complex128("0.81881+0.55626i"), new Complex128("0.76823+0.46202i"), new Complex128("0.56919+0.58119i")},
                {new Complex128("0.62407+0.05079i"), new Complex128("0.58854+0.25756i"), new Complex128("0.80829+0.29632i"), new Complex128("0.87106+0.27112i"), new Complex128("0.63053+0.53858i")},
                {new Complex128("0.16092+0.84309i"), new Complex128("0.06968+0.77818i"), new Complex128("0.8741+0.25259i"), new Complex128("0.55985+0.88026i"), new Complex128("0.10439+0.94986i")}};
        B = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("-0.64542-0.78504i"), new Complex128("-0.78863-0.16445i"), new Complex128("-0.81881-0.55626i"), new Complex128("0.09589014367851245+0.0020575427725429973i"), new Complex128("-0.56919-0.58119i")},
                {new Complex128("-0.62407-0.05079i"), new Complex128("-0.58854-0.25756i"), new Complex128("-0.80829-0.29632i"), new Complex128("-0.8192921012863932+0.6028706137569583i"), new Complex128("-0.63053-0.53858i")},
                {new Complex128("-0.16092-0.84309i"), new Complex128("-0.06968-0.77818i"), new Complex128("-0.8741-0.25259i"), new Complex128("-0.55985-0.88026i"), new Complex128("0.24355643549027173-0.4334486746127668i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DenseCooFieldMatrixOps.sub(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new Complex128[]{new Complex128("0.6907434894473184+0.5031295442002623i"), new Complex128("0.03229906329481269+0.17744734837949105i"), new Complex128("0.628911680222165+0.6716533189511541i"), new Complex128("0.8176512894236148+0.06108845413688435i"), new Complex128("0.3694547169018447+0.08400778038547219i"), new Complex128("0.34659776908318607+0.13962028870070875i"), new Complex128("0.7411889157136219+0.6236878630390096i"), new Complex128("0.6082841811530453+0.7702079661274945i"), new Complex128("0.9176201416728813+0.7402652203934886i")};
        aRowIndices = new int[]{0, 0, 2, 4, 5, 5, 6, 7, 7};
        aColIndices = new int[]{2, 3, 4, 4, 1, 2, 2, 0, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.71274+0.45198i"), new Complex128("0.40301+0.44065i"), new Complex128("0.0448+0.16388i"), new Complex128("0.68599+0.92123i"), new Complex128("0.44005+0.67785i")},
                {new Complex128("0.052+0.61173i"), new Complex128("0.35935+0.33899i"), new Complex128("0.0226+0.27895i"), new Complex128("0.26702+0.27556i"), new Complex128("0.91464+0.23937i")},
                {new Complex128("0.46262+0.3401i"), new Complex128("0.73727+0.4888i"), new Complex128("0.22095+0.7225i"), new Complex128("0.75802+0.38349i"), new Complex128("0.80627+0.77942i")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(Exception.class, ()-> DenseCooFieldMatrixOps.sub(A, finalB));
    }
}
