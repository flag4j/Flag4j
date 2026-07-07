package org.flag4j.arrays.sparse.coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatrixOps;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseMatOps;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixAddSubTests {
    Shape aShape;
    double[] aEntries;
    int[] aRowIndices, aColIndices;
    CooMatrix A;

    Shape bShape, expShape;
    int[] bRowIndices, bColIndices, expRowIndices, expColIndices;

    @Test
    void realSparseRealSparseSubTest() {
        double[] bEntries;
        CooMatrix B;
        double[] expEntries;
        CooMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.7197167862026044, 0.8274741025737611, 0.44635157987459506, 0.9106722384653576, 0.11927378948791945};
        aRowIndices = new int[]{0, 1, 1, 2, 3};
        aColIndices = new int[]{0, 2, 4, 1, 4};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.47157415615670795, 0.18857508525597733, 0.35596697752675244, 0.007607366738096366, 0.7964396954007252};
        bRowIndices = new int[]{0, 1, 1, 2, 4};
        bColIndices = new int[]{4, 2, 3, 4, 1};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new double[]{0.7197167862026044, -0.47157415615670795, 0.6388990173177838, -0.35596697752675244, 0.44635157987459506, 0.9106722384653576, -0.007607366738096366, 0.11927378948791945, -0.7964396954007252};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 2, 2, 3, 4};
        expColIndices = new int[]{0, 4, 2, 3, 4, 1, 4, 4, 1};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.7234897223335167, 0.4569611853869002, 0.17516090675041163};
        aRowIndices = new int[]{2, 2, 2};
        aColIndices = new int[]{0, 1, 2};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.8234991857678657, 0.7336285038619206, 0.18279094243044902};
        bRowIndices = new int[]{1, 1, 2};
        bColIndices = new int[]{1, 2, 2};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new double[]{-0.8234991857678657, -0.7336285038619206, 0.7234897223335167, 0.4569611853869002, -0.007630035680037395};
        expRowIndices = new int[]{1, 1, 2, 2, 2};
        expColIndices = new int[]{1, 2, 0, 1, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.9180438547915879, 0.07235711960675528, 0.2657856824106707, 0.47790644045925923, 0.003070610781322758, 0.34202224059852215, 0.27031789435156306, 0.9499530477799731, 0.08321871287149174};
        aRowIndices = new int[]{0, 1, 3, 4, 6, 6, 6, 8, 8};
        aColIndices = new int[]{3, 0, 0, 1, 0, 2, 3, 1, 3};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.646214900340821, 0.046463495109891007, 0.052648659293113576};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 2, 2};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }


    @Test
    void realSparseComplexSparseSubTest() {
        Complex128[] bEntries;
        CooCMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.7540312001068363, 0.9428223651945967, 0.4650843608728411, 0.5206860140992323, 0.38382860306767685};
        aRowIndices = new int[]{0, 0, 1, 2, 3};
        aColIndices = new int[]{0, 2, 3, 0, 1};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new Complex128[]{new Complex128("0.36149192006867104+0.04319130594448195i"), new Complex128("0.4360933807209263+0.7955125874980291i"), new Complex128("0.8429639560748449+0.867498720512465i"), new Complex128("0.016258069078200377+0.4119653895482479i"), new Complex128("0.9945335908417812+0.42006061053116317i")};
        bRowIndices = new int[]{1, 2, 2, 3, 4};
        bColIndices = new int[]{4, 0, 4, 1, 3};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("0.7540312001068363"), new Complex128("0.9428223651945967"), new Complex128("0.4650843608728411"), new Complex128("-0.36149192006867104-0.04319130594448195i"), new Complex128("0.08459263337830603-0.7955125874980291i"), new Complex128("-0.8429639560748449-0.867498720512465i"), new Complex128("0.3675705339894765-0.4119653895482479i"), new Complex128("-0.9945335908417812-0.42006061053116317i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 4};
        expColIndices = new int[]{0, 2, 3, 4, 0, 4, 1, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealComplexSparseMatOps.sub(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.4602784916680196, 0.6975585135512675, 0.2503474806442201};
        aRowIndices = new int[]{0, 0, 2};
        aColIndices = new int[]{1, 3, 0};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new Complex128[]{new Complex128("0.02594980913117173+0.4379367199630513i"), new Complex128("0.8883365921139904+0.5585597123976269i"), new Complex128("0.5294253993237487+0.7455315910395643i")};
        bRowIndices = new int[]{0, 0, 1};
        bColIndices = new int[]{1, 3, 1};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.43432868253684787-0.4379367199630513i"), new Complex128("-0.19077807856272289-0.5585597123976269i"), new Complex128("-0.5294253993237487-0.7455315910395643i"), new Complex128("0.2503474806442201")};
        expRowIndices = new int[]{0, 0, 1, 2};
        expColIndices = new int[]{1, 3, 1, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealComplexSparseMatOps.sub(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.6270128869218086, 0.16240110664574325, 0.45932672845870137, 0.7474735763620591, 0.538473104946311, 0.7052133339436125, 0.2592123105328553, 0.3787513850108264, 0.6772122005094601};
        aRowIndices = new int[]{1, 2, 3, 3, 4, 4, 5, 5, 8};
        aColIndices = new int[]{4, 0, 0, 2, 2, 4, 1, 4, 3};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new Complex128[]{new Complex128("0.5868276372946869+0.2596163134750755i"), new Complex128("0.17707162901335272+0.6257920740208713i"), new Complex128("0.505831932260592+0.10185702848481692i")};
        bRowIndices = new int[]{0, 0, 2};
        bColIndices = new int[]{0, 4, 1};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix finalB = B;
        assertThrows(Exception.class, ()-> RealComplexSparseMatOps.sub(A, finalB));
    }


    @Test
    void realSparseRealDenseSubTest() {
        double[][] bEntries;
        Matrix B;
        double[][] expEntries;
        Matrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.6121384636921453, 0.8740186531145545, 0.029381038939396142, 0.018338236585478507, 0.8225690346245645};
        aRowIndices = new int[]{1, 1, 2, 3, 3};
        aColIndices = new int[]{1, 2, 1, 1, 3};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.01532, 0.57101, 0.9194, 0.62077, 0.16928},
                {0.36628, 0.44294, 0.77413, 0.27968, 0.14308},
                {0.11866, 0.19659, 0.20545, 0.95883, 0.19763},
                {0.89007, 0.78864, 0.66616, 0.88569, 0.21498},
                {0.08218, 0.4592, 0.39446, 0.76215, 0.09589}};
        B = new Matrix(bEntries);

        expEntries = new double[][]{
                {-0.01532, -0.57101, -0.9194, -0.62077, -0.16928},
                {-0.36628, 0.1691984636921453, 0.09988865311455453, -0.27968, -0.14308},
                {-0.11866, -0.16720896106060384, -0.20545, -0.95883, -0.19763},
                {-0.89007, -0.7703017634145215, -0.66616, -0.06312096537543543, -0.21498},
                {-0.08218, -0.4592, -0.39446, -0.76215, -0.09589}};
        exp = new Matrix(expEntries);

        assertEquals(exp, RealDenseSparseMatrixOps.sub(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.6453030157220373, 0.046510654054616074, 0.26315624640775137};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{0, 4, 2};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.60345, 0.38265, 0.93371, 0.8198, 0.58958},
                {0.27596, 0.04954, 0.97929, 0.95783, 0.12248},
                {0.29138, 0.92012, 0.0453, 0.19584, 0.80848}};
        B = new Matrix(bEntries);

        expEntries = new double[][]{
                {-0.60345, -0.38265, -0.93371, -0.8198, -0.58958},
                {0.3693430157220373, -0.04954, -0.97929, -0.95783, -0.07596934594538393},
                {-0.29138, -0.92012, 0.21785624640775136, -0.19584, -0.80848}};
        exp = new Matrix(expEntries);

        assertEquals(exp, RealDenseSparseMatrixOps.sub(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.5155800494513947, 0.546713180697451, 0.7643616937716915, 0.5419056469988012, 0.4939928805261391, 0.8767329019553655, 0.7469740304018805, 0.7002667865676956, 0.2988567558850965};
        aRowIndices = new int[]{0, 1, 1, 2, 4, 6, 8, 8, 8};
        aColIndices = new int[]{0, 1, 4, 1, 2, 1, 1, 2, 4};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.61128, 0.19402, 0.09326, 0.9641, 0.31154},
                {0.33576, 0.2249, 0.27908, 0.38607, 0.80359},
                {0.80024, 0.14538, 0.48963, 0.01383, 0.61448}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(Exception.class, ()->RealDenseSparseMatrixOps.sub(A, finalB));
    }


    @Test
    void realSparseComplexDenseSubTest() {
        Complex128[][] bEntries;
        CMatrix B;
        Complex128[][] expEntries;
        CMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.5179066536182447, 0.29699462002610133, 0.1384083495187035, 0.8642135522873594, 0.38365259331915025};
        aRowIndices = new int[]{2, 3, 3, 4, 4};
        aColIndices = new int[]{1, 0, 4, 0, 1};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.75637+0.83187i"), new Complex128("0.31084+0.34292i"), new Complex128("0.25448+0.05582i"), new Complex128("0.2331+0.79232i"), new Complex128("0.89203+0.76204i")},
                {new Complex128("0.66323+0.60234i"), new Complex128("0.77748+0.54931i"), new Complex128("0.88589+0.76489i"), new Complex128("0.2303+0.0718i"), new Complex128("0.46611+0.1212i")},
                {new Complex128("0.29809+0.28872i"), new Complex128("0.3664+0.65027i"), new Complex128("0.81276+0.94666i"), new Complex128("0.83843+0.8782i"), new Complex128("0.3193+0.76019i")},
                {new Complex128("0.21547+0.89237i"), new Complex128("0.95833+0.90932i"), new Complex128("0.51026+0.15335i"), new Complex128("0.86221+0.93154i"), new Complex128("0.70569+0.87415i")},
                {new Complex128("0.12112+0.58718i"), new Complex128("0.55609+0.2181i"), new Complex128("0.61738+0.18952i"), new Complex128("0.51587+0.89177i"), new Complex128("0.76042+0.31481i")}};
        B = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("-0.75637-0.83187i"), new Complex128("-0.31084-0.34292i"), new Complex128("-0.25448-0.05582i"), new Complex128("-0.2331-0.79232i"), new Complex128("-0.89203-0.76204i")},
                {new Complex128("-0.66323-0.60234i"), new Complex128("-0.77748-0.54931i"), new Complex128("-0.88589-0.76489i"), new Complex128("-0.2303-0.0718i"), new Complex128("-0.46611-0.1212i")},
                {new Complex128("-0.29809-0.28872i"), new Complex128("0.15150665361824472-0.65027i"), new Complex128("-0.81276-0.94666i"), new Complex128("-0.83843-0.8782i"), new Complex128("-0.3193-0.76019i")},
                {new Complex128("0.08152462002610134-0.89237i"), new Complex128("-0.95833-0.90932i"), new Complex128("-0.51026-0.15335i"), new Complex128("-0.86221-0.93154i"), new Complex128("-0.5672816504812965-0.87415i")},
                {new Complex128("0.7430935522873594-0.58718i"), new Complex128("-0.17243740668084973-0.2181i"), new Complex128("-0.61738-0.18952i"), new Complex128("-0.51587-0.89177i"), new Complex128("-0.76042-0.31481i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, RealFieldDenseCooMatrixOps.sub(A, B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.11496605515101443, 0.781677500733315, 0.8839774286026595};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 3, 0};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.52624+0.2508i"), new Complex128("0.83235+0.29734i"), new Complex128("0.92588+0.90113i"), new Complex128("0.38166+0.20651i"), new Complex128("0.78835+0.65337i")},
                {new Complex128("0.73068+0.81259i"), new Complex128("0.79852+0.93415i"), new Complex128("0.13249+0.03554i"), new Complex128("0.31218+0.74991i"), new Complex128("0.03226+0.56384i")},
                {new Complex128("0.53334+0.51873i"), new Complex128("0.45876+0.354i"), new Complex128("0.53354+0.6198i"), new Complex128("0.76748+0.68158i"), new Complex128("0.87113+0.69653i")}};
        B = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("-0.52624-0.2508i"), new Complex128("-0.83235-0.29734i"), new Complex128("-0.8109139448489856-0.90113i"), new Complex128("0.400017500733315-0.20651i"), new Complex128("-0.78835-0.65337i")},
                {new Complex128("0.1532974286026595-0.81259i"), new Complex128("-0.79852-0.93415i"), new Complex128("-0.13249-0.03554i"), new Complex128("-0.31218-0.74991i"), new Complex128("-0.03226-0.56384i")},
                {new Complex128("-0.53334-0.51873i"), new Complex128("-0.45876-0.354i"), new Complex128("-0.53354-0.6198i"), new Complex128("-0.76748-0.68158i"), new Complex128("-0.87113-0.69653i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, RealFieldDenseCooMatrixOps.sub(A, B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.12806056946267386, 0.3115401448576318, 0.22374771721866793, 0.17735659189845965, 0.3894332597306345, 0.6439604056373636, 0.843316756247472, 0.4266067850722429, 0.9094096898230055};
        aRowIndices = new int[]{0, 3, 3, 4, 4, 4, 5, 6, 7};
        aColIndices = new int[]{0, 1, 4, 0, 3, 4, 2, 2, 1};
        A = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.96146+0.25417i"), new Complex128("0.38568+0.95976i"), new Complex128("0.35263+0.51317i"), new Complex128("0.16861+0.32575i"), new Complex128("0.45787+0.60226i")},
                {new Complex128("0.75246+0.26293i"), new Complex128("0.4652+0.18691i"), new Complex128("0.30874+0.838i"), new Complex128("0.85089+0.18647i"), new Complex128("0.55567+0.09379i")},
                {new Complex128("0.81861+0.37771i"), new Complex128("0.01698+0.29668i"), new Complex128("0.05255+0.14805i"), new Complex128("0.57575+0.25943i"), new Complex128("0.92083+0.02978i")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(Exception.class, ()-> RealFieldDenseCooMatrixOps.sub(A, finalB));
    }


    @Test
    void addDoubleTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double b;

        double[][] expEntries;
        Matrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.07877, 0.56112, 0.45943};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.5029;

        expEntries = new double[][]{
                {0.58167, 0, 0, 0, 0},
                {0, 1.06402, 0, 0, 0},
                {0, 0.96233, 0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.09339, 0.45538, 0.19015, 0.99002, 0.36725};
        aRowIndices = new int[]{0, 1, 6, 7, 7};
        aColIndices = new int[]{2, 19, 11, 0, 7};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.92145;

        expEntries = new double[][]{
                {0, 0, 1.01484, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.37683, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.1116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1.91147, 0, 0, 0, 0, 0, 0, 1.2887, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.9748, 0.11903, 0.49796, 0.49884};
        aRowIndices = new int[]{0, 1, 4, 4};
        aColIndices = new int[]{0, 2, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.47886;

        expEntries = new double[][]{
                {1.45366, 0, 0},
                {0, 0, 0.59789},
                {0, 0, 0},
                {0, 0, 0},
                {0, 0.97682, 0.9777}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.04699, 0.05726, 0.65267, 0.8001};
        aRowIndices = new int[]{2, 2, 3, 4};
        aColIndices = new int[]{1, 2, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.69321;

        expEntries = new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {0, 0.7402, 0.75047},
                {1.34588, 0, 0},
                {1.4933100000000001, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.47584, 0.91786, 0.12886, 0.75403};
        aRowIndices = new int[]{0, 2, 3, 4};
        aColIndices = new int[]{0, 1, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.27254;

        expEntries = new double[][]{
                {0.74838, 0, 0},
                {0, 0, 0},
                {0, 1.1904, 0},
                {0, 0.4014, 0},
                {1.02657, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.70614, 0.09468, 0.58868, 0.48552};
        aRowIndices = new int[]{1, 2, 2, 4};
        aColIndices = new int[]{1, 0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.38668;

        expEntries = new double[][]{
                {0, 0, 0},
                {0, 1.0928200000000001, 0},
                {0.48136, 0.97536, 0},
                {0, 0, 0},
                {0, 0.8722000000000001, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));
    }


    @Test
    void addComplex128Test() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Complex128 b;

        Complex128[][] expEntries;
        CMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.41845, 0.24416, 0.94638};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{3, 3, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.92786, 0.0899);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("1.34631+0.0899i"), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("1.17202+0.0899i"), Complex128.ZERO},
                {Complex128.ZERO, new Complex128("1.87424+0.0899i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.7284, 0.05728, 0.19851, 0.37527, 0.38768};
        aRowIndices = new int[]{5, 8, 9, 10, 10};
        aColIndices = new int[]{0, 15, 17, 5, 14};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(-0.29236, 0.52224);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.43604000000000004+0.52224i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("-0.23508+0.52224i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("-0.09385000000000002+0.52224i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("0.08290999999999998+0.52224i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("0.09532000000000002+0.52224i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.98254, 0.02491, 0.0166, 0.07118};
        aRowIndices = new int[]{1, 3, 4, 4};
        aColIndices = new int[]{1, 1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(-0.07159, -0.79401);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.9109499999999999-0.79401i"), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("-0.04668-0.79401i"), Complex128.ZERO},
                {new Complex128("-0.05499-0.79401i"), new Complex128("-0.0004100000000000076-0.79401i"), Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.92351, 0.33742, 0.87826, 0.25944};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{1, 0, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(-0.37554, 0.38333);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.5479700000000001+0.38333i"), Complex128.ZERO},
                {new Complex128("-0.03811999999999999+0.38333i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.5027200000000001+0.38333i"), Complex128.ZERO},
                {new Complex128("-0.11609999999999998+0.38333i"), Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.69952, 0.51199, 0.67603, 0.90057};
        aRowIndices = new int[]{0, 2, 4, 4};
        aColIndices = new int[]{0, 1, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.14523, 0.5006);

        expEntries = new Complex128[][]{
                {new Complex128("0.84475+0.5006i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.6572199999999999+0.5006i"), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.82126+0.5006i"), Complex128.ZERO, new Complex128("1.0458+0.5006i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.8617, 0.47043, 0.35871, 0.76007};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{1, 0, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.2298, -0.66038);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, new Complex128("1.0915-0.66038i"), Complex128.ZERO},
                {new Complex128("0.70023-0.66038i"), Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.58851-0.66038i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.98987-0.66038i"), Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.add(b));
    }


    @Test
    void subDoubleTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double b;

        double[][] expEntries;
        Matrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.57509, 0.1909, 0.18736};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{3, 1, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.50067;

        expEntries = new double[][]{
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0.07442000000000004, 0},
                {0, -0.30977, 0, 0, -0.31331}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.02362, 0.20739, 0.42558, 0.12155, 0.39385};
        aRowIndices = new int[]{2, 3, 7, 9, 10};
        aColIndices = new int[]{10, 5, 18, 9, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.32174;

        expEntries = new double[][]{
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.29812000000000005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, -0.11435000000000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.10383999999999999, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, -0.20019000000000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0.07210999999999995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.85959, 0.17781, 0.40395, 0.31349};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{0, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.23653;

        expEntries = new double[][]{
                {0.62306, 0, 0},
                {-0.058719999999999994, 0, 0},
                {0, 0, 0},
                {0.16741999999999999, 0, 0},
                {0, 0, 0.07696}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.70185, 0.91872, 0.23836, 0.88307};
        aRowIndices = new int[]{2, 2, 3, 4};
        aColIndices = new int[]{0, 1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.19768;

        expEntries = new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {0.50417, 0.72104, 0},
                {0, 0, 0.040679999999999994},
                {0, 0, 0.68539}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.58303, 0.1641, 0.96525, 0.256};
        aRowIndices = new int[]{0, 1, 4, 4};
        aColIndices = new int[]{0, 2, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.84434;

        expEntries = new double[][]{
                {-0.26130999999999993, 0, 0},
                {0, 0, -0.68024},
                {0, 0, 0},
                {0, 0, 0},
                {0.12091000000000007, 0, -0.58834}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.21466, 0.80583, 0.62783, 0.79117};
        aRowIndices = new int[]{0, 1, 1, 3};
        aColIndices = new int[]{0, 0, 2, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = 0.67201;

        expEntries = new double[][]{
                {-0.45735000000000003, 0, 0},
                {0.13382000000000005, 0, -0.04418},
                {0, 0, 0},
                {0, 0.11916000000000004, 0},
                {0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));
    }


    @Test
    void subComplexTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Complex128 b;

        Complex128[][] expEntries;
        CMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.79786, 0.90141, 0.10994};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{4, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.39038, -0.26221);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("0.40748+0.26221i")},
                {Complex128.ZERO, new Complex128("0.5110300000000001+0.26221i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, new Complex128("-0.28044+0.26221i"), Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.0756, 0.86767, 0.6857, 0.82702, 0.20509};
        aRowIndices = new int[]{1, 1, 3, 5, 8};
        aColIndices = new int[]{8, 20, 4, 12, 15};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(-0.40542, 0.6929);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("0.48102-0.6929i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("1.27309-0.6929i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("1.09112-0.6929i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("1.23244-0.6929i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128("0.61051-0.6929i"), Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.44113, 0.46052, 0.36516, 0.85082};
        aRowIndices = new int[]{0, 3, 3, 3};
        aColIndices = new int[]{1, 0, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.12447, 0.18706);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, new Complex128("0.31666000000000005-0.18706i"), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.33604999999999996-0.18706i"), new Complex128("0.24069-0.18706i"), new Complex128("0.72635-0.18706i")},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.1362, 0.90253, 0.94696, 0.26031};
        aRowIndices = new int[]{0, 2, 3, 4};
        aColIndices = new int[]{1, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.24715, -0.94979);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, new Complex128("-0.11095000000000002+0.94979i"), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.6553800000000001+0.94979i"), Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.69981+0.94979i"), Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, new Complex128("0.013159999999999977+0.94979i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.86481, 0.70241, 0.38859, 0.88652};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(0.7504, 0.1731);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, new Complex128("0.11441000000000001-0.1731i"), Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, new Complex128("-0.04798999999999998-0.1731i"), new Complex128("-0.36180999999999996-0.1731i")},
                {Complex128.ZERO, Complex128.ZERO, new Complex128("0.13612000000000002-0.1731i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.54717, 0.71262, 0.7463, 0.11747};
        aRowIndices = new int[]{2, 2, 3, 4};
        aColIndices = new int[]{0, 1, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Complex128(-0.10197, 0.35104);

        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128("0.64914-0.35104i"), new Complex128("0.81459-0.35104i"), Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.84827-0.35104i"), Complex128.ZERO},
                {Complex128.ZERO, new Complex128("0.21944000000000002-0.35104i"), Complex128.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp.toCoo(), a.sub(b));
    }
}
