package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixGetRowColTests {

    @Test
    void getRowTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape expShape;
        int[] expIndices;
        Complex128[] expEntries;
        CooCVector exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.85014+0.13396i"), new Complex128("0.42107+0.86061i"), new Complex128("0.03743+0.07013i")};
        aRowIndices = new int[]{1, 1, 3};
        aColIndices = new int[]{1, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new Complex128[]{new Complex128("0.79679+0.33414i"), new Complex128("0.58492+0.62096i"), new Complex128("0.36384+0.32372i"), new Complex128("0.1865+0.4848i"), new Complex128("0.17452+0.82274i")};
        aRowIndices = new int[]{7, 11, 14, 15, 17};
        aColIndices = new int[]{0, 10, 2, 9, 5};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(18));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new Complex128[]{new Complex128("0.11974+0.29942i"), new Complex128("0.33674+0.72905i"), new Complex128("0.44925+0.98181i"), new Complex128("0.63092+0.24232i"), new Complex128("0.4517+0.99098i"), new Complex128("0.95548+0.92438i"), new Complex128("0.32395+0.54536i"), new Complex128("0.10559+0.60299i"), new Complex128("0.25432+0.20652i"), new Complex128("0.84965+0.78219i"), new Complex128("0.89495+0.15426i"), new Complex128("0.82645+0.86562i"), new Complex128("0.9018+0.25203i"), new Complex128("0.84499+0.62506i"), new Complex128("0.39673+0.95609i"), new Complex128("0.01925+0.94541i"), new Complex128("0.07557+0.23793i"), new Complex128("0.7463+0.00383i"), new Complex128("0.68323+0.07234i"), new Complex128("0.02258+0.93409i"), new Complex128("0.78448+0.56676i"), new Complex128("0.50248+0.54428i"), new Complex128("0.3104+0.39744i"), new Complex128("0.02423+0.31276i"), new Complex128("0.64779+0.63546i"), new Complex128("0.74091+0.01208i"), new Complex128("0.94424+0.50957i"), new Complex128("0.5501+0.64393i"), new Complex128("0.80174+0.76936i"), new Complex128("0.14829+0.14918i"), new Complex128("0.30923+0.78332i"), new Complex128("0.9671+0.00538i"), new Complex128("0.18883+0.4137i"), new Complex128("0.83491+0.1725i"), new Complex128("0.58367+0.24299i"), new Complex128("0.60332+0.88367i"), new Complex128("0.44552+0.81197i"), new Complex128("0.98848+0.99708i"), new Complex128("0.42437+0.98984i"), new Complex128("0.84689+0.49237i"), new Complex128("0.33991+0.38412i"), new Complex128("0.15283+0.7487i"), new Complex128("0.78185+0.88555i"), new Complex128("0.00577+0.00208i"), new Complex128("0.14539+0.91764i"), new Complex128("0.48055+0.1103i"), new Complex128("0.06489+0.32284i"), new Complex128("0.95589+0.64668i"), new Complex128("0.58155+0.46647i"), new Complex128("0.83394+0.03572i"), new Complex128("0.38546+0.78337i"), new Complex128("0.72633+0.27578i"), new Complex128("0.21335+0.37489i"), new Complex128("0.15618+0.68374i"), new Complex128("0.71501+0.72455i"), new Complex128("0.49728+0.45328i"), new Complex128("0.06926+0.55557i"), new Complex128("0.31163+0.49307i"), new Complex128("0.99262+0.60487i"), new Complex128("0.56775+0.96894i"), new Complex128("0.79921+0.86901i"), new Complex128("0.15873+0.78661i"), new Complex128("0.35638+0.27835i"), new Complex128("0.13726+0.92973i"), new Complex128("0.2732+0.46305i"), new Complex128("0.59421+0.53173i"), new Complex128("0.22635+0.88244i"), new Complex128("0.3523+0.6287i"), new Complex128("0.50599+0.18976i"), new Complex128("0.69124+0.24652i"), new Complex128("0.41526+0.30357i"), new Complex128("0.2517+0.49469i"), new Complex128("0.4421+0.82912i"), new Complex128("0.37641+0.15445i"), new Complex128("0.43514+0.65402i"), new Complex128("0.44325+0.46175i"), new Complex128("0.27009+0.95715i"), new Complex128("0.60988+0.7563i"), new Complex128("0.92206+0.78866i"), new Complex128("0.50325+0.45975i"), new Complex128("0.7036+0.78527i"), new Complex128("0.67046+0.96845i"), new Complex128("0.34831+0.00696i"), new Complex128("0.39499+0.18733i"), new Complex128("0.28952+0.7921i"), new Complex128("0.09863+0.34144i"), new Complex128("0.5973+0.06066i"), new Complex128("0.44424+0.85083i"), new Complex128("0.40567+0.01632i"), new Complex128("0.91006+0.6258i"), new Complex128("0.85823+0.09209i"), new Complex128("0.01048+0.38648i"), new Complex128("0.90538+0.41257i"), new Complex128("0.83084+0.35436i"), new Complex128("0.02735+0.49404i"), new Complex128("0.74524+0.31438i"), new Complex128("0.32781+0.56013i"), new Complex128("0.08377+0.26885i")};
        aRowIndices = new int[]{11, 20, 60, 82, 86, 93, 127, 132, 143, 146, 161, 172, 184, 200, 201, 205, 224, 225, 231, 233, 247, 249, 250, 259, 263, 264, 266, 280, 293, 300, 304, 315, 322, 323, 351, 367, 368, 370, 371, 397, 409, 413, 418, 439, 440, 442, 453, 455, 457, 464, 478, 492, 506, 512, 522, 529, 537, 563, 567, 567, 571, 576, 596, 623, 626, 632, 651, 669, 673, 679, 701, 710, 714, 729, 736, 742, 750, 761, 772, 773, 791, 810, 823, 841, 848, 856, 867, 870, 879, 894, 897, 913, 937, 938, 940, 966, 967, 972};
        aColIndices = new int[]{1, 4, 2, 0, 4, 1, 3, 2, 1, 1, 3, 0, 0, 3, 4, 1, 0, 1, 4, 0, 1, 2, 1, 4, 3, 1, 4, 2, 1, 1, 0, 1, 1, 4, 4, 0, 0, 0, 3, 0, 1, 2, 2, 2, 0, 1, 1, 3, 4, 3, 3, 2, 3, 4, 1, 4, 0, 3, 1, 3, 2, 0, 1, 2, 4, 0, 4, 4, 2, 0, 2, 0, 4, 2, 3, 0, 1, 3, 3, 3, 2, 1, 2, 4, 4, 1, 0, 2, 4, 4, 1, 0, 0, 2, 2, 3, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.38665+0.32236i"), new Complex128("0.94511+0.28497i"), new Complex128("0.01232+0.50662i"), new Complex128("0.82279+0.29668i")};
        aRowIndices = new int[]{0, 1, 1, 1};
        aColIndices = new int[]{0, 0, 2, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getRow(-1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.20673+0.73121i"), new Complex128("0.97694+0.41429i"), new Complex128("0.82297+0.58838i"), new Complex128("0.92729+0.25731i")};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{2, 4, 0, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getRow(3));
    }


    @Test
    void getRowSliceTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape expShape;
        int[] expIndices;
        Complex128[] expEntries;
        CooCVector exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.35378+0.77266i"), new Complex128("0.63132+0.30232i"), new Complex128("0.65128+0.57601i")};
        aRowIndices = new int[]{3, 3, 4};
        aColIndices = new int[]{0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(2);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(2, 1, 3));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new Complex128[]{new Complex128("0.77877+0.24558i"), new Complex128("0.089+0.01851i"), new Complex128("0.77293+0.68228i"), new Complex128("0.46986+0.69681i"), new Complex128("0.71046+0.60961i")};
        aRowIndices = new int[]{5, 8, 13, 19, 22};
        aColIndices = new int[]{7, 9, 3, 9, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(7);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(18, 0, 7));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new Complex128[]{new Complex128("0.1197+0.94022i"), new Complex128("0.3949+0.09479i"), new Complex128("0.78573+0.64632i"), new Complex128("0.08456+0.66652i"), new Complex128("0.30287+0.21925i"), new Complex128("0.93975+0.15341i"), new Complex128("0.3586+0.84386i"), new Complex128("0.41191+0.41938i"), new Complex128("0.35808+0.49009i"), new Complex128("0.52045+0.41904i"), new Complex128("0.67332+0.66356i"), new Complex128("0.40089+0.77476i"), new Complex128("0.58731+0.87191i"), new Complex128("0.29405+0.19581i"), new Complex128("0.66139+0.12067i"), new Complex128("0.88585+0.93897i"), new Complex128("0.20441+0.6388i"), new Complex128("0.21772+0.49946i"), new Complex128("0.82521+0.71675i"), new Complex128("0.07633+0.09192i"), new Complex128("0.25744+0.06878i"), new Complex128("0.11043+0.919i"), new Complex128("0.49275+0.36692i"), new Complex128("0.53135+0.36618i"), new Complex128("0.28591+0.51644i"), new Complex128("0.71491+0.69877i"), new Complex128("0.40118+0.86539i"), new Complex128("0.59451+0.5814i"), new Complex128("0.73749+0.04487i"), new Complex128("0.65396+0.36245i"), new Complex128("0.3453+0.54983i"), new Complex128("0.87021+0.67375i"), new Complex128("0.66758+0.63047i"), new Complex128("0.1711+0.84147i"), new Complex128("0.08805+0.81387i"), new Complex128("0.33511+0.91619i"), new Complex128("0.92696+0.99792i"), new Complex128("0.31654+0.57498i"), new Complex128("0.81819+0.75673i"), new Complex128("0.58752+0.5038i"), new Complex128("0.53634+0.22678i"), new Complex128("0.64952+0.56675i"), new Complex128("0.23476+0.82129i"), new Complex128("0.4719+0.93008i"), new Complex128("0.92368+0.74603i"), new Complex128("0.91491+0.28891i"), new Complex128("0.3979+0.64265i"), new Complex128("0.70253+0.91635i"), new Complex128("0.06055+0.8594i"), new Complex128("0.59036+0.95835i"), new Complex128("0.07784+0.08281i"), new Complex128("0.72404+0.48835i"), new Complex128("0.88943+0.71433i"), new Complex128("0.47356+0.83234i"), new Complex128("0.32389+0.55687i"), new Complex128("0.57991+0.92723i"), new Complex128("0.24254+0.98131i"), new Complex128("0.36353+0.02801i"), new Complex128("0.288+0.65081i"), new Complex128("0.69278+0.96687i"), new Complex128("0.00599+0.16409i"), new Complex128("0.58035+0.8071i"), new Complex128("0.48673+0.61306i"), new Complex128("0.23278+0.01698i"), new Complex128("0.26711+0.45153i"), new Complex128("0.63749+0.87085i"), new Complex128("0.26591+0.3407i"), new Complex128("0.20613+0.65707i"), new Complex128("0.37655+0.53529i"), new Complex128("0.37078+0.95481i"), new Complex128("0.22896+0.99965i"), new Complex128("0.34101+0.74184i"), new Complex128("0.78109+0.66099i"), new Complex128("0.54804+0.72688i"), new Complex128("0.81518+0.16435i"), new Complex128("0.69293+0.03331i"), new Complex128("0.10422+0.42116i"), new Complex128("0.12137+0.04954i"), new Complex128("0.71186+0.73485i"), new Complex128("0.7353+0.63799i"), new Complex128("0.78497+0.69427i"), new Complex128("0.73468+0.80776i"), new Complex128("0.43855+0.23145i"), new Complex128("0.6132+0.36626i"), new Complex128("0.73995+0.42844i"), new Complex128("0.95834+0.50414i"), new Complex128("0.40703+0.38778i"), new Complex128("0.03179+0.62317i"), new Complex128("0.59098+0.10735i"), new Complex128("0.98355+0.96119i"), new Complex128("0.18479+0.701i"), new Complex128("0.86234+0.7115i"), new Complex128("0.66956+0.62245i"), new Complex128("0.57683+0.27578i"), new Complex128("0.29046+0.83643i"), new Complex128("0.32007+0.22106i"), new Complex128("0.08457+0.13594i"), new Complex128("0.825+0.59432i")};
        aRowIndices = new int[]{5, 11, 25, 30, 42, 70, 120, 121, 139, 141, 155, 155, 162, 174, 210, 212, 229, 236, 242, 246, 248, 287, 288, 293, 314, 320, 320, 322, 332, 333, 339, 342, 367, 370, 374, 375, 426, 437, 452, 476, 484, 500, 504, 516, 520, 525, 526, 537, 538, 542, 564, 569, 592, 599, 600, 613, 613, 637, 643, 653, 659, 671, 702, 711, 717, 754, 782, 783, 784, 802, 806, 807, 809, 819, 820, 823, 827, 834, 844, 847, 849, 849, 857, 860, 866, 879, 883, 892, 914, 925, 945, 949, 955, 963, 965, 975, 981, 994};
        aColIndices = new int[]{1, 4, 2, 0, 0, 2, 0, 2, 0, 4, 1, 3, 4, 4, 4, 0, 1, 0, 0, 1, 3, 1, 2, 0, 2, 0, 3, 1, 1, 4, 4, 3, 3, 4, 3, 0, 0, 0, 2, 2, 0, 2, 4, 2, 2, 3, 4, 4, 2, 2, 2, 0, 3, 4, 3, 3, 4, 0, 4, 4, 0, 3, 2, 3, 1, 0, 1, 4, 4, 2, 2, 1, 3, 0, 1, 3, 0, 2, 4, 2, 1, 2, 0, 2, 4, 2, 0, 4, 2, 1, 1, 3, 4, 1, 3, 2, 4, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getRow(0, 1, 4));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.12396+0.86038i"), new Complex128("0.34522+0.98801i"), new Complex128("0.80877+0.71381i"), new Complex128("0.20895+0.4778i")};
        aRowIndices = new int[]{0, 1, 1, 1};
        aColIndices = new int[]{1, 2, 3, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getRow(-1, 1, 3));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.69148+0.16422i"), new Complex128("0.06132+0.18029i"), new Complex128("0.13537+0.12001i"), new Complex128("0.31402+0.31333i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{0, 1, 1, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getRow(3, 1, 3));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.79463+0.80497i"), new Complex128("0.00641+0.67837i"), new Complex128("0.94995+0.63211i"), new Complex128("0.36218+0.88425i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{2, 4, 4, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.getRow(2, -1, 3));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.84672+0.12739i"), new Complex128("0.25641+0.30972i"), new Complex128("0.7404+0.04103i"), new Complex128("0.9449+0.43956i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{2, 1, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.getRow(2, 1, 6));
    }


    @Test
    void getColTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape expShape;
        int[] expIndices;
        Complex128[] expEntries;
        CooCVector exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.52573+0.02579i"), new Complex128("0.57563+0.65813i"), new Complex128("0.95458+0.46474i")};
        aRowIndices = new int[]{0, 0, 2};
        aColIndices = new int[]{0, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3);
        expEntries = new Complex128[]{new Complex128("0.57563+0.65813i"), new Complex128("0.95458+0.46474i")};
        expIndices = new int[]{0, 2};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(2));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new Complex128[]{new Complex128("0.62677+0.5383i"), new Complex128("0.5423+0.04262i"), new Complex128("0.14555+0.63608i"), new Complex128("0.62864+0.62878i"), new Complex128("0.46009+0.41949i")};
        aRowIndices = new int[]{0, 1, 7, 9, 10};
        aColIndices = new int[]{19, 2, 14, 21, 17};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(18));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new Complex128[]{new Complex128("0.81941+0.83623i"), new Complex128("0.43526+0.89961i"), new Complex128("0.94661+0.7718i"), new Complex128("0.06616+0.75347i"), new Complex128("0.62433+0.77103i"), new Complex128("0.43355+0.56655i"), new Complex128("0.76684+0.6886i"), new Complex128("0.99477+0.71806i"), new Complex128("0.53282+0.99906i"), new Complex128("0.46773+0.25194i"), new Complex128("0.45387+0.93992i"), new Complex128("0.40758+0.32458i"), new Complex128("0.64513+0.24518i"), new Complex128("0.23103+0.88614i"), new Complex128("0.68931+0.49549i"), new Complex128("0.2527+0.75961i"), new Complex128("0.51978+0.19159i"), new Complex128("0.24262+0.18176i"), new Complex128("0.09952+0.48402i"), new Complex128("0.50078+0.96738i"), new Complex128("0.41476+0.36692i"), new Complex128("0.22889+0.5233i"), new Complex128("0.34546+0.92331i"), new Complex128("0.94987+0.92101i"), new Complex128("0.12872+0.19649i"), new Complex128("0.33031+0.34002i"), new Complex128("0.11559+0.99033i"), new Complex128("0.71143+0.55397i"), new Complex128("0.85522+0.61865i"), new Complex128("0.36824+0.34697i"), new Complex128("0.05205+0.27399i"), new Complex128("0.95124+0.58186i"), new Complex128("0.31644+0.63731i"), new Complex128("0.29784+0.13279i"), new Complex128("0.72311+0.35388i"), new Complex128("0.85525+0.33403i"), new Complex128("0.6992+0.56404i"), new Complex128("0.44082+0.38637i"), new Complex128("0.05131+0.38769i"), new Complex128("0.55726+0.33684i"), new Complex128("0.86177+0.17227i"), new Complex128("0.74883+0.30594i"), new Complex128("0.81294+0.67142i"), new Complex128("0.70094+0.02111i"), new Complex128("0.75377+0.46927i"), new Complex128("0.89763+0.96551i"), new Complex128("0.58239+0.03134i"), new Complex128("0.11629+0.11796i"), new Complex128("0.63411+0.75399i"), new Complex128("0.76202+0.09171i"), new Complex128("0.45672+0.05116i"), new Complex128("0.17444+0.94845i"), new Complex128("0.28709+0.47079i"), new Complex128("0.33794+0.69366i"), new Complex128("0.78622+0.55435i"), new Complex128("0.85967+0.00645i"), new Complex128("0.14846+0.11643i"), new Complex128("0.17426+0.47954i"), new Complex128("0.92358+0.50308i"), new Complex128("0.07435+0.11918i"), new Complex128("0.81383+0.11174i"), new Complex128("0.25964+0.20103i"), new Complex128("0.67321+0.24806i"), new Complex128("0.94147+0.57974i"), new Complex128("0.54147+0.34612i"), new Complex128("0.05394+0.30398i"), new Complex128("0.03086+0.14663i"), new Complex128("0.67111+0.30903i"), new Complex128("0.34601+0.42421i"), new Complex128("0.36428+0.3424i"), new Complex128("0.81316+0.53793i"), new Complex128("0.95982+0.89398i"), new Complex128("0.7595+0.66258i"), new Complex128("0.94777+0.84973i"), new Complex128("0.66903+0.09519i"), new Complex128("0.71345+0.403i"), new Complex128("0.6934+0.46659i"), new Complex128("0.14266+0.26808i"), new Complex128("0.3643+0.26018i"), new Complex128("0.98397+0.30296i"), new Complex128("0.06829+0.98988i"), new Complex128("0.30133+0.67498i"), new Complex128("0.77603+0.6338i"), new Complex128("0.30349+0.38981i"), new Complex128("0.73994+0.73514i"), new Complex128("0.33015+0.73564i"), new Complex128("0.343+0.3776i"), new Complex128("0.1338+0.79356i"), new Complex128("0.64898+0.09421i"), new Complex128("0.9954+0.04248i"), new Complex128("0.51493+0.1487i"), new Complex128("0.08156+0.36688i"), new Complex128("0.68079+0.51004i"), new Complex128("0.93257+0.5522i"), new Complex128("0.99435+0.24752i"), new Complex128("0.61442+0.53134i"), new Complex128("0.82084+0.70099i"), new Complex128("0.88014+0.18377i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{24, 32, 124, 173, 188, 204, 221, 232, 292, 346, 355, 379, 390, 576, 679, 681, 717, 732, 865, 921, 922, 924, 955, 119, 144, 171, 207, 213, 299, 320, 629, 704, 758, 834, 848, 925, 31, 103, 126, 138, 158, 267, 303, 322, 342, 372, 384, 524, 582, 630, 636, 651, 657, 683, 746, 788, 829, 897, 967, 49, 104, 119, 122, 217, 324, 365, 458, 477, 573, 595, 605, 626, 699, 701, 915, 926, 929, 45, 172, 213, 258, 332, 350, 375, 382, 401, 455, 501, 505, 518, 577, 637, 645, 737, 853, 857, 869, 954};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(0));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.35032+0.66588i"), new Complex128("0.06781+0.09802i"), new Complex128("0.46817+0.24215i"), new Complex128("0.05259+0.75862i")};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{2, 0, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.getCol(-1));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.25736+0.649i"), new Complex128("0.76755+0.41935i"), new Complex128("0.33088+0.55851i"), new Complex128("0.7935+0.69879i")};
        aRowIndices = new int[]{0, 2, 3, 4};
        aColIndices = new int[]{0, 2, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        CooCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.getCol(3));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.8627+0.19761i"), new Complex128("0.84461+0.84861i"), new Complex128("0.58419+0.08294i"), new Complex128("0.88528+0.83323i")};
        aRowIndices = new int[]{0, 2, 3, 3};
        aColIndices = new int[]{2, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new Complex128[]{new Complex128("0.8627+0.19761i"), new Complex128("0.88528+0.83323i")};
        expIndices = new int[]{0, 3};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(2));

        // ---------------------  sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.80594+0.65649i"), new Complex128("0.47635+0.58139i"), new Complex128("0.28509+0.47459i"), new Complex128("0.6398+0.58163i")};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 0, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5);
        expEntries = new Complex128[]{};
        expIndices = new int[]{};
        exp = new CooCVector(expShape.getSize(0), expEntries, expIndices);

        assertEquals(exp, a.getCol(2));
    }
}
